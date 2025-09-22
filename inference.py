import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from .utils.init_setup import dist_init, setup_model
import logging
from .utils import logging_config
from .dataloaders.random_dataset import StreamDataset, ExternalFrameDataset
from omegaconf import OmegaConf
import matplotlib.pyplot as plt  # Added for depth visualization
import cv2
import torch.nn.functional as F
import queue
import threading
import time

class FlashDepthProcessor:
    def __init__(self, config_path="configs/flashdepth", url=None, stream_mode=False, save_depth_png=False, save_frame=False, max_frames=None, run_dir=None, external_frame_mode=False):
        self.cfg = None
        self.process_dict = None
        self.run_dir = run_dir
        # Indicates whether the processor has finished/stopped processing
        self.stopped = False
        self.pred = None  # To store the latest depth map (only depth, no frame)
        
        # External frame mode support
        self.external_frame_mode = external_frame_mode
        self.frame_queue = queue.Queue(maxsize=2) if external_frame_mode else None
        self.model = None  # Store model for external frame processing
        self.model_ready = threading.Event() if external_frame_mode else None
        
        self._load_config(config_path)
        self._setup()
        self.stream_mode = stream_mode
        self.save_depth_png = save_depth_png
        self.max_frames = max_frames
        self.save_frame = save_frame  # Changed from None to save_frame parameter 
        self.url = url
        
        # Build eval_args for model forward calls (needed for both regular and external frame modes)
        self.eval_args = {
            'stream': getattr(self.cfg.eval, 'stream', False),
            'save_depth_png': getattr(self.cfg.eval, 'save_depth_png', False),
            'save_depth_npy': getattr(self.cfg.eval, 'save_depth_npy', False),
            'save_vis_map': getattr(self.cfg.eval, 'save_vis_map', False),
            'save_frame_png': getattr(self.cfg.eval, 'save_frame_png', False),
            'out_video': getattr(self.cfg.eval, 'out_video', False),
            'out_mp4': getattr(self.cfg.eval, 'out_mp4', None),
            'use_mamba': getattr(self.cfg.model, 'use_mamba', True),
            'resolution': getattr(self.cfg.eval, 'save_res', None),
            'print_time': True,
            'use_all_frames': True,
            'run_dir': self.run_dir
        }
        
    def _load_config(self, config_path):
        """Load configuration using OmegaConf."""
        
        if os.path.exists(config_path):
            self.cfg = OmegaConf.load(config_path)
            logging.info(f"Loaded config from: {config_path}")
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
    def _setup(self):
        # initialize distributed/process
        self.process_dict = dist_init()
        logging_config.configure_logging()

        # Determine where to create stream_{X} directories.
        # If self.run_dir was provided by caller, treat it as the parent directory
        # under which to create the per-run `stream_{N}` folders. Otherwise
        # fall back to the original behavior of creating `result/stream_{N}`
        if self.run_dir:
            parent = os.path.abspath(self.run_dir)
            os.makedirs(parent, exist_ok=True)
        else:
            parent = os.path.abspath(os.path.join(os.getcwd(), 'result'))
            os.makedirs(parent, exist_ok=True)

        # Prefer a plain 'stream' folder if available (first run). If it exists,
        # start enumerating 'stream_1', 'stream_2', ... to avoid collisions.
        base_candidate = os.path.join(parent, 'stream')
        if not os.path.exists(base_candidate):
            os.makedirs(base_candidate)
            self.run_dir = base_candidate
        else:
            idx = 1
            while True:
                candidate = os.path.join(parent, f'stream_{idx}')
                if not os.path.exists(candidate):
                    os.makedirs(candidate)
                    self.run_dir = candidate
                    break
                idx += 1

        # Add file logger to run_dir so logs are saved to stream_{N}/run.log
        fh = logging.FileHandler(os.path.join(self.run_dir, 'run.log'))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')
        fh.setFormatter(formatter)
        logging.getLogger().addHandler(fh)

        # Set config_dir for compatibility
        if hasattr(self.cfg, 'config_dir'):
            self.cfg.config_dir = os.getcwd()
    
    def process_external_frame(self, frame):
        """接收外部帧进行处理"""
        if not self.external_frame_mode or self.frame_queue is None:
            return
        
        try:
            # 非阻塞插入，如果队列满了就丢弃旧帧
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            # 丢弃最老的帧，插入新帧
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
            except queue.Empty:
                pass
        
    @torch.no_grad()
    def run_inference(self):
        """Run stream inference (only inference logic)."""
        cfg = self.cfg
        # logging.info('[cfg]:{}'.format(cfg))
        process_dict = self.process_dict
        run_dir = self.run_dir

        # Ensure inference mode
        cfg.inference = True

        model, train_step = setup_model(cfg, process_dict)
        model.eval()
        self.model = model  # 存储model供外部帧处理使用
        logging.info(f"Inference from step {train_step}")

        if getattr(cfg.eval, 'compile', False):
            model = torch.compile(model)
            self.model = model

        logging.info('[inference Run_dir]:{}'.format(run_dir))
        
        # 如果是外部帧模式，通知模型已准备好
        if self.external_frame_mode and self.model_ready:
            self.model_ready.set()
            logging.info("Model ready for external frame processing")

        # 构建 eval_args，与常规流模式保持一致
        eval_args = self.eval_args.copy()
        eval_args.update({
            'run_dir': run_dir
        })

        # 外部帧模式：使用 ExternalFrameDataset，逻辑与流模式一致，但数据来自队列
        if self.external_frame_mode:
            logging.info("[ExternalFrameMode] Using ExternalFrameDataset with queue-driven frames")
            dataset = ExternalFrameDataset(resolution=getattr(self.cfg.dataset, 'resolution', None), crop_type=None)
            test_idx = 0
            # MAIN LOOP (mirrors stream path)
            try:
                while not self.stopped:
                    # 读取外部帧（阻塞短超时）
                    try:
                        frame = self.frame_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    # 更新数据集帧并取出 batch
                    dataset.update_frame(frame)
                    batch = dataset[0]

                    # 获取 batch 与原始帧
                    if isinstance(batch, dict):
                        batch_tensor = batch['batch']
                        original_frame = batch.get('frame', None)
                        # 将原始帧转换成 torch.Tensor 以复用后续插值逻辑
                        if original_frame is not None and not isinstance(original_frame, torch.Tensor):
                            original_frame = torch.from_numpy(original_frame)
                    else:
                        batch_tensor = batch
                        original_frame = None

                    # 设备迁移
                    model_device = next(model.parameters()).device
                    if isinstance(batch_tensor, torch.Tensor) and batch_tensor.device != model_device:
                        batch_tensor = batch_tensor.to(model_device)

                    # 规范为 [B, T=1, C, H, W] 以满足 stream 推理要求
                    try:
                        if isinstance(batch_tensor, torch.Tensor):
                            if batch_tensor.dim() == 4:
                                # [B, C, H, W] -> [B, 1, C, H, W]
                                batch_tensor = batch_tensor.unsqueeze(1)
                            elif batch_tensor.dim() == 5:
                                # [B, T, C, H, W] -> 取 T=1
                                if batch_tensor.shape[1] != 1:
                                    batch_tensor = batch_tensor[:, :1]
                            else:
                                logging.warning(f"[ExternalFrameMode] Unexpected batch dim={batch_tensor.dim()}, model may fail")
                            batch_tensor = batch_tensor.contiguous()
                    except Exception as _e:
                        logging.warning(f"[ExternalFrameMode] Failed to normalize batch dims: {_e}")

                    # 推理
                    try:
                        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                            depth_pred = model(
                                batch_tensor,
                                gif_path=f'{run_dir}/{os.path.basename(getattr(cfg, "config_dir", "cfg").rstrip("/"))}_{train_step}_ext_{test_idx}.gif',
                                **eval_args
                            )
                            self.pred = depth_pred

                            test_idx += 1
                            if self.max_frames is not None and test_idx >= int(self.max_frames):
                                logging.info(f'[ExternalFrameMode] Reached max_frames {self.max_frames}, stopping')
                                break
                    except Exception as e:
                        logging.warning(f"[ExternalFrameMode] Error processing frame {test_idx}: {e}")
                        continue
            except Exception as e:
                logging.warning(f"[ExternalFrameMode] Exception in run_inference loop: {e}")
            finally:
                self.stopped = True
                logging.info(f"[FlashDepthProcessor] run_inference finished (external mode), stopped={self.stopped}")
            return
        

        # StreamMode
        if self.url is None:
            stream_addr = getattr(cfg.eval, 'stream_url', None) or getattr(cfg.eval, 'url', None)
            if stream_addr is None:
                raise ValueError('Stream address required: set cfg.eval.stream_url or cfg.eval.url')
        else:
            stream_addr = self.url

        warmup = getattr(cfg.eval, 'stream_warmup_frames', 5)
        if self.max_frames is not None:
            try:
                self.max_frames = int(self.max_frames)
                if self.max_frames <= 0:
                    self.max_frames = None
            except Exception:
                logging.warning(f"Invalid max_frames value: {self.max_frames!r}. Treating as unlimited.")
                self.max_frames = None
        logging.info(f"Stream Max frames set to: {self.max_frames}")

        logging.info('Using URL input: {}'.format(stream_addr))
        dataset = StreamDataset(stream_url=stream_addr, resolution=cfg.dataset.resolution, warmup_frames=warmup)
        test_dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False, drop_last=False)

        # Pre-check flags for efficiency
        if not self.stream_mode:
            self.stream_mode = eval_args.get('stream', False)
        logging.info(f"[Stream Mode]: {self.stream_mode}")

        if not self.save_depth_png:
            self.save_depth_png = eval_args.get('save_depth_png', False)
        logging.info(f"[Save Depth PNG]: {self.save_depth_png}")

        if not self.save_frame:
            self.save_frame = eval_args.get('save_frame_png', False)
        logging.info(f"[Save Original Frame]: {self.save_frame}")

        if self.max_frames is None:
            pbar = tqdm(test_dataloader)
        else:
            pbar = tqdm(test_dataloader, total=self.max_frames)

        # MAIN LOOP
        try:
            for test_idx, batch in enumerate(pbar):

                if (self.max_frames is not None) and (test_idx >= int(self.max_frames)):
                    logging.info(f'Reached max_frames {self.max_frames}, stopping')
                    break

                # Get batch and original frame    
                if isinstance(batch, dict): # _get_item_: return dict(batch=img.unsqueeze(0), frame=origin_frame)
                    batch_tensor = batch['batch']
                    original_frame = batch['frame']  # Original frame if available
                    original_frame = original_frame.squeeze(0)  # 在GPU上展开为 [H, W, C]
                else:
                    batch_tensor = batch

                model_device = next(model.parameters()).device
                if isinstance(batch_tensor, torch.Tensor) and batch_tensor.device != model_device:
                    batch_tensor = batch_tensor.to(model_device)
                
                if self.save_depth_png:
                    png_dir = f'{run_dir}/saved_PNGs'
                    os.makedirs(png_dir, exist_ok=True)

                try:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        if self.stream_mode:
                            # Stream mode: return single depth (H, W)
                            depth_pred = model(
                                batch_tensor,
                                gif_path=f'{run_dir}/{os.path.basename(getattr(cfg, "config_dir", "cfg").rstrip("/"))}_{train_step}_{test_idx}.gif',
                                **eval_args
                            )
                            # Only store depth map, no frame data
                            self.pred = depth_pred  # Store only depth map (Tensor, shape [H, W])
                            
                            # Optional: save depth and frame for debugging
                            if self.save_depth_png:
                                depth_np = depth_pred.float().cpu().numpy()
                                plt.imsave(f'{png_dir}/{test_idx}_depth.png', depth_np, cmap='inferno')
                            
                            if self.save_frame and original_frame is not None:
                                # Resize original_frame to ensure resolution is multiple of 14 for consistency with depth
                                h, w = original_frame.shape[:2]
                                target_h = ((h + 13) // 14) * 14
                                target_w = ((w + 13) // 14) * 14
                                if h != target_h or w != target_w:
                                    original_frame = F.interpolate(
                                        original_frame.permute(2, 0, 1).unsqueeze(0),
                                        size=(target_h, target_w),
                                        mode='bilinear',
                                        align_corners=False
                                    ).squeeze(0).permute(1, 2, 0)
                                
                                frame_np = original_frame.cpu().numpy()  # 已经展开，直接转换为numpy
                                frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)  # 转换为 RGB 颜色空间
                                plt.imsave(f'{png_dir}/{test_idx}_frame.png', frame_np)
                except Exception as e:
                    logging.warning(f"Error processing frame {test_idx}: {e}")
                    continue
        except Exception as e:
            logging.warning(f"Exception in run_inference loop: {e}")
        finally:
            try:
                pbar.close()
            except Exception:
                pass
            # Mark processor as stopped so external callers (e.g., Pano2stereo) can react
            self.stopped = True
            logging.info(f"[FlashDepthProcessor] run_inference finished, stopped={self.stopped}")
    
    def _run_external_frame_inference(self):
        """Deprecated: external frame processing handled inside run_inference()"""
        logging.info("_run_external_frame_inference is deprecated; external mode handled in run_inference")

    def cleanup(self):
        # Only destroy process group if it was initialized (distributed mode)
        if hasattr(dist, '_default_pg') and dist._default_pg is not None:
            dist.destroy_process_group()


if __name__ == '__main__':
    # Testing
    # Create processor with default config path
    processor = FlashDepthProcessor(config_path='/home/test/Lijunjie/Pano2stereo/submodule/Flashdepth/configs/flashdepth/config.yaml',
                                    save_depth_png=True,
                                    save_frame=True,
                                    stream_mode=True,
                                    max_frames=50)  # Example: test 100 frames
    try:
        processor.run_inference()
    finally:
        processor.cleanup()
