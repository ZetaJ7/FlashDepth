import os
from os.path import join
import cv2
import torch
import numpy as np
import tempfile, shutil
import glob
import logging
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize
from PIL import Image
import torch.distributed as dist
from .depthanything_preprocess import _load_and_process_image, _load_and_process_depth

class RandomDataset(Dataset):
    def __init__(self, root_dir, resolution=None, crop_type=None, large_dir=None):
        self.root_dir = root_dir
        self.resolution = resolution
        self.crop_type = crop_type
        self.large_dir = large_dir

        if self.root_dir.endswith('.mp4'):
            self.seq_paths = [self.root_dir]
        elif os.path.isdir(self.root_dir):
            self.seq_paths = glob.glob(join(self.root_dir, '*.mp4'))
            self.seq_paths = sorted(self.seq_paths)
        else:
            raise ValueError(f"provide an mp4 file or a directory of mp4 files")

        
        
    def __len__(self):
        return len(self.seq_paths)
        
    def __getitem__(self, idx):

        img_paths, tmpdirname = self.parse_seq_path(self.seq_paths[idx])
        img_paths = sorted(img_paths, key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
        imgs = []

        first_img = cv2.imread(img_paths[0])
        h, w = first_img.shape[:2]
        if max(h, w) > 2044: # set max long side to 2044
            logging.info("resizing long side of video to 2044")
            scale = 2044 / max(h, w)
            res = (int(w * scale), int(h * scale))
            logging.info(f"new resolution: {res}")
        else:
            res = (w, h)

        for img_path in img_paths:
            img, _ = _load_and_process_image(img_path, resolution=res, crop_type=None)
            imgs.append(img)
        
        if tmpdirname is not None:
            shutil.rmtree(tmpdirname)

        return dict(batch=torch.stack(imgs).float(), 
                scene_name=os.path.basename(self.seq_paths[idx].split('.')[0]))

    def parse_seq_path(self, p):
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file {p}")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_fps == 0:
            cap.release()
            raise ValueError(f"Error: Video FPS is 0 for {p}")
        frame_interval = 1
        frame_indices = list(range(0, total_frames, frame_interval))
        print(
            f" - Video FPS: {video_fps}, Frame Interval: {frame_interval}, Total Frames to Read: {len(frame_indices)}"
        )
        img_paths = []
        tmpdirname = tempfile.mkdtemp()
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(tmpdirname, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            img_paths.append(frame_path)
        cap.release()
        return img_paths, tmpdirname

class StreamDataset(Dataset):
    def __init__(self, stream_url, resolution=None, crop_type=None, warmup_frames=5):
        # StreamDataset simplified: only responsible for live stream capture
        self.resolution = resolution
        self.crop_type = crop_type
        self.url = stream_url
        self.cap = None

        # Open capture immediately and perform warmup reads to reduce first-frame latency
        try:
            # Force TCP transport for better stability
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
            self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            
            # Set multiple buffer-related properties for better real-time performance
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FPS, 30)  # Limit FPS to prevent overload
            except Exception:
                pass

            if not self.cap.isOpened():
                logging.warning(f"StreamDataset: failed to open stream {self.url} during init")
            else:
                # Pre-read and discard more frames to warm decoder and clear buffer
                for _ in range(max(int(warmup_frames), 10)):
                    ret, _ = self.cap.read()
                    if not ret:
                        break
        except Exception as e:
            logging.warning(f"StreamDataset init warning: {e}")

    def __len__(self):
        # For stream-like usage return a large finite number
        return 10**9

    def __getitem__(self, idx):
        # Ensure capture is open (fallback if warmup/open failed at init)
        if getattr(self, 'cap', None) is None or not self.cap.isOpened():
            try:
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
                self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                try:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                except Exception:
                    pass
                if not self.cap.isOpened():
                    raise ValueError(f"Error opening video stream {self.url}")
                
                # Warm up the new connection
                for _ in range(5):
                    ret, _ = self.cap.read()
                    if not ret:
                        break
            except Exception as e:
                logging.error(f"Failed to reopen stream: {e}")
                raise ValueError(f"Error opening video stream {self.url}")

        # Try to read frame with retry mechanism
        max_retries = 1
        for attempt in range(max_retries):
            ret, frame = self.cap.read()
            origin_frame = frame.copy() if ret else None
            if ret and frame is not None and frame.size > 0:
                break
            else:
                logging.warning(f"Failed to read frame (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(0.01)  # Brief pause before retry
                else:
                    raise RuntimeError("Error reading frame from stream after multiple attempts")

        # Additional validation
        if frame is None or frame.size == 0:
            raise RuntimeError("Received empty frame from stream")

        h, w = frame.shape[:2]
        # normalize/parse resolution if configured (allow numeric strings)
        res_val = None
        if self.resolution is not None:
            try:
                res_val = int(self.resolution)
            except Exception:
                import re
                s = str(self.resolution).lower()
                m = re.search(r"\d+", s)
                if m:
                    num = int(m.group())
                    # support shorthand like '2k' -> 2*1024
                    if 'k' in s:
                        res_val = num * 1024
                    else:
                        res_val = num

        if res_val is not None and max(h, w) > res_val:
            scale = res_val / max(h, w)
            res = (int(w * scale), int(h * scale))
        else:
            res = (w, h)

        img, _ = _load_and_process_image(frame, resolution=res, crop_type=self.crop_type)
        # Normalize output to a torch tensor with shape (C, H, W)
        if isinstance(img, torch.Tensor):
            # Handle possible extra leading dims
            if img.ndim == 4 and img.shape[0] == 1:
                img = img.squeeze(0)
            elif img.ndim == 5 and img.shape[0] == 1 and img.shape[1] == 1:
                img = img.squeeze(0).squeeze(0)
            # If img is (H, W, C) as tensor, permute to (C, H, W)
            if img.ndim == 3 and img.shape[0] not in (1,3):
                # guess HWC
                img = img.permute(2, 0, 1)
        else:
            # numpy array
            if isinstance(img, np.ndarray):
                if img.ndim == 3:
                    # HWC -> CHW
                    img = torch.from_numpy(img).permute(2, 0, 1)
                elif img.ndim == 4 and img.shape[0] == 1:
                    img = torch.from_numpy(img[0]).permute(2, 0, 1)
                else:
                    # fallback: convert and attempt to permute last dim to channels
                    img = torch.from_numpy(img)
                    if img.ndim == 3:
                        img = img.permute(2, 0, 1)

        img = img.float()

        # sanity check spatial dims
        if img.shape[-2] == 0 or img.shape[-1] == 0:
            raise RuntimeError(f"Processed image has zero spatial dimension: {img.shape}")

        return dict(batch=img.unsqueeze(0), frame=origin_frame)

    def __del__(self):
        try:
            if getattr(self, 'cap', None) is not None and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass

class ExternalFrameDataset(Dataset):
    def __init__(self, resolution=None, crop_type=None):
        """Dataset for processing external frames that can be updated dynamically.
        
        Args:
            resolution: Target resolution for processing
            crop_type: Crop type for processing
        """
        self.resolution = resolution
        self.crop_type = crop_type
        self.frame = None
        self.origin_frame = None

    def update_frame(self, frame):
        """Update the current frame to process.
        
        Args:
            frame: Input frame as numpy array (H, W, C) or torch tensor
        """
        if not isinstance(frame, (np.ndarray, torch.Tensor)):
            raise ValueError("Frame must be numpy array or torch tensor")
        
        self.frame = frame
        
        # Store original frame copy for output
        if isinstance(frame, torch.Tensor):
            self.origin_frame = frame.detach().cpu().numpy() if frame.is_cuda else frame.numpy()
        else:
            self.origin_frame = frame.copy()

    def __len__(self):
        return 1 if self.frame is not None else 0

    def __getitem__(self, idx):
        if self.frame is None:
            raise RuntimeError("No frame available. Call update_frame() first.")
        
        if idx != 0:
            raise IndexError("ExternalFrameDataset only contains one frame")
        
        frame = self.frame
        
        # Convert to numpy for processing if it's a tensor
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        
        # Ensure frame is in correct format (H, W, C)
        if frame.ndim == 3 and frame.shape[2] in [1, 3]:  # HWC format
            pass  # Already correct
        elif frame.ndim == 3 and frame.shape[0] in [1, 3]:  # CHW format
            frame = np.transpose(frame, (1, 2, 0))  # CHW -> HWC
        elif frame.ndim == 2:  # Grayscale
            frame = np.expand_dims(frame, axis=2)  # HW -> HWC
        else:
            raise ValueError(f"Unsupported frame shape: {frame.shape}")

        h, w = frame.shape[:2]
        
        # normalize/parse resolution if configured (allow numeric strings)
        res_val = None
        if self.resolution is not None:
            try:
                res_val = int(self.resolution)
            except Exception:
                import re
                s = str(self.resolution).lower()
                m = re.search(r"\d+", s)
                if m:
                    num = int(m.group())
                    # support shorthand like '2k' -> 2*1024
                    if 'k' in s:
                        res_val = num * 1024
                    else:
                        res_val = num

        if res_val is not None and max(h, w) > res_val:
            scale = res_val / max(h, w)
            res = (int(w * scale), int(h * scale))
        else:
            res = (w, h)

        img, _ = _load_and_process_image(frame, resolution=res, crop_type=self.crop_type)
        
        # Normalize output to a torch tensor with shape (C, H, W)
        if isinstance(img, torch.Tensor):
            # Handle possible extra leading dims
            if img.ndim == 4 and img.shape[0] == 1:
                img = img.squeeze(0)
            elif img.ndim == 5 and img.shape[0] == 1 and img.shape[1] == 1:
                img = img.squeeze(0).squeeze(0)
            # If img is (H, W, C) as tensor, permute to (C, H, W)
            if img.ndim == 3 and img.shape[0] not in (1, 3):
                # guess HWC
                img = img.permute(2, 0, 1)
        else:
            # numpy array
            if isinstance(img, np.ndarray):
                if img.ndim == 3:
                    # HWC -> CHW
                    img = torch.from_numpy(img).permute(2, 0, 1)
                elif img.ndim == 4 and img.shape[0] == 1:
                    img = torch.from_numpy(img[0]).permute(2, 0, 1)
                else:
                    # fallback: convert and attempt to permute last dim to channels
                    img = torch.from_numpy(img)
                    if img.ndim == 3:
                        img = img.permute(2, 0, 1)

        img = img.float()

        # sanity check spatial dims
        if img.shape[-2] == 0 or img.shape[-1] == 0:
            raise RuntimeError(f"Processed image has zero spatial dimension: {img.shape}")

        return dict(batch=img.unsqueeze(0), frame=self.origin_frame)