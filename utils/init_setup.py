import os 
import re
import random
import logging
import shutil
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.nn.parallel import DistributedDataParallel as DDP

from flashdepth.model import FlashDepth
from .helpers import LinearWarmupExponentialDecay, get_warmup_lambda

def dist_init():
    # Check if running in distributed mode
    if "LOCAL_RANK" in os.environ and "LOCAL_WORLD_SIZE" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", timeout=timedelta(seconds=3600))
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        node_idx = rank // local_world_size
        num_nodes = world_size // local_world_size
    else:
        # Single process mode
        local_rank = 0
        local_world_size = 1
        rank = 0
        world_size = 1
        node_idx = 0
        num_nodes = 1
        # No need to init process group for single process

    seed = 42 + rank
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # dist.barrier()   # device_ids=[local_rank]
    return dict(
        rank=rank, 
        world_size=world_size,
        local_rank=local_rank, 
        local_world_size=local_world_size,
        node_idx=node_idx, 
        num_nodes=num_nodes
    )

def setup_model(cfg, process_dict):
    '''
    Instantiate model and load weights for inference only.
    '''
    model = FlashDepth(**dict(
        batch_size=1,  # Inference batch size fixed为1
        training=False,
        **cfg.model,
    ))
    model = model.to(torch.cuda.current_device())

    # Load checkpoint if specified
    train_step = 0
    if cfg.load is not None or cfg.inference:
        model = model.cpu()
        train_step = load_checkpoint(cfg, model)
        model = model.to(torch.cuda.current_device())

    return model, train_step

def load_checkpoint(cfg, model):
    if cfg.load is not None and cfg.inference:
        if isinstance(cfg.load, str) and '.pth' in cfg.load:
            print(cfg.load)
            checkpoint_path = cfg.load
            logging.info(f"force loading checkpoint {checkpoint_path}...")
        elif cfg.load is True:
            print(cfg)
            existing_ckpts = get_existing_ckpts(cfg)
            if existing_ckpts:
                checkpoint_path = existing_ckpts[-1]
                logging.info(f"loading latest checkpoint {checkpoint_path}...")
            else:
                raise FileNotFoundError(f"No checkpoints found in config_dir ({existing_ckpts}), cannot load model!")
        else:
            print(cfg)
            existing_ckpts = get_existing_ckpts(cfg)
            if existing_ckpts:
                checkpoint_path = existing_ckpts[-1]
                logging.info(f"loading latest checkpoint {checkpoint_path}...")
            else:
                raise FileNotFoundError(f"No checkpoints found in config_dir ({existing_ckpts}), cannot load model!")
    else:
        existing_ckpts = get_existing_ckpts(cfg)
        if len(existing_ckpts) > 0:
            checkpoint_path = existing_ckpts[-1]
        else:
            checkpoint_path = cfg.load
            logging.info(f"assuming checkpoint is pretrained...")
        logging.info(f"existing ckpts: {existing_ckpts}")
        logging.info(f"loading checkpoint from {checkpoint_path}...")

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist, cannot load model!")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(model_state_dict, strict=False)
    train_step = checkpoint.get('step', 0)
    logging.info(f"checkpoint loaded successfully!")
    return train_step

def save_checkpoint(cfg, model, optimizer, lr_scheduler, train_step):


    checkpoint = {
        'model': model.module.state_dict(),
        'step': train_step
    }
    
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    if lr_scheduler is not None:
        checkpoint['lr_scheduler'] = lr_scheduler.state_dict()

    save_path = os.path.join(cfg.config_dir, f"iter_{train_step}.pth")
    if dist.get_rank() == 0:
        torch.save(checkpoint, save_path)
        logging.info(f"Saved checkpoint to {save_path} at step {train_step}")
        # cleanup(args)

    dist.barrier()

def cleanup(cfg, keep_latest_n=2, keep_freq=10000):
    '''
    keep_latest_n: number of latest checkpoints to keep
    keep_freq: overwrite keep_latest_n and continue to store checkpoints at these frequencies
    TODO: save based on validation accuracy
    '''

    existing_ckpts = get_existing_ckpts(cfg)
    for ckpt in existing_ckpts[:-keep_latest_n]:
        step = int(ckpt.split('iter_')[1].split('.')[0])
        if step % keep_freq != 0:
            if os.path.isfile(ckpt):
                os.remove(ckpt)


def get_existing_ckpts(cfg):
    existing_ckpts = [
        item for item in os.listdir(cfg.config_dir)
        if os.path.isfile(os.path.join(cfg.config_dir, item)) and re.match(r'^iter_\d+.pth$', item)
    ]

    existing_ckpts = sorted(existing_ckpts, key=lambda x: int(x.split('iter_')[1].split('.')[0]))
    existing_ckpts = [os.path.join(cfg.config_dir, ckpt) for ckpt in existing_ckpts]
    return existing_ckpts



def has_valid_gradients(model, train_step, loss, max_grad_norm=20.0, max_loss=10.0):
    """Check if gradients and loss are within valid ranges."""
    _tensor_cache = torch.tensor([0], device=torch.cuda.current_device())

    if train_step < 5000:
        max_grad_norm *= 5

    if train_step < 500:    
        max_loss *= 200
    
    # Check loss magnitude
    if loss.item() > max_loss or not torch.isfinite(loss):
        logging.warning(f"WARNING: skip optimizer.step(), as rank {dist.get_rank()} step {train_step} found loss value of {loss.item()}")
        _tensor_cache[0] = 1
    
    # Check gradients if loss was okay
    if _tensor_cache[0] == 0:
        with torch.no_grad():
            # First check for inf/nan
            for n, p in model.module.named_parameters():
                if (p.requires_grad) and (p.grad is not None):
                    invalid_grad_cnt = p.grad.numel() - torch.isfinite(p.grad).sum().item()
                    if invalid_grad_cnt > 0:
                        logging.warning(f"WARNING: skip optimizer.step(), as rank {dist.get_rank()} step {train_step} found {invalid_grad_cnt} invalid grads for {n}")
                        _tensor_cache[0] = 1
                        break
            
            # Then check gradient norms
            if _tensor_cache[0] == 0:
                for n, p in model.module.named_parameters():
                    if (p.requires_grad) and (p.grad is not None):
                        grad_norm = torch.norm(p.grad.detach(), p=2)
                        if grad_norm > max_grad_norm:
                            logging.warning(f"WARNING: skip optimizer.step(), as rank {dist.get_rank()} step {train_step} found large gradient norm {grad_norm:.1f} > {max_grad_norm} for {n}")
                            _tensor_cache[0] = 1
                            break
    
    # Gather results from all processes
    skip_optim_step_list = [torch.tensor([0], device=torch.cuda.current_device()) for _ in range(dist.get_world_size())]
    dist.all_gather(skip_optim_step_list, _tensor_cache)
    
    return not any(t.item() == 1 for t in skip_optim_step_list)