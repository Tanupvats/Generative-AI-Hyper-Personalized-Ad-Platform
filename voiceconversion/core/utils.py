import os
import glob
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

def load_checkpoint(checkpoint_path, model, optimizer=None, load_opt=True):
    """
    Loads a PyTorch checkpoint for VITS/RVC models.
    """
    assert os.path.isfile(checkpoint_path), f"No checkpoint found at {checkpoint_path}"
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    
    iteration = checkpoint_dict.get("iteration", 0)
    learning_rate = checkpoint_dict.get("learning_rate", 1e-4)
    
    if optimizer is not None and load_opt:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
        
    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
        
    new_state_dict = {}
    for k, v in saved_state_dict.items():
        try:
            new_state_dict[k] = v
        except Exception:
            logger.warning(f"{k} is not in the model architecture and will be skipped.")
            continue
            
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)
        
    logger.info(f"Loaded checkpoint '{checkpoint_path}' (iteration {iteration})")
    return model, optimizer, learning_rate, iteration

def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    """
    Saves a training checkpoint.
    """
    logger.info(f"Saving checkpoint at iteration {iteration} to {checkpoint_path}")
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
        
    torch.save(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )

def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    """
    Finds the most recently modified checkpoint in a directory.
    """
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=os.path.getmtime)
    if len(f_list) == 0:
        return None
    return f_list[-1]

def inference(spec, device, model, aggressiveness, mp_param):
    """
    Specialized windowed inference for UVR5 vocal separation.
    This handles the tiling/overlap logic to prevent artifacts on large spectrograms.
    """
    with torch.no_grad():
        # spec: [Batch, Channels, Freq, Time]
        # UVR5 models expect magnitude spectrograms
        v_spec_m = torch.abs(spec)
        v_spec_m = v_spec_m.to(device)
        
        if "cuda" in str(device):
            v_spec_m = v_spec_m.half()
            model = model.half()
            
        # Determine tiling parameters
        batch_size = 1
        window_size = 512 # Standard for many UVR5 models
        hop_size = window_size // 2
        
        n_bins = v_spec_m.shape[2]
        n_frames = v_spec_m.shape[3]
        
        # Pad the spectrogram to fit window increments
        pad_size = window_size - n_frames % window_size
        v_spec_m = torch.nn.functional.pad(v_spec_m, (0, pad_size))
        
        new_n_frames = v_spec_m.shape[3]
        result_mask = torch.zeros_like(v_spec_m)
        counter = torch.zeros_like(v_spec_m)
        
        # Sliding window inference
        for i in range(0, new_n_frames - hop_size, hop_size):
            end_idx = i + window_size
            if end_idx > new_n_frames:
                break
                
            patch = v_spec_m[:, :, :, i:end_idx]
            mask_patch = model(patch)
            
            result_mask[:, :, :, i:end_idx] += mask_patch
            counter[:, :, :, i:end_idx] += 1.0
            
        # Average the overlaps and crop back to original size
        result_mask = result_mask / torch.clamp(counter, min=1.0)
        result_mask = result_mask[:, :, :, :n_frames]
        
        # Post-processing aggressiveness (optional refinement)
        if aggressiveness["value"] != 0:
            # Apply thresholding or weighting if needed by specific UVR5 models
            pass
            
        # Convert result mask back to CPU float32
        pred_spec = (spec.to(device) * result_mask).cpu().float()
        
        return pred_spec, None, None

def get_hparams_from_file(config_path):
    """
    Loads JSON hyperparameters from a file.
    """
    import json
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)
    return config