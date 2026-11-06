from pathlib import Path
import numpy as np
import torch
import soundfile as sf

def load_audio(path: str | Path, device: str = "cpu") -> tuple[torch.Tensor, int]:
    """
    Loads audio, standardizes shape to (batch, channels, samples),
    and returns a PyTorch FloatTensor on the specified device.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Audio file path does not exist: {path}")
    
    # 1. Load using soundfile (returns np.ndarray)
    waveform_np, sample_rate = sf.read(path, dtype='float32')
    
    if not np.isfinite(waveform_np).all():
        raise ValueError(f"Audio file at {path} contains non-finite values.")
        
    # 2. Standardize dimensions in NumPy first
    if waveform_np.ndim == 1:
        waveform_np = waveform_np[np.newaxis, np.newaxis, :]
    elif waveform_np.ndim == 2:
        waveform_np = waveform_np.T[np.newaxis, :]
    else:
        raise ValueError(f"Unexpected audio shape layout: {waveform_np.shape}")
        
    # 3. Convert to PyTorch Tensor (Zero-copy memory sharing)
    waveform_tensor = torch.from_numpy(waveform_np)
    
    # 4. Optional: Move directly to your target device (e.g., 'cuda' or 'cpu')
    if device != "cpu":
        waveform_tensor = waveform_tensor.to(device)
        
    return waveform_tensor, sample_rate
