import torch
import numpy as np

def torch_stft(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    Computes STFT using PyTorch matching Banquet specifications.
    Input shape:  (batch, channels, samples)
    Output shape: (batch, channels, frequencies, times) -> Complex Tensor
    """
    batch, channels, samples = waveform.shape
    
    # torch.stft requires the channel/batch dimensions to be flattened into a single leading dimension
    # (batch, channels, samples) -> (batch * channels, samples)
    flat_waveform = waveform.view(-1, samples)
    
    # Create the Hann window on the same device as the input tensor
    window = torch.hann_window(2048, device=waveform.device)
    
    # Compute the STFT
    # returns a complex tensor of shape (batch * channels, 1025, times)
    stft_matrix = torch.stft(
        flat_waveform,
        n_fft=2048,
        hop_length=512,
        win_length=2048,
        window=window,
        center=True,       # Centered: Yes
        pad_mode='reflect', # Padding: Reflect
        normalized=True,   # Normalized: Yes
        onesided=True,     # One-sided: Yes
        return_complex=True
    )
    
    # Reshape back to separate batch and channel dimensions
    # Final shape: (batch, channels, 1025, times)
    freq_bins, times = stft_matrix.shape[-2], stft_matrix.shape[-1]
    return stft_matrix.view(batch, channels, freq_bins, times)


def torch_istft(stft_matrix: torch.Tensor, sample_rate: int, original_length: int | None = None) -> torch.Tensor:
    """
    Computes iSTFT using PyTorch matching Banquet specifications.
    Input shape:  (batch, channels, frequencies, times) -> Complex Tensor
    Output shape: (batch, channels, samples)
    """
    batch, channels, freq_bins, times = stft_matrix.shape
    
    # Flatten leading dimensions for PyTorch: (batch * channels, frequencies, times)
    flat_stft = stft_matrix.view(-1, freq_bins, times)
    
    window = torch.hann_window(2048, device=stft_matrix.device)
    
    # Compute the inverse STFT
    flat_waveform = torch.istft(
        flat_stft,
        n_fft=2048,
        hop_length=512,
        win_length=2048,
        window=window,
        center=True,
        normalized=True,
        onesided=True,
        length=original_length # Passing this truncates any padding artifacts perfectly
    )
    
    # Reshape back to (batch, channels, samples)
    return flat_waveform.view(batch, channels, -1)
