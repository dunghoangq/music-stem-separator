import torch
from torch import nn

class BandSplit(nn.Module):
    """
    The input band_edges must match the perceived output tracks' frequency.
    Say, bass (~20 Hz - 250 Hz), mid/vocal (~250 Hz - 4 kHz), Treble (~4 kHz - 20 kHz)
    But we need to convert these freq ranges to integer bin indices.

    Bin index = round(freq in Hz x N_FFT / Sampling Rate)

    e.g., sample rate = 44100 Hz, N_FFT = 2048, then:
        - Bass: (0, 12)
        - Low-mid: (12, 46)
        - High-mid: (46, 186)
        - Treble: (186, 1025)
    """

    def __init__(self, band_edges: list[tuple[int, int]]):
        super().__init__()
        self.band_edges = band_edges
        for start, end in band_edges:
            if start < 0 or end <= start:
                raise ValueError(f"Invalid band boundaries: {(start, end)}.")

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        x: complex STFT, shape (B, C, F, T).
        Returns:
            list of complex tensors, one per freq band.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected (B, C, F, T), got {x.shape}")

        freq_bins = x.shape[-2]
        bands = []
        for start, end in self.band_edges:
            if end > freq_bins:
                raise ValueError(
                    f"Band {(start, end)} exceeds "
                    f"{freq_bins} frequency bins"
                )
            band = x[:, :, start:end, :]
            bands.append(band)

        return bands