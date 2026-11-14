import torch
from torch import nn

class PatchExtractor(nn.Module):

    def __init__(self, patch_height: int, patch_width: int, stride_height: int, stride_width: int):
        super().__init__()
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.stride_height = stride_height
        self.stride_width = stride_width

    def forward(self, spectrogram: torch.Tensor) -> tuple:
        """
        Input: (B, C, F, T)
        """
        if spectrogram.ndim != 4:
            raise ValueError("Expected shape (B, C, F, T)")
        B, C, F, T = spectrogram.shape

        ph = self.patch_height
        pw = self.patch_width
        sh = self.stride_height
        sw = self.stride_width

        if ph > F or pw > T:
            raise ValueError("Patch dimensions exceed spectrogram size")

        # (B, C, Fp, T, ph)
        patches = spectrogram.unfold(2, ph, sh)

        # (B, C, Fp, Tp, ph, pw)
        patches = patches.unfold(3, pw, sw)

        Fp, Tp = patches.shape[2], patches.shape[3]

        # Reshape (B, Fp, Tp, C, ph, pw)
        patches = patches.permute(0, 2, 3, 1, 4, 5)

        # (B, Fp * Tp, C * ph * pw)
        patches = patches.contiguous().view(
            B, Fp * Tp, C * ph * pw
        )

        return patches, (Fp, Tp)