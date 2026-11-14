import torch
from torch import nn

class FactorisedPositionalEncoding(nn.Module):

    def __init__(
        self,
        max_freq_patches: int,
        max_time_patches: int,
        embed_dim: int
    ):
        super().__init__()

        """
        The shapes match a 4D patch grid of (Batch, Freq_Patches, Time_Patches, Embed_Dim)

        - freq_embedding: The 1 in the time dimension means every time step at the same frequency shares the exact same frequency embedding.
        - time_embedding: The 1 in the frequency dimension means every frequency bin at the same time step shares the exact same time embedding.

        When they are added together in forward() (sliced to the actual (Fp, Tp) from the PatchExtractor), PyTorch automatically broadcasts them across the missing dimensions.
        """

        self.freq_embedding = nn.Parameter(
            torch.randn(1, max_freq_patches, 1, embed_dim)*.02
        )
        self.time_embedding = nn.Parameter(
            torch.randn(1, 1, max_time_patches, embed_dim)*.02
        )

    def forward(self, embeddings: torch.Tensor, grid: tuple):
        B, N, D = embeddings.shape
        Fp, Tp = grid

        if Fp > self.freq_embedding.shape[1]:
            raise ValueError("Too many frequency patches.")
        if Tp > self.time_embedding.shape[2]:
            raise ValueError("Too many time patches.")

        x = embeddings.reshape(B, Fp, Tp, D)

        x = x + self.freq_embedding[:, :Fp] + self.time_embedding[:, :, :Tp]

        return x.reshape(B, N, D)