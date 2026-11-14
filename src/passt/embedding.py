import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):

    def __init__(self, patch_dim: int, embed_dim: int):
        super().__init__()
        self.projection = nn.Linear(patch_dim, embed_dim)

    def forward(self, patches: torch.Tensor):
        # patches (B, N, P) -> (B, N, D)
        return self.projection(patches)