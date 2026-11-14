import torch
from torch import nn

class RestoreTokens(nn.Module):

    def __init__(self, embed_dim: int):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1,1,embed_dim))

    def forward(self, encoded, indices, total_tokens):
        # encoded: (B, K, D)
        # indices: (B, K)
        B, K, D = encoded.shape

        full = self.mask_token.expand(B, total_tokens, D).clone()
        full.scatter_(
            dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, D),
            src=encoded
        )

        return full