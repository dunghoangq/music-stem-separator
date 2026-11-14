import torch
from torch import nn

class TransformerEncoder(nn.Module):
    """
    This is a conventional Transformer encoder implementation inspired by the PaSST architecture, not a bit-for-bit reproduction of its DeiT-based implementation.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        mlp_ratio: int = 4,
        dropout: float = .1
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers
        )

    def forward(self, x: torch.Tensor):
        return self.encoder(x)