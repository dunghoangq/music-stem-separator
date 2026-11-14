import torch
import torch.nn as nn

class UnstructuredPatchout(nn.Module):

    def __init__(self, dropout_ratio: float = .5):
        super().__init__()
        if not 0 <= dropout_ratio < 1:
            raise ValueError("Invalid dropout ratio")

        self.dropout_ratio = dropout_ratio

    def forward(self, x: torch.Tensor):
        # x: (B, N, D)
        B, N, D = x.shape

        # No patchout during inference
        if not self.training or self.dropout_ratio == 0.0:
            indices = torch.arange(N, device=x.device)\
                .unsqueeze(0).expand(B, -1)
            return x, indices
        
        keep_out = max(1, round(N*(1 - self.dropout_ratio)))

        # Randomly select indices for each sample
        indices = torch.rand(B, N, device=x.device)\
            .topk(keep_out, dim=1).indices

        # Restore original sequence ordering
        # indices:= the original positions of those selected tokens
        indices = indices.sort(dim=1).values

        # Select tokens
        # selected:= the reduced Transformer input
        selected = torch.gather(
            x, dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, D)
        )

        return selected, indices