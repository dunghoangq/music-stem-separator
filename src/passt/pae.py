import torch
import torch.nn as nn
from passt.extractor import PatchExtractor
from passt.embedding import PatchEmbedding
from passt.positional_encoding import FactorisedPositionalEncoding
from passt.patchout import UnstructuredPatchout
from passt.transformer import TransformerEncoder

class PatchoutAudioEncoder(nn.Module):

    def __init__(
        self,
        patch_height: int,
        patch_width: int,
        stride_height: int,
        stride_width: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout_ratio: float = .5,
        max_freq_patches: int = 128,
        max_time_patches: int = 512
    ):
        super().__init__()
        self.extractor = PatchExtractor(patch_height, patch_width, stride_height, stride_width)
        
        patch_dim = patch_width * patch_height

        self.embedding = PatchEmbedding(patch_dim, embed_dim)
        self.positional_encoding = FactorisedPositionalEncoding(max_freq_patches, max_time_patches, embed_dim)
        self.patchout = UnstructuredPatchout(dropout_ratio)
        self.transformer = TransformerEncoder(embed_dim, num_heads, num_layers)

    def forward(self, spectrogram):
        patches, grid = self.extractor(spectrogram)
        embeddings = self.embedding(patches)
        embeddings = self.positional_encoding(embeddings, grid)
        embeddings, indices = self.patchout(embeddings)
        encoded = self.transformer(embeddings)

        return encoded, indices, grid