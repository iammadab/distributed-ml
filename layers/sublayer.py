import torch
import torch.nn as nn


class SublayerConnection(nn.Module):
    def __init__(self, d_model: int, sublayer: nn.Module):
        super().__init__()
        self.sublayer = sublayer
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, token_embeddings: torch.Tensor, **kwargs) -> torch.Tensor:
        sublayer_output = self.sublayer(
            token_embeddings=token_embeddings,
            **kwargs,
        )
        return self.layer_norm(token_embeddings + sublayer_output)
