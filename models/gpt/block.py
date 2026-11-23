import torch
import torch.nn as nn

from layers import GELUFeedForward, MaskedSelfAttention, SublayerConnection

from .config import GPTConfig


class GPTBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attention_connection = SublayerConnection(
            config.d_model,
            MaskedSelfAttention(
                config.d_model,
                config.n_heads,
                checkpoint_attention=config.activation_recomputation == "selective",
            ),
            config.dropout,
        )
        self.ffn_connection = SublayerConnection(
            config.d_model,
            GELUFeedForward(config.d_model, config.d_ff),
            config.dropout,
        )

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.attention_connection(token_embeddings)
        return self.ffn_connection(token_embeddings)
