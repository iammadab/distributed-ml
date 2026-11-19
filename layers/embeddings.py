import torch.nn as nn
import torch


class Embedding(nn.Module):
    """Look up token embeddings."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)

    @property
    def weight(self) -> torch.Tensor:
        return self.embedding.weight
