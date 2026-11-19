import torch
import torch.nn as nn


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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, initial_seq_len: int):
        assert d_model % 2 == 0

        super().__init__()
        self.initial_seq_len = initial_seq_len

        exponents = torch.arange(0, d_model, 2, dtype=torch.float32) / -d_model
        denoms = torch.pow(10000, exponents)
        pos = torch.arange(0, initial_seq_len).unsqueeze(1)

        self.register_buffer("encoding_table", torch.empty((initial_seq_len, d_model)))
        self.encoding_table[:, 0::2] = torch.sin(pos * denoms)
        self.encoding_table[:, 1::2] = torch.cos(pos * denoms)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        seq_len = len(token_embeddings)
        assert seq_len <= self.initial_seq_len
        return token_embeddings + self.encoding_table[:seq_len]
