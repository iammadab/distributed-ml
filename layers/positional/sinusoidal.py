import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
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
        assert token_embeddings.ndim == 3
        seq_len = token_embeddings.size(-2)
        assert seq_len <= self.initial_seq_len
        return token_embeddings + self.encoding_table[:seq_len].unsqueeze(0)
