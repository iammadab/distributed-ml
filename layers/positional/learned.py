import torch
import torch.nn as nn


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        assert token_embeddings.ndim == 3
        seq_len = token_embeddings.size(-2)
        assert seq_len <= self.position_embedding.num_embeddings

        positions = torch.arange(seq_len, device=token_embeddings.device)
        position_embeddings = self.position_embedding(positions)
        return token_embeddings + position_embeddings.unsqueeze(0)
