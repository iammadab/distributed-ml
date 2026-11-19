import math

import torch
import torch.nn as nn


def split_heads(x: torch.Tensor, n_heads: int, d_k: int) -> torch.Tensor:
    """Convert [batch_size, seq_len, d_model] into [batch_size, n_heads, seq_len, d_k]."""
    batch_size, seq_len, _ = x.shape
    return x.reshape(batch_size, seq_len, n_heads, d_k).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """Convert [batch_size, n_heads, seq_len, d_k] into [batch_size, seq_len, d_model]."""
    batch_size, n_heads, seq_len, d_k = x.shape
    return x.transpose(1, 2).reshape(batch_size, seq_len, n_heads * d_k)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    scores = q @ k.transpose(-1, -2)
    scores = scores / math.sqrt(q.size(-1))

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    return weights @ v


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        assert d_model % n_heads == 0
        super().__init__()

        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.projection_matrix = nn.Linear(d_model, 3 * d_model)
        self.output_matrix = nn.Linear(d_model, d_model)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        assert token_embeddings.ndim == 3
        projection = self.projection_matrix(token_embeddings)
        q, k, v = projection.chunk(3, dim=-1)

        q = split_heads(q, self.n_heads, self.d_k)
        k = split_heads(k, self.n_heads, self.d_k)
        v = split_heads(v, self.n_heads, self.d_k)

        return self.output_matrix(merge_heads(attention(q, k, v)))


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        assert d_model % n_heads == 0
        super().__init__()

        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.kv_projection_matrix = nn.Linear(d_model, 2 * d_model)
        self.q_projection_matrix = nn.Linear(d_model, d_model)
        self.output_matrix = nn.Linear(d_model, d_model)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        encoder_output: torch.Tensor,
    ) -> torch.Tensor:
        assert token_embeddings.ndim == 3
        assert encoder_output.ndim == 3
        kv_projection = self.kv_projection_matrix(encoder_output)
        q = self.q_projection_matrix(token_embeddings)
        k, v = kv_projection.chunk(2, dim=-1)

        q = split_heads(q, self.n_heads, self.d_k)
        k = split_heads(k, self.n_heads, self.d_k)
        v = split_heads(v, self.n_heads, self.d_k)

        return self.output_matrix(merge_heads(attention(q, k, v)))


class MaskedSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        assert d_model % n_heads == 0
        super().__init__()

        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv_projection_matrix = nn.Linear(d_model, 3 * d_model)
        self.output_matrix = nn.Linear(d_model, d_model)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        assert token_embeddings.ndim == 3
        projection = self.qkv_projection_matrix(token_embeddings)
        q, k, v = projection.chunk(3, dim=-1)

        q = split_heads(q, self.n_heads, self.d_k)
        k = split_heads(k, self.n_heads, self.d_k)
        v = split_heads(v, self.n_heads, self.d_k)

        seq_len = token_embeddings.size(1)
        mask = torch.tril(
            torch.ones(
                (1, 1, seq_len, seq_len),
                dtype=torch.bool,
                device=token_embeddings.device,
            )
        )

        return self.output_matrix(merge_heads(attention(q, k, v, mask)))
