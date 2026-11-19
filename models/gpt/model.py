import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import Embedding, LearnedPositionalEmbedding

from .block import GPTBlock
from .config import GPTConfig


class GPT(nn.Module):
    def __init__(
        self,
        config: GPTConfig,
        embedding: Embedding | None = None,
    ):
        super().__init__()
        self.config = config
        self.embedding = embedding if embedding is not None else Embedding(
            config.vocab_size,
            config.d_model,
        )
        if self.embedding.weight.shape != (config.vocab_size, config.d_model):
            raise ValueError(
                "embedding must have shape "
                f"({config.vocab_size}, {config.d_model})"
            )

        self.position_embedding = LearnedPositionalEmbedding(
            config.d_model,
            config.max_seq_len,
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [GPTBlock(config) for _ in range(config.n_layers)]
        )

        if config.tie_output_weights:
            self.output_projection = None
        else:
            self.output_projection = nn.Linear(
                config.d_model,
                config.vocab_size,
            )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        assert token_ids.ndim == 2
        token_embeddings = self.embedding(token_ids)
        token_embeddings = token_embeddings * math.sqrt(self.config.d_model)
        token_embeddings = self.position_embedding(token_embeddings)
        token_embeddings = self.embedding_dropout(token_embeddings)

        for block in self.blocks:
            token_embeddings = block(token_embeddings)

        if self.config.tie_output_weights:
            return F.linear(token_embeddings, self.embedding.weight)
        return self.output_projection(token_embeddings)
