import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import (
    CrossAttention,
    Embedding,
    FFN,
    MaskedSelfAttention,
    SelfAttention,
    SublayerConnection,
)
from .config import TransformerConfig
from layers.positional import SinusoidalPositionalEncoding


class EncoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention_connection = SublayerConnection(
            config.d_model,
            SelfAttention(config.d_model, config.n_heads),
            config.dropout,
        )
        self.ffn_connection = SublayerConnection(
            config.d_model,
            FFN(config.d_model, config.d_ff),
            config.dropout,
        )

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.attention_connection(token_embeddings)
        return self.ffn_connection(token_embeddings)


class Encoder(nn.Module):
    def __init__(self, config: TransformerConfig, embedding: Embedding):
        super().__init__()
        self.d_model = config.d_model
        self.embedding = embedding
        self.position_encoding = SinusoidalPositionalEncoding(
            config.d_model,
            config.max_seq_len,
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.n_layers)]
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        assert token_ids.ndim == 2
        token_embeddings = self.embedding(token_ids)
        token_embeddings = token_embeddings * math.sqrt(self.d_model)
        token_embeddings = self.position_encoding(token_embeddings)
        token_embeddings = self.embedding_dropout(token_embeddings)

        for layer in self.layers:
            token_embeddings = layer(token_embeddings)

        return token_embeddings


class DecoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.masked_attention_connection = SublayerConnection(
            config.d_model,
            MaskedSelfAttention(config.d_model, config.n_heads),
            config.dropout,
        )
        self.cross_attention_connection = SublayerConnection(
            config.d_model,
            CrossAttention(config.d_model, config.n_heads),
            config.dropout,
        )
        self.ffn_connection = SublayerConnection(
            config.d_model,
            FFN(config.d_model, config.d_ff),
            config.dropout,
        )

    def forward(
        self,
        target_token_embeddings: torch.Tensor,
        encoder_output: torch.Tensor,
    ) -> torch.Tensor:
        target_token_embeddings = self.masked_attention_connection(
            target_token_embeddings
        )
        target_token_embeddings = self.cross_attention_connection(
            token_embeddings=target_token_embeddings,
            encoder_output=encoder_output,
        )
        return self.ffn_connection(target_token_embeddings)


class Decoder(nn.Module):
    def __init__(self, config: TransformerConfig, embedding: Embedding):
        super().__init__()
        self.d_model = config.d_model
        self.embedding = embedding
        self.position_encoding = SinusoidalPositionalEncoding(
            config.d_model,
            config.max_seq_len,
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.n_layers)]
        )

    def forward(
        self,
        target_token_ids: torch.Tensor,
        encoder_output: torch.Tensor,
    ) -> torch.Tensor:
        assert target_token_ids.ndim == 2
        assert encoder_output.ndim == 3
        target_token_embeddings = self.embedding(target_token_ids)
        target_token_embeddings = target_token_embeddings * math.sqrt(self.d_model)
        target_token_embeddings = self.position_encoding(target_token_embeddings)
        target_token_embeddings = self.embedding_dropout(target_token_embeddings)

        for layer in self.layers:
            target_token_embeddings = layer(
                target_token_embeddings,
                encoder_output,
            )

        return target_token_embeddings


class Transformer(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        embedding: Embedding,
    ):
        super().__init__()
        if config.source_vocab_size != config.target_vocab_size:
            raise ValueError(
                "source_vocab_size and target_vocab_size must match when sharing "
                "the embedding"
            )
        if embedding.weight.shape != (config.source_vocab_size, config.d_model):
            raise ValueError(
                "embedding must have shape "
                f"({config.source_vocab_size}, {config.d_model})"
            )

        self.config = config
        self.embedding = embedding
        self.encoder = Encoder(config, embedding)
        self.decoder = Decoder(config, embedding)

    def forward(
        self,
        source_token_ids: torch.Tensor,
        target_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        encoder_output = self.encoder(source_token_ids)
        decoder_output = self.decoder(target_token_ids, encoder_output)
        return F.linear(decoder_output, self.embedding.weight)
