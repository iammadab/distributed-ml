import math

import torch
import torch.nn as nn

from layers import (
    CrossAttention,
    Embedding,
    FFN,
    MaskedSelfAttention,
    PositionalEncoding,
    SelfAttention,
    SublayerConnection,
)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int, dropout: float):
        super().__init__()
        self.attention_connection = SublayerConnection(
            d_model,
            SelfAttention(d_model, n_heads),
            dropout,
        )
        self.ffn_connection = SublayerConnection(
            d_model,
            FFN(d_model, d_ff),
            dropout,
        )

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.attention_connection(token_embeddings)
        return self.ffn_connection(token_embeddings)


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        n_layers: int,
        max_seq_len: int,
        dropout: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model, max_seq_len)
        self.embedding_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, d_ff, n_heads, dropout) for _ in range(n_layers)]
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.embedding(token_ids)
        token_embeddings = token_embeddings * math.sqrt(self.d_model)
        token_embeddings = self.position_encoding(token_embeddings)
        token_embeddings = self.embedding_dropout(token_embeddings)

        for layer in self.layers:
            token_embeddings = layer(token_embeddings)

        return token_embeddings


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int, dropout: float):
        super().__init__()
        self.masked_attention_connection = SublayerConnection(
            d_model,
            MaskedSelfAttention(d_model, n_heads),
            dropout,
        )
        self.cross_attention_connection = SublayerConnection(
            d_model,
            CrossAttention(d_model, n_heads),
            dropout,
        )
        self.ffn_connection = SublayerConnection(
            d_model,
            FFN(d_model, d_ff),
            dropout,
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
    def __init__(
        self,
        target_vocab_size: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        n_layers: int,
        max_seq_len: int,
        dropout: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = Embedding(target_vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model, max_seq_len)
        self.embedding_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, d_ff, n_heads, dropout) for _ in range(n_layers)]
        )

    def forward(
        self,
        target_token_ids: torch.Tensor,
        encoder_output: torch.Tensor,
    ) -> torch.Tensor:
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
        source_vocab_size: int,
        target_vocab_size: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        n_layers: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = Encoder(
            source_vocab_size,
            d_model,
            d_ff,
            n_heads,
            n_layers,
            max_seq_len,
            dropout,
        )
        self.decoder = Decoder(
            target_vocab_size,
            d_model,
            d_ff,
            n_heads,
            n_layers,
            max_seq_len,
            dropout,
        )
        self.output_projection = nn.Linear(d_model, target_vocab_size)

    def forward(
        self,
        source_token_ids: torch.Tensor,
        target_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        encoder_output = self.encoder(source_token_ids)
        decoder_output = self.decoder(target_token_ids, encoder_output)
        return self.output_projection(decoder_output)
