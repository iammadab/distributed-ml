from dataclasses import dataclass


@dataclass
class TransformerConfig:
    source_vocab_size: int
    target_vocab_size: int
    d_model: int
    d_ff: int
    n_heads: int
    n_layers: int
    max_seq_len: int
    dropout: float = 0.1
