from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int
    max_seq_len: int
    d_model: int
    d_ff: int
    n_heads: int
    n_layers: int
    dropout: float = 0.1
