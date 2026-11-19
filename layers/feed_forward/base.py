import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, activation: nn.Module):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.activation = activation
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.activation(self.linear_1(token_embeddings)))
