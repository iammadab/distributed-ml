import torch.nn as nn

from .base import FeedForward

class GELUFeedForward(FeedForward):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__(d_model, d_ff, nn.GELU())


GELUFFN = GELUFeedForward
