from pathlib import Path
import sys

import torch
import torch.nn.functional as F

# Make the repository's top-level packages available when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layers import Embedding
from models.gpt import GPT, GPTConfig


VOCAB_SIZE = 40_478
SEQUENCE_LENGTH = 512
BATCH_SIZE = 2
NUM_STEPS = 5


def train_step(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    # The input is all but the last token; the target is all but the first.
    inputs = token_ids[:, :-1]
    targets = token_ids[:, 1:]

    optimizer.zero_grad()

    # Forward pass.
    logits = model(inputs)

    # Compute the next-token prediction loss.
    loss = F.cross_entropy(
        # [batch, seq_len, vocab_size] => [batch * seq_len, vocab_size]
        logits.reshape(-1, logits.size(-1)),
        # [batch, seq_len] => [batch * seq_len]
        targets.reshape(-1),
    )

    # Backward pass and optimization step.
    loss.backward()
    optimizer.step()

    return loss


def main() -> None:
    config = GPTConfig(
        vocab_size=VOCAB_SIZE,
        max_seq_len=SEQUENCE_LENGTH,
        d_model=768,
        d_ff=3_072,
        n_heads=12,
        n_layers=12,
    )
    model = GPT(config, Embedding(config.vocab_size, config.d_model))
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)

    # count and print the total number of trainable parameters
    total_parameters = sum([parameter.numel() for parameter in model.parameters()])
    print(f"Total parameters: {total_parameters:,}")

    # here, I am generating a synthentic token sequence
    # because for now I only care about a single forward,
    # backward and optim.step
    token_ids = torch.randint(
        0,
        config.vocab_size,
        (BATCH_SIZE, SEQUENCE_LENGTH),
    )

    for step in range(NUM_STEPS):
        loss = train_step(model, optimizer, token_ids)
        print(f"step {step}: loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
