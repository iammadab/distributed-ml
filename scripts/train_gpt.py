from pathlib import Path
import sys
import os

import torch
import torch.nn.functional as F

# Make the repository's top-level packages available when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layers import Embedding
from models.gpt import GPT, GPTConfig
from profiling import (
    Profiler,
)


VOCAB_SIZE = 40_478
DEFAULT_BATCH_SIZE = 2


def train_step(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    token_ids: torch.Tensor,
    profiler: Profiler,
) -> torch.Tensor:
    # The input is all but the last token; the target is all but the first.
    inputs = token_ids[:, :-1]
    targets = token_ids[:, 1:]

    optimizer.zero_grad()

    with profiler.region("forward"):
        logits = model(inputs)

    with profiler.region("loss"):
        loss = F.cross_entropy(
            # [batch, seq_len, vocab_size] => [batch * seq_len, vocab_size]
            logits.reshape(-1, logits.size(-1)),
            # [batch, seq_len] => [batch * seq_len]
            targets.reshape(-1),
        )

    with profiler.region("backward"):
        loss.backward()

    with profiler.region("optimizer_step"):
        optimizer.step()

    return loss


def main() -> None:
    device = torch.device(os.getenv("DEVICE", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available")
    sequence_length = int(os.getenv("SEQ_LEN", "512"))
    batch_size = int(os.getenv("BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))
    if sequence_length <= 0:
        raise ValueError("SEQ_LEN must be positive")
    if batch_size <= 0:
        raise ValueError("BATCH_SIZE must be positive")

    print(f"Running on device: {device}")
    print(f"Sequence length: {sequence_length}")
    print(f"Batch size: {batch_size}")

    config = GPTConfig(
        vocab_size=VOCAB_SIZE,
        max_seq_len=sequence_length,
        d_model=768,
        d_ff=3_072,
        n_heads=12,
        n_layers=12,
        activation_recomputation=os.getenv("ACTIVATION_RECOMPUTATION", "none"),
    )
    model = GPT(config, Embedding(config.vocab_size, config.d_model)).to(device)
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
        (batch_size, sequence_length),
    ).to(device)

    profiler = Profiler.from_environment(
        Path("traces"),
        device,
        metadata={"batch_size": batch_size, "sequence_length": sequence_length},
    )

    def make_step():
        nonlocal model, optimizer
        torch.manual_seed(0)
        model = GPT(config, Embedding(config.vocab_size, config.d_model)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)
        return lambda: train_step(model, optimizer, token_ids, profiler)

    result = profiler.run(
        step_factory=make_step,
        warmup_steps=int(os.getenv("WARMUP_STEPS", "1")),
        measured_steps=int(os.getenv("BENCHMARK_STEPS", "3")),
    )

    if result.timing is None and result.memory is None:
        loss = make_step()()
        print(f"step 0: loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
