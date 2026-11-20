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
    collect_memory_profile,
    load_memory_profile,
    serve_visualizer,
    write_memory_visualizer,
)


VOCAB_SIZE = 40_478
SEQUENCE_LENGTH = 512
BATCH_SIZE = 2


def train_step(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    # The input is all but the last token; the target is all but the first.
    inputs = token_ids[:, :-1]
    targets = token_ids[:, 1:]

    optimizer.zero_grad()

    with torch.profiler.record_function("forward"):
        logits = model(inputs)

    with torch.profiler.record_function("loss"):
        loss = F.cross_entropy(
            # [batch, seq_len, vocab_size] => [batch * seq_len, vocab_size]
            logits.reshape(-1, logits.size(-1)),
            # [batch, seq_len] => [batch * seq_len]
            targets.reshape(-1),
        )

    with torch.profiler.record_function("backward"):
        loss.backward()

    with torch.profiler.record_function("optimizer_step"):
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

    if os.getenv("PROFILE_MEMORY") == "1":
        artifacts = collect_memory_profile(
            step_fn=lambda: train_step(model, optimizer, token_ids),
            output_dir=Path("traces"),
        )
        profile = load_memory_profile(
            artifacts.trace_path,
            artifacts.memory_timeline_path,
        )
        visualizer_path = Path("traces/gpt_memory.html")
        write_memory_visualizer(profile, visualizer_path)
        serve_visualizer(visualizer_path)
    else:
        loss = train_step(model, optimizer, token_ids)
        print(f"step 0: loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
