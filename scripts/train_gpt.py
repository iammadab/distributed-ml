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
    benchmark_training_step,
    collect_memory_profile,
    load_memory_profile,
    serve_visualizer,
    timed_region,
    write_timing,
    write_memory_visualizer,
)


VOCAB_SIZE = 40_478
BATCH_SIZE = 2


def train_step(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    token_ids: torch.Tensor,
    timer=None,
) -> torch.Tensor:
    # The input is all but the last token; the target is all but the first.
    inputs = token_ids[:, :-1]
    targets = token_ids[:, 1:]

    optimizer.zero_grad()

    with torch.profiler.record_function("forward"), timed_region(timer, "forward"):
        logits = model(inputs)

    with torch.profiler.record_function("loss"), timed_region(timer, "loss"):
        loss = F.cross_entropy(
            # [batch, seq_len, vocab_size] => [batch * seq_len, vocab_size]
            logits.reshape(-1, logits.size(-1)),
            # [batch, seq_len] => [batch * seq_len]
            targets.reshape(-1),
        )

    with torch.profiler.record_function("backward"), timed_region(timer, "backward"):
        loss.backward()

    with torch.profiler.record_function("optimizer_step"), timed_region(timer, "optimizer_step"):
        optimizer.step()

    return loss


def main() -> None:
    device = torch.device(os.getenv("DEVICE", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available")
    sequence_length = int(os.getenv("SEQ_LEN", "512"))
    if sequence_length <= 0:
        raise ValueError("SEQ_LEN must be positive")

    print(f"Running on device: {device}")
    print(f"Sequence length: {sequence_length}")

    config = GPTConfig(
        vocab_size=VOCAB_SIZE,
        max_seq_len=sequence_length,
        d_model=768,
        d_ff=3_072,
        n_heads=12,
        n_layers=12,
        activation_recomputation=os.getenv("ACTIVATION_RECOMPUTATION") == "1",
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
        (BATCH_SIZE, sequence_length),
    ).to(device)

    profile_requested = os.getenv("PROFILE") == "1" or os.getenv("PROFILE_MEMORY") == "1"
    benchmark_requested = os.getenv("BENCHMARK") == "1"

    profile = None
    if profile_requested:
        artifacts = collect_memory_profile(
            step_fn=lambda: train_step(model, optimizer, token_ids),
            output_dir=Path("traces"),
            device=device,
        )
        profile = load_memory_profile(
            artifacts.trace_path,
            artifacts.memory_timeline_path,
            device=artifacts.device,
            peak_allocated_bytes=artifacts.peak_allocated_bytes,
            peak_reserved_bytes=artifacts.peak_reserved_bytes,
        )
    timing = None
    if benchmark_requested:
        if profile_requested:
            model = GPT(config, Embedding(config.vocab_size, config.d_model)).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)
        timing = benchmark_training_step(
            step_fn=lambda timer: train_step(model, optimizer, token_ids, timer),
            device=device,
            warmup_steps=int(os.getenv("WARMUP_STEPS", "1")),
            measured_steps=int(os.getenv("BENCHMARK_STEPS", "3")),
        )
        print(f"step: {timing.step_time_ms:.2f} ms")
        for region, duration_ms in timing.regions_ms.items():
            print(f"{region}: {duration_ms:.2f} ms")
        write_timing(timing, Path("traces/gpt_timing.json"))

    if profile_requested:
        visualizer_path = Path("traces/gpt_memory.html")
        write_memory_visualizer(profile, visualizer_path, timing=timing)
        serve_visualizer(visualizer_path)
    elif not benchmark_requested:
        loss = train_step(model, optimizer, token_ids)
        print(f"step 0: loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
