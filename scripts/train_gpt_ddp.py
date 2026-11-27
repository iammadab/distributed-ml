from pathlib import Path
from contextlib import nullcontext
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

# Make the repository's top-level packages available when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layers import Embedding
from models.gpt import GPT, GPTConfig
from profiling import (
    Profiler,
)

VOCAB_SIZE = 40_478
DEFAULT_BATCH_SIZE = 2

'''
this returns the rank, local rank and the world size
essentially tells us if we are in a distributed environment
'''
def distributed_environment() -> tuple[int, int, int, bool]:
    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    is_distributed = any(name in os.environ for name in required)
    if not is_distributed:
        return 0, 0, 1, False

    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(
            "distributed execution requires RANK, LOCAL_RANK, and WORLD_SIZE; "
            f"missing {', '.join(missing)}"
        )

    return (
        int(os.environ["RANK"]),
        int(os.environ["LOCAL_RANK"]),
        int(os.environ["WORLD_SIZE"]),
        True,
    )


'''
returns the torch device
'''
def select_device(backend: str, local_rank: int) -> torch.device:
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL requires CUDA, but CUDA is not available")
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} exceeds available GPUs "
                f"({torch.cuda.device_count()})"
            )
        return torch.device("cuda", local_rank)

    if backend == "gloo":
        return torch.device("cpu")

    raise ValueError(f"unsupported distributed backend: {backend!r}")


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    token_ids: torch.Tensor,
    profiler: Profiler | None = None,
) -> torch.Tensor:
    inputs = token_ids[:, :-1]
    targets = token_ids[:, 1:]

    optimizer.zero_grad()
    with profiler.region("forward") if profiler is not None else nullcontext():
        logits = model(inputs)
    with profiler.region("loss") if profiler is not None else nullcontext():
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
    with profiler.region("backward") if profiler is not None else nullcontext():
        loss.backward()
    with profiler.region("optimizer_step") if profiler is not None else nullcontext():
        optimizer.step()
    return loss.detach()


def main() -> None:
    # default the backend to gloo
    backend = os.getenv("DIST_BACKEND", "gloo").lower()
    # get the environment info
    rank, local_rank, world_size, is_distributed = distributed_environment()
    # return the torch device, if cuda also returns the local rank
    device = select_device(backend, local_rank)

    if is_distributed:
        # init the process group for things like communication
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )

    try:
        sequence_length = int(os.getenv("SEQ_LEN", "512"))
        batch_size = int(os.getenv("BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))

        if sequence_length <= 0:
            raise ValueError("SEQ_LEN must be positive")
        if batch_size <= 0:
            raise ValueError("BATCH_SIZE must be positive")

        torch.manual_seed(0)
        if device.type == "cuda":
            torch.cuda.manual_seed(0)

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

        if is_distributed:
            # for a distributed run we wrap the model in pytorch ddp
            model = DistributedDataParallel(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
            )

        optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)

        token_ids = torch.randint(
            0,
            config.vocab_size,
            (batch_size, sequence_length),
            device=device,
        )

        observed_losses: list[torch.Tensor] = []
        def synchronize() -> None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if is_distributed:
                dist.barrier()

        profiler = Profiler.from_environment(
            output_dir=Path("traces") / f"rank-{rank}",
            device=device,
            synchronize=synchronize,
            metadata={
                "rank": rank,
                "local_rank": local_rank,
                "world_size": world_size,
                "backend": backend,
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "activation_recomputation": config.activation_recomputation,
            },
        )

        def make_step():
            nonlocal model, optimizer
            torch.manual_seed(0)
            if device.type == "cuda":
                torch.cuda.manual_seed(0)
            model = GPT(config, Embedding(config.vocab_size, config.d_model)).to(device)
            if is_distributed:
                model = DistributedDataParallel(
                    model,
                    device_ids=[local_rank] if device.type == "cuda" else None,
                )
            optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)
            return lambda: observed_losses.append(
                train_step(model, optimizer, token_ids, profiler)
            )

        result = profiler.run(
            step_factory=make_step,
            warmup_steps=int(os.getenv("WARMUP_STEPS", "1")),
            measured_steps=int(os.getenv("BENCHMARK_STEPS", "3")),
        )
        loss = result.step_result
        if loss is None:
            raise RuntimeError("profiler did not execute a training step")

        if is_distributed:
            # Note: this is only for reporting
            # computes the loss across ranks for reporting
            # costs an extra all-reduce
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= world_size
            dist.barrier()

        if rank == 0:
            profiler.report()

        if rank == 0:
            print(f"Running on device: {device}")
            print(f"World size: {world_size}")
            print(f"Local batch size: {batch_size}")
            print(f"Sequence length: {sequence_length}")
            print(f"step 0: loss: {loss.item():.4f}")

    finally:
        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
