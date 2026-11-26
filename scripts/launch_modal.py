from pathlib import Path
import os
import subprocess
import sys

import modal

MAX_GPUS_PER_NODE = 8
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# get the number of gpu that will participate in the distributed 
# training run
def requested_gpu_count() -> int:
    value = os.getenv("GPU_COUNT", "1")
    try:
        count = int(value)
    except ValueError as error:
        raise ValueError("GPU_COUNT must be a positive integer") from error
    if count <= 0:
        raise ValueError("GPU_COUNT must be a positive integer")
    if count > MAX_GPUS_PER_NODE:
        raise ValueError(
            f"GPU_COUNT={count} exceeds the single-node limit of "
            f"{MAX_GPUS_PER_NODE}; multi-node support is not implemented"
        )
    return count

MODAL_GPU_COUNT = requested_gpu_count()
MODAL_GPU = "A100" if MODAL_GPU_COUNT == 1 else f"A100:{MODAL_GPU_COUNT}"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.14.1", "numpy>=2.5.4")
    .add_local_dir(
        REPOSITORY_ROOT / "models",
        remote_path="/root/transformer-models/models",
    )
    .add_local_dir(
        REPOSITORY_ROOT / "layers",
        remote_path="/root/transformer-models/layers",
    )
    .add_local_file(
        REPOSITORY_ROOT / "scripts" / "train_gpt_ddp.py",
        remote_path="/root/transformer-models/scripts/train_gpt_ddp.py",
    )
)

app = modal.App("transformer-models-ddp")

# returns the following
# gpu_count:            total number of gpus participating in the distributed training run
# process_count:        number of gpus per node, gpu_count / process_count implies the number of nodes
# training_variables:   certain varibales needed to modify how training executes
def configuration() -> tuple[int, int, dict[str, str]]:
    process_count = positive_integer("NPROC_PER_NODE", MODAL_GPU_COUNT)
    if process_count > MODAL_GPU_COUNT:
        raise ValueError(
            "NPROC_PER_NODE cannot exceed GPU_COUNT for the single-node launcher"
        )

    training_variables = {
        name: os.environ[name]
        for name in ("BATCH_SIZE", "SEQ_LEN", "ACTIVATION_RECOMPUTATION")
        if name in os.environ
    }
    return MODAL_GPU_COUNT, process_count, training_variables


@app.function(gpu=MODAL_GPU, image=image, timeout=60 * 60)
def run_training(gpu_count: int, process_count: int, training_variables: dict[str, str]) -> None:
    import torch

    # invoke torch run on the node, spins up the required number of workers
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(process_count),
        "/root/transformer-models/scripts/train_gpt_ddp.py",
    ]
    environment = os.environ.copy()
    environment.update(training_variables)
    environment["DIST_BACKEND"] = "nccl"

    # Fail explicitly if Modal's allocation does not match the request.
    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count != gpu_count:
        raise RuntimeError(
            f"Modal provided {visible_gpu_count} visible GPUs, expected {gpu_count}"
        )

    print(f"Running NCCL DDP on {gpu_count} A100 GPUs")
    print(f"Processes per node: {process_count}")
    subprocess.run(command, cwd="/root/transformer-models", env=environment, check=True)


@app.local_entrypoint()
def main() -> None:
    try:
        gpu_count, process_count, training_variables = configuration()
    except ValueError as error:
        print(f"launch_modal.py: {error}", file=sys.stderr)
        raise SystemExit(2) from error

    run_training.remote(gpu_count, process_count, training_variables)

def positive_integer(name: str, default: int) -> int:
    value = os.getenv(name, str(default))
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed
