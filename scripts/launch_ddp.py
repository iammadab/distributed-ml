from pathlib import Path
import os
import shutil
import subprocess
import sys

import modal


MAX_GPUS_PER_NODE = 8
MODAL_INVOCATION = "MODAL_INVOCATION"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def positive_integer(name: str, default: int) -> int:
    value = os.getenv(name, str(default))
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def requested_gpu_count() -> int:
    count = positive_integer("GPU_COUNT", 1)
    if count > MAX_GPUS_PER_NODE:
        raise ValueError(
            f"GPU_COUNT={count} exceeds the single-node limit of "
            f"{MAX_GPUS_PER_NODE}; multi-node support is not implemented"
        )
    return count


def local_torchrun() -> str:
    virtualenv_torchrun = REPOSITORY_ROOT / ".venv" / "bin" / "torchrun"
    torchrun = (
        str(virtualenv_torchrun)
        if virtualenv_torchrun.exists()
        else shutil.which("torchrun")
    )
    if torchrun is None:
        raise RuntimeError("could not find torchrun in .venv/bin or PATH")
    return torchrun


def training_script() -> str:
    return str(Path(__file__).resolve().with_name("train_gpt_ddp.py"))


def local_torchrun_command(processes: int) -> list[str]:
    return [
        local_torchrun(),
        "--standalone",
        "--nproc_per_node",
        str(processes),
        training_script(),
    ]


def training_variables() -> dict[str, str]:
    return {
        name: os.environ[name]
        for name in ("BATCH_SIZE", "SEQ_LEN", "ACTIVATION_RECOMPUTATION")
        if name in os.environ
    }


def configuration() -> tuple[int, int, dict[str, str]]:
    gpu_count = requested_gpu_count()
    process_count = positive_integer("NPROC_PER_NODE", gpu_count)
    if process_count > gpu_count:
        raise ValueError(
            "NPROC_PER_NODE cannot exceed GPU_COUNT for the single-node launcher"
        )
    return gpu_count, process_count, training_variables()


def launch_local_gloo() -> int:
    process_count = positive_integer("NPROC_PER_NODE", 1)
    environment = os.environ.copy()
    environment["DIST_BACKEND"] = "gloo"
    completed = subprocess.run(
        local_torchrun_command(process_count),
        env=environment,
        check=False,
    )
    return completed.returncode


def modal_command() -> list[str]:
    modal_executable = shutil.which("modal")
    if modal_executable is None:
        raise RuntimeError("could not find modal in PATH")
    return [modal_executable, "run", str(Path(__file__).resolve())]


def launch_modal() -> int:
    environment = os.environ.copy()
    environment[MODAL_INVOCATION] = "1"
    completed = subprocess.run(
        modal_command(),
        env=environment,
        check=False,
    )
    return completed.returncode


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


@app.function(gpu=MODAL_GPU, image=image, timeout=60 * 60)
def run_modal_training(
    gpu_count: int,
    process_count: int,
    variables: dict[str, str],
) -> None:
    import torch

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
    environment.update(variables)
    environment["DIST_BACKEND"] = "nccl"

    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count != gpu_count:
        raise RuntimeError(
            f"Modal provided {visible_gpu_count} visible GPUs, expected {gpu_count}"
        )

    print(f"Running NCCL DDP on {gpu_count} A100 GPUs")
    print(f"Processes per node: {process_count}")
    subprocess.run(
        command,
        cwd="/root/transformer-models",
        env=environment,
        check=True,
    )


@app.local_entrypoint()
def main() -> None:
    backend = os.getenv("DIST_BACKEND", "gloo").lower()
    try:
        if backend == "gloo":
            sys.exit(launch_local_gloo())
        if backend == "nccl":
            if os.getenv(MODAL_INVOCATION) != "1":
                sys.exit(launch_modal())

            gpu_count, process_count, variables = configuration()
            run_modal_training.remote(gpu_count, process_count, variables)
            return
        raise ValueError("DIST_BACKEND must be either 'gloo' or 'nccl'")
    except ValueError as error:
        print(f"launch_ddp.py: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    except RuntimeError as error:
        print(f"launch_ddp.py: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
