from pathlib import Path
import io
import os
import shutil
import subprocess
import sys
import tarfile

import modal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def training_script() -> str:
    return str(Path(__file__).resolve().with_name("train_gpt_ddp.py"))

def local_torchrun_command(processes: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(processes),
        training_script(),
    ]

def training_variables() -> dict[str, str]:
    return {
        name: os.environ[name]
        for name in (
            "BATCH_SIZE",
            "SEQ_LEN",
            "ACTIVATION_RECOMPUTATION",
            "PROFILE_MEMORY",
            "BENCHMARK",
            "WARMUP_STEPS",
            "BENCHMARK_STEPS",
        )
        if name in os.environ
    }


def create_trace_archive(trace_dir: Path) -> bytes:
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        for path in trace_dir.rglob("*"):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(trace_dir))
    return archive.getvalue()


def extract_trace_archive(archive: bytes, trace_dir: Path) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        tar.extractall(trace_dir)

def configuration() -> tuple[int, int, dict[str, str]]:
    gpu_count = requested_gpu_count()
    process_count = positive_integer("NPROC_PER_NODE", gpu_count)
    if process_count > gpu_count:
        raise ValueError(
            "NPROC_PER_NODE cannot exceed GPU_COUNT for the single-node launcher"
        )
    return gpu_count, process_count, training_variables()

def launch_local_gloo() -> int:
    from profiling import report_profile

    process_count = positive_integer("NPROC_PER_NODE", 1)
    environment = os.environ.copy()
    environment["DIST_BACKEND"] = "gloo"
    completed = subprocess.run(
        local_torchrun_command(process_count),
        env=environment,
        check=False,
    )
    if completed.returncode == 0:
        report_profile(REPOSITORY_ROOT / "traces" / "rank-0")
    return completed.returncode

def modal_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "modal",
        "run",
        str(Path(__file__).resolve()),
    ]

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
    .pip_install("torch>=2.14.1", "numpy>=2.5.4", "plotly>=7.1.0")
    .add_local_dir(
        REPOSITORY_ROOT / "models",
        remote_path="/root/distributed-ml/models",
    )
    .add_local_dir(
        REPOSITORY_ROOT / "layers",
        remote_path="/root/distributed-ml/layers",
    )
    .add_local_dir(
        REPOSITORY_ROOT / "profiling",
        remote_path="/root/distributed-ml/profiling",
    )
    .add_local_file(
        REPOSITORY_ROOT / "scripts" / "train_gpt_ddp.py",
        remote_path="/root/distributed-ml/scripts/train_gpt_ddp.py",
    )
)

app = modal.App("distributed-ml-ddp")

@app.function(gpu=MODAL_GPU, image=image, timeout=60 * 60)
def run_modal_training(
    gpu_count: int,
    process_count: int,
    variables: dict[str, str],
) -> bytes:
    import torch

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(process_count),
        "/root/distributed-ml/scripts/train_gpt_ddp.py",
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
        cwd="/root/distributed-ml",
        env=environment,
        check=True,
    )
    return create_trace_archive(Path("/root/distributed-ml/traces"))

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
            archive = run_modal_training.remote(gpu_count, process_count, variables)
            extract_trace_archive(archive, REPOSITORY_ROOT / "traces")
            print(f"Downloaded Modal traces to {REPOSITORY_ROOT / 'traces'}")
            from profiling import report_profile

            report_profile(REPOSITORY_ROOT / "traces" / "rank-0")
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
