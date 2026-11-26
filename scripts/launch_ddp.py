from pathlib import Path
import os
import shutil
import subprocess
import sys

def torchrun_command(processes: int) -> list[str]:
    repository_root = Path(__file__).resolve().parents[1]
    virtualenv_torchrun = repository_root / ".venv" / "bin" / "torchrun"
    torchrun = str(virtualenv_torchrun) if virtualenv_torchrun.exists() else shutil.which("torchrun")
    if torchrun is None:
        raise RuntimeError("could not find torchrun in .venv/bin or PATH")

    return [
        torchrun,
        "--standalone",
        "--nproc_per_node",
        str(processes),
        str(Path(__file__).resolve().with_name("train_gpt_ddp.py")),
    ]

def main() -> int:
    try:
        processes = int(os.getenv("NPROC_PER_NODE", "1"))
    except ValueError as error:
        raise ValueError("NPROC_PER_NODE must be a positive integer") from error

    if processes <= 0:
        raise ValueError("NPROC_PER_NODE must be a positive integer")

    # create a copy of the enivornment and sets the backend to gloo
    # this will change once I add nccl via modal
    environment = os.environ.copy()
    environment["DIST_BACKEND"] = "gloo"

    completed = subprocess.run(
        torchrun_command(processes),
        env=environment,
        check=False,
    )
    return completed.returncode

if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValueError as error:
        print(f"launch_ddp.py: {error}", file=sys.stderr)
        sys.exit(2)
    except RuntimeError as error:
        print(f"launch_ddp.py: {error}", file=sys.stderr)
        sys.exit(1)
