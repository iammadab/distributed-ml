import os
import json
from pathlib import Path
import re
import subprocess
import sys


SCRIPT = Path(__file__).with_name("train_gpt.py")


def run_candidate(sequence_length: int, recomputation: bool) -> dict:
    environment = os.environ.copy()
    environment.update(
        {
            "BENCHMARK": "1",
            "DEVICE": environment.get("DEVICE", "cuda"),
            "SEQ_LEN": str(sequence_length),
            "WARMUP_STEPS": "0",
            "BENCHMARK_STEPS": "1",
            "ACTIVATION_RECOMPUTATION": "1" if recomputation else "0",
        }
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        env=environment,
        text=True,
        capture_output=True,
    )
    output = result.stdout + result.stderr
    print(output, end="")
    if result.returncode == 0:
        step_match = re.search(r"step: ([0-9.]+) ms", output)
        return {
            "status": "passed",
            "step_time_ms": float(step_match.group(1)) if step_match else None,
        }
    if "out of memory" in output.lower() or "cuda oom" in output.lower():
        return {"status": "oom", "step_time_ms": None}
    raise RuntimeError(f"sequence-length candidate failed unexpectedly: {output}")


def find_limit(
    recomputation: bool,
    start: int,
    maximum: int,
    results: list[dict],
) -> tuple[int, int | None]:
    label = "checkpointed" if recomputation else "baseline"

    def test(sequence_length: int) -> bool:
        result = run_candidate(sequence_length, recomputation)
        results.append({
            "mode": label,
            "seq_len": sequence_length,
            "batch_size": int(os.getenv("BATCH_SIZE", "2")),
            **result,
        })
        return result["status"] == "passed"

    passing = 0
    candidate = start
    while candidate <= maximum and test(candidate):
        passing = candidate
        candidate *= 2

    if candidate > maximum:
        return passing, None

    failing = candidate
    low, high = passing + 1, failing - 1
    while low <= high:
        candidate = (low + high) // 2
        if test(candidate):
            passing = candidate
            low = candidate + 1
        else:
            failing = candidate
            high = candidate - 1
    return passing, failing


def main() -> None:
    start = int(os.getenv("SWEEP_START", "512"))
    maximum = int(os.getenv("SWEEP_MAX", "8192"))
    if start <= 0 or maximum < start:
        raise ValueError("SWEEP_START must be positive and no larger than SWEEP_MAX")

    results = []
    limits = {}
    for recomputation in (False, True):
        label = "checkpointed" if recomputation else "baseline"
        limits[label] = find_limit(recomputation, start, maximum, results)

    output_path = Path(os.getenv("SWEEP_OUTPUT", "traces/seq_len_sweep.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"results": results, "limits": limits}, indent=2) + "\n")

    print("\nLargest passing sequence length")
    for label in ("baseline", "checkpointed"):
        passing, failing = limits[label]
        print(f"{label}: {passing}")
    print(f"\nDetailed results: {output_path}")


if __name__ == "__main__":
    main()
