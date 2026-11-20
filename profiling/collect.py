from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


@dataclass(frozen=True)
class ProfileArtifacts:
    trace_path: Path
    memory_timeline_path: Path


def collect_memory_profile(
    step_fn: Callable[[], object],
    output_dir: Path,
) -> ProfileArtifacts:
    """Profile one training step and export CPU memory artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "gpt_memory.json"
    memory_timeline_path = output_dir / "gpt_memory_timeline.json"

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
    ) as profiler:
        step_fn()
        profiler.step()

    profiler.export_chrome_trace(str(trace_path))
    profiler.export_memory_timeline(
        str(memory_timeline_path),
        device="cpu",
    )

    return ProfileArtifacts(
        trace_path=trace_path,
        memory_timeline_path=memory_timeline_path,
    )
