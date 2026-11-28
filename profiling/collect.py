from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


@dataclass(frozen=True)
class ProfileArtifacts:
    trace_path: Path
    memory_timeline_path: Path
    device: str
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None


def collect_memory_profile(
    step_fn: Callable[[], object],
    output_dir: Path,
    device: torch.device,
    trace_name: str = "gpt_memory.json",
    memory_timeline_name: str = "gpt_memory_timeline.json",
) -> ProfileArtifacts:
    """Profile one training step and export device-specific memory artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / trace_name
    memory_timeline_path = output_dir / memory_timeline_name

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    with torch.profiler.profile(
        activities=activities,
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
    ) as profiler:
        step_fn()
        profiler.step()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
        peak_reserved_bytes = torch.cuda.max_memory_reserved(device)
    else:
        peak_allocated_bytes = None
        peak_reserved_bytes = None

    profiler.export_chrome_trace(str(trace_path))
    memory_device = str(device)
    if device.type == "cuda":
        memory_device = f"cuda:{device.index or 0}"
    profiler.export_memory_timeline(
        str(memory_timeline_path),
        device=memory_device,
    )

    return ProfileArtifacts(
        trace_path=trace_path,
        memory_timeline_path=memory_timeline_path,
        device=str(device),
        peak_allocated_bytes=peak_allocated_bytes,
        peak_reserved_bytes=peak_reserved_bytes,
    )
