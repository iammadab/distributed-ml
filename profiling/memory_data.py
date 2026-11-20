import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MEMORY_CATEGORIES = (
    "parameters",
    "optimizer_state",
    "input",
    "temporary",
    "activation",
    "gradient",
    "autograd_detail",
    "unclassified",
)


@dataclass(frozen=True)
class MemoryPoint:
    time_ms: float
    total_bytes: int
    categories: dict[str, int]


@dataclass(frozen=True)
class Phase:
    name: str
    start_ms: float
    end_ms: float


@dataclass(frozen=True)
class MemoryProfile:
    points: list[MemoryPoint]
    phases: list[Phase]
    peak_memory_bytes: int
    duration_ms: float


def load_memory_profile(
    trace_path: Path,
    memory_timeline_path: Path,
) -> MemoryProfile:
    """Load profiler output into data structures used by the visualizer."""
    trace = _load_json(trace_path)
    timeline = _load_json(memory_timeline_path)

    timestamps, memory_values = timeline
    if not timestamps or not memory_values:
        raise ValueError("memory timeline does not contain any points")
    if len(timestamps) != len(memory_values):
        raise ValueError("memory timeline timestamps and values have different lengths")

    start_timestamp = timestamps[0]
    points = [
        MemoryPoint(
            time_ms=(timestamp - start_timestamp) / 1_000,
            total_bytes=sum(int(value) for value in values[1:]),
            categories={
                category: int(values[index])
                for index, category in enumerate(MEMORY_CATEGORIES, start=1)
            },
        )
        for timestamp, values in zip(timestamps, memory_values)
    ]

    phases = _load_phases(trace, start_timestamp)
    return MemoryProfile(
        points=points,
        phases=phases,
        peak_memory_bytes=max(point.total_bytes for point in points),
        duration_ms=points[-1].time_ms,
    )


def _load_json(path: Path) -> Any:
    with path.open() as file:
        return json.load(file)


def _load_phases(trace: dict[str, Any], start_timestamp: int) -> list[Phase]:
    phase_names = {"forward", "loss", "backward", "optimizer_step"}
    phases = []
    trace_start_timestamp = trace.get("baseTimeNanoseconds", 0) / 1_000

    for event in trace.get("traceEvents", []):
        if event.get("name") not in phase_names or event.get("ph") != "X":
            continue

        event_start = trace_start_timestamp + event["ts"]
        phases.append(
            Phase(
                name=event["name"],
                start_ms=(event_start - start_timestamp) / 1_000,
                end_ms=(event_start + event["dur"] - start_timestamp) / 1_000,
            )
        )

    return sorted(phases, key=lambda phase: phase.start_ms)
