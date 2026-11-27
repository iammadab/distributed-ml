from contextlib import ExitStack, nullcontext
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable
import os

import torch

from .collect import ProfileArtifacts, collect_memory_profile
from .memory_data import load_memory_profile
from .server import serve_visualizer
from .timing import StepTiming, Timer, benchmark_training_step, timed_region, write_timing
from .visualizer import write_memory_visualizer


@dataclass(frozen=True)
class ProcessInfo:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str | None = None


@dataclass(frozen=True)
class ProfileResult:
    memory: ProfileArtifacts | None = None
    timing: StepTiming | None = None


class Profiler:
    """Coordinate named timing and PyTorch memory regions for one process."""

    def __init__(
        self,
        output_dir: Path,
        device: torch.device,
        process: ProcessInfo = ProcessInfo(),
        synchronize: Callable[[], None] | None = None,
        metadata: dict[str, object] | None = None,
        capture_memory: bool = False,
        benchmark_enabled: bool = False,
        show_results: bool = True,
    ) -> None:
        self.output_dir = output_dir / f"rank-{process.rank}"
        self.device = device
        self.process = process
        self._synchronize = synchronize
        self._timer: Timer | None = None
        self._record_regions = False
        self._metadata = metadata or {}
        self.capture_memory = capture_memory
        self.benchmark_enabled = benchmark_enabled
        self.show_results = show_results

    @classmethod
    def from_environment(cls, *args, **kwargs):
        return cls(
            *args,
            capture_memory=os.getenv("PROFILE_MEMORY") == "1",
            benchmark_enabled=os.getenv("BENCHMARK") == "1",
            **kwargs,
        )

    def region(self, name: str):
        """Name one region in both timing output and PyTorch profiler traces."""
        if not self._record_regions:
            return nullcontext()
        stack = ExitStack()
        stack.enter_context(torch.profiler.record_function(name))
        stack.enter_context(timed_region(self._timer, name))
        return stack

    def profile_memory(self, step_fn: Callable[[], object]) -> ProfileArtifacts:
        self._synchronize_processes()
        self._record_regions = True
        try:
            return collect_memory_profile(
                step_fn=step_fn,
                output_dir=self.output_dir,
                device=self.device,
            )
        finally:
            self._record_regions = False
            self._synchronize_processes()

    def run(
        self,
        step_factory: Callable[[], Callable[[], object]],
        warmup_steps: int = 1,
        measured_steps: int = 3,
        report: bool = True,
    ) -> ProfileResult:
        """Run enabled captures, rebuilding the step for each capture."""
        if not self.capture_memory and not self.benchmark_enabled:
            return ProfileResult()

        self.write_metadata()
        memory = (
            self.profile_memory(step_factory())
            if self.capture_memory
            else None
        )
        timing = None
        if self.benchmark_enabled:
            timing = self.benchmark(
                step_factory(),
                warmup_steps=warmup_steps,
                measured_steps=measured_steps,
            )
        result = ProfileResult(memory=memory, timing=timing)
        if report and self.show_results:
            self.show(result)
        return result

    def show(self, result: ProfileResult) -> None:
        """Print and display results from rank zero only."""
        if self.process.rank != 0:
            return
        if result.timing is not None:
            print(f"step: {result.timing.step_time_ms:.2f} ms")
            for name, duration_ms in result.timing.regions_ms.items():
                print(f"{name}: {duration_ms:.2f} ms")
        if result.memory is None:
            return
        profile = load_memory_profile(
            result.memory.trace_path,
            result.memory.memory_timeline_path,
            device=result.memory.device,
            peak_allocated_bytes=result.memory.peak_allocated_bytes,
            peak_reserved_bytes=result.memory.peak_reserved_bytes,
        )
        visualizer_path = self.output_dir / "memory.html"
        write_memory_visualizer(profile, visualizer_path, timing=result.timing)
        serve_visualizer(visualizer_path)

    def benchmark(
        self,
        step_fn: Callable[[], object],
        warmup_steps: int = 1,
        measured_steps: int = 3,
    ) -> StepTiming:
        self._synchronize_processes()
        self._record_regions = True

        def timed_step(timer: Timer | None) -> object:
            self._timer = timer
            try:
                return step_fn()
            finally:
                self._timer = None

        try:
            timing = benchmark_training_step(
                step_fn=timed_step,
                device=self.device,
                warmup_steps=warmup_steps,
                measured_steps=measured_steps,
            )
            write_timing(timing, self.output_dir / "timing.json")
            return timing
        finally:
            self._record_regions = False
            self._synchronize_processes()

    def write_metadata(self, **extra: object) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            **asdict(self.process),
            "device": str(self.device),
            **self._metadata,
            **extra,
        }
        (self.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )

    def _synchronize_processes(self) -> None:
        if self._synchronize is not None:
            self._synchronize()
