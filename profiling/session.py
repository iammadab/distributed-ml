from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
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
class ProfileResult:
    memory: ProfileArtifacts | None = None
    timing: StepTiming | None = None
    step_result: object | None = None


class Profiler:
    """Coordinate named timing and PyTorch memory regions for one process."""

    def __init__(
        self,
        output_dir: Path,
        device: torch.device,
        synchronize: Callable[[], None] | None = None,
        metadata: dict[str, object] | None = None,
        capture_memory: bool = False,
        benchmark_enabled: bool = False,
    ) -> None:
        self.output_dir = output_dir
        self.device = device
        self._synchronize = synchronize
        self._timer: Timer | None = None
        self._record_regions = False
        self._metadata = metadata or {}
        self.capture_memory = capture_memory
        self.benchmark_enabled = benchmark_enabled
        self._last_result: ProfileResult | None = None

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
    ) -> ProfileResult:
        """Run enabled captures, rebuilding the step for each capture."""
        if not self.capture_memory and not self.benchmark_enabled:
            result = ProfileResult(step_result=step_factory()())
            self._last_result = result
            return result

        self.write_metadata()
        observed_results: list[object] = []

        def observed_step_factory():
            step = step_factory()

            def observed_step():
                result = step()
                observed_results.append(result)
                return result

            return observed_step

        memory = (
            self.profile_memory(observed_step_factory())
            if self.capture_memory
            else None
        )
        timing = None
        if self.benchmark_enabled:
            timing = self.benchmark(
                observed_step_factory(),
                warmup_steps=warmup_steps,
                measured_steps=measured_steps,
            )
        result = ProfileResult(
            memory=memory,
            timing=timing,
            step_result=observed_results[-1] if observed_results else None,
        )
        self._last_result = result
        return result

    def report(self, result: ProfileResult | None = None) -> None:
        """Print and display the latest local profiling result."""
        result = result or self._last_result
        if result is None:
            raise RuntimeError("run() must be called before report()")
        if result.timing is not None:
            print(f"step: {result.timing.step_time_ms:.2f} ms")
            for name, duration_ms in result.timing.regions_ms.items():
                print(f"{name}: {duration_ms:.2f} ms")
        if result.step_result is not None and isinstance(result.step_result, torch.Tensor):
            value = result.step_result.item() if result.step_result.numel() == 1 else result.step_result
            print(f"step result: {value}")
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
