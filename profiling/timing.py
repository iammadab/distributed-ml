from dataclasses import dataclass
from contextlib import nullcontext
from statistics import median
from time import perf_counter
from typing import Callable

import torch


@dataclass(frozen=True)
class StepTiming:
    step_time_ms: float
    regions_ms: dict[str, float]


class Timer:
    def __init__(self, device: torch.device):
        self.device = device
        self.measurements: list[tuple[str, _Measurement]] = []

    def region(self, name: str):
        measurement = _Measurement(self)
        self.measurements.append((name, measurement))
        return measurement


class _Measurement:
    def __init__(self, timer: Timer):
        self.timer = timer

    def __enter__(self):
        if self.timer.device.type == "cuda":
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
        else:
            self.start = perf_counter()
        return self

    def __exit__(self, *_):
        if self.timer.device.type == "cuda":
            self.end.record()
        else:
            self.end = perf_counter()

    def elapsed_ms(self) -> float:
        if self.timer.device.type == "cuda":
            return self.start.elapsed_time(self.end)
        return (self.end - self.start) * 1_000


def timed_region(timer: Timer | None, name: str):
    return nullcontext() if timer is None else timer.region(name)


def benchmark_training_step(
    step_fn: Callable[[Timer | None], object],
    device: torch.device,
    warmup_steps: int = 1,
    measured_steps: int = 3,
) -> StepTiming:
    """Benchmark training steps without profiler instrumentation."""
    if warmup_steps < 0 or measured_steps <= 0:
        raise ValueError("warmup_steps must be non-negative and measured_steps must be positive")

    for _ in range(warmup_steps):
        step_fn(None)
    _synchronize(device)

    measurements = []
    for _ in range(measured_steps):
        _synchronize(device)
        timer = Timer(device)
        start = perf_counter()
        step_fn(timer)
        _synchronize(device)
        region_timings = {
            name: measurement.elapsed_ms()
            for name, measurement in timer.measurements
        }
        measurements.append(
            StepTiming(
                step_time_ms=(perf_counter() - start) * 1_000,
                regions_ms=region_timings,
            )
        )

    region_names = set(measurements[0].regions_ms)
    if any(set(item.regions_ms) != region_names for item in measurements[1:]):
        raise ValueError("step_fn must measure the same regions on every step")

    return StepTiming(
        step_time_ms=median(item.step_time_ms for item in measurements),
        regions_ms={
            region: median(item.regions_ms[region] for item in measurements)
            for region in region_names
        },
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
