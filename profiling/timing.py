from dataclasses import dataclass
from statistics import median
from time import perf_counter
from typing import Callable

import torch


@dataclass(frozen=True)
class StepTiming:
    step_time_ms: float
    regions_ms: dict[str, float]

    @property
    def forward_time_ms(self) -> float:
        return self.regions_ms["forward"]

    @property
    def loss_time_ms(self) -> float:
        return self.regions_ms["loss"]

    @property
    def backward_time_ms(self) -> float:
        return self.regions_ms["backward"]

    @property
    def optimizer_step_time_ms(self) -> float:
        return self.regions_ms["optimizer_step"]


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
        start = perf_counter()
        timer = Timer(device)
        step_fn(timer)
        _synchronize(device)
        region_timings = {
            name: measurement.elapsed_ms()
            for name, measurement in timer.measurements
        }
        expected_regions = {"forward", "loss", "backward", "optimizer_step"}
        if set(region_timings) != expected_regions:
            raise ValueError(f"step_fn must measure {sorted(expected_regions)}")
        measurements.append(
            StepTiming(
                step_time_ms=(perf_counter() - start) * 1_000,
                regions_ms=region_timings,
            )
        )

    return StepTiming(
        step_time_ms=median(item.step_time_ms for item in measurements),
        regions_ms={
            region: median(item.regions_ms[region] for item in measurements)
            for region in ("forward", "loss", "backward", "optimizer_step")
        },
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
