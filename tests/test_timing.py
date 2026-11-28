import unittest
from pathlib import Path
import tempfile

import torch

from profiling import Profiler

REGIONS = ("forward", "loss", "backward", "optimizer_step")


class TimingTest(unittest.TestCase):
    def test_disabled_regions_are_transparent(self):
        profiler = Profiler(Path(tempfile.mkdtemp()), torch.device("cpu"))
        value = 0

        with profiler.region("disabled"):
            value += 1

        self.assertEqual(value, 1)

    def test_benchmark_reports_all_regions(self):
        with tempfile.TemporaryDirectory() as directory:
            profiler = Profiler(Path(directory), torch.device("cpu"))

            def step():
                for region in REGIONS:
                    with profiler.region(region):
                        _ = torch.ones(8, 8) @ torch.ones(8, 8)

            timing = profiler.benchmark(step, warmup_steps=1, measured_steps=2)

        self.assertGreater(timing.step_time_ms, 0)
        self.assertEqual(set(timing.regions_ms), set(REGIONS))
        self.assertTrue(all(value >= 0 for value in timing.regions_ms.values()))

    def test_warmup_steps_are_not_returned_as_regions(self):
        calls = 0

        profiler = Profiler(Path(tempfile.mkdtemp()), torch.device("cpu"))

        def step():
            nonlocal calls
            calls += 1
            for region in REGIONS:
                with profiler.region(region):
                    pass

        profiler.benchmark(step, warmup_steps=3, measured_steps=2)

        self.assertEqual(calls, 5)


if __name__ == "__main__":
    unittest.main()
