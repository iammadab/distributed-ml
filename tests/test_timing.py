import unittest

import torch

from profiling import REGIONS, benchmark_training_step


class TimingTest(unittest.TestCase):
    def test_benchmark_reports_all_regions(self):
        def step(timer):
            for region in REGIONS:
                context = timer.region(region) if timer is not None else None
                if context is None:
                    continue
                with context:
                    _ = torch.ones(8, 8) @ torch.ones(8, 8)

        timing = benchmark_training_step(
            step_fn=step,
            device=torch.device("cpu"),
            warmup_steps=1,
            measured_steps=2,
        )

        self.assertGreater(timing.step_time_ms, 0)
        self.assertEqual(set(timing.regions_ms), set(REGIONS))
        self.assertTrue(all(value >= 0 for value in timing.regions_ms.values()))

    def test_warmup_steps_are_not_returned_as_regions(self):
        calls = 0

        def step(timer):
            nonlocal calls
            calls += 1
            if timer is not None:
                for region in REGIONS:
                    with timer.region(region):
                        pass

        benchmark_training_step(step, torch.device("cpu"), warmup_steps=3, measured_steps=2)

        self.assertEqual(calls, 5)


if __name__ == "__main__":
    unittest.main()
