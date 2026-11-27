import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.train_gpt_ddp import rank_trace_directory, write_profile_metadata


class DDPProfilingTest(unittest.TestCase):
    def test_rank_trace_directory_is_rank_scoped(self):
        self.assertEqual(rank_trace_directory(0), Path("traces/rank-0"))
        self.assertEqual(rank_trace_directory(3), Path("traces/rank-3"))

    def test_profile_metadata_contains_rank_configuration(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "rank-1"
            write_profile_metadata(
                output_dir,
                rank=1,
                local_rank=0,
                world_size=2,
                backend="gloo",
                device=torch.device("cpu"),
                batch_size=2,
                sequence_length=128,
                activation_recomputation="full",
            )

            metadata = json.loads((output_dir / "metadata.json").read_text())

        self.assertEqual(metadata["rank"], 1)
        self.assertEqual(metadata["local_rank"], 0)
        self.assertEqual(metadata["world_size"], 2)
        self.assertEqual(metadata["backend"], "gloo")
        self.assertEqual(metadata["device"], "cpu")
        self.assertEqual(metadata["activation_recomputation"], "full")


if __name__ == "__main__":
    unittest.main()
