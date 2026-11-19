import unittest

import torch

from models import Transformer


class TransformerTest(unittest.TestCase):
    def test_forward_pass(self):
        model = Transformer(
            source_vocab_size=100,
            target_vocab_size=120,
            d_model=16,
            d_ff=32,
            n_heads=4,
            n_layers=2,
            max_seq_len=10,
        )

        source_token_ids = torch.randint(0, 100, (7,))
        target_token_ids = torch.randint(0, 120, (5,))

        logits = model(source_token_ids, target_token_ids)

        self.assertEqual(logits.shape, (5, 120))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
