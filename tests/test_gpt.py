import unittest

import torch

from layers import Embedding
from models.gpt import GPT, GPTConfig


class GPTTest(unittest.TestCase):
    def test_forward_pass(self):
        config = GPTConfig(
            vocab_size=100,
            max_seq_len=10,
            d_model=16,
            d_ff=32,
            n_heads=4,
            n_layers=2,
        )
        model = GPT(config)
        token_ids = torch.randint(0, 100, (3, 7))

        logits = model(token_ids)

        self.assertEqual(logits.shape, (3, 7, 100))
        self.assertTrue(torch.isfinite(logits).all())

    def test_shared_embedding_and_output_weights(self):
        config = GPTConfig(
            vocab_size=100,
            max_seq_len=10,
            d_model=16,
            d_ff=32,
            n_heads=4,
            n_layers=2,
            tie_output_weights=True,
        )
        embedding = Embedding(config.vocab_size, config.d_model)
        model = GPT(config, embedding=embedding)

        self.assertIs(model.embedding, embedding)
        self.assertIsNone(model.output_projection)

    def test_identical_batch_items_have_identical_outputs(self):
        config = GPTConfig(
            vocab_size=100,
            max_seq_len=10,
            d_model=16,
            d_ff=32,
            n_heads=4,
            n_layers=2,
        )
        model = GPT(config)
        model.eval()
        token_item = torch.randint(0, 100, (7,))
        logits = model(token_item.unsqueeze(0).repeat(2, 1))

        self.assertTrue(torch.allclose(logits[0], logits[1], rtol=1e-5, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
