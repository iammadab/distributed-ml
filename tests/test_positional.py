import unittest

import torch

from layers.positional import LearnedPositionalEmbedding


class LearnedPositionalEmbeddingTest(unittest.TestCase):
    def test_batched_position_lookup(self):
        position_embedding = LearnedPositionalEmbedding(d_model=8, max_seq_len=10)
        token_embeddings = torch.zeros(3, 5, 8)

        output = position_embedding(token_embeddings)

        self.assertEqual(output.shape, (3, 5, 8))
        self.assertTrue(position_embedding.position_embedding.weight.requires_grad)
        self.assertTrue(torch.equal(output[0], output[1]))

    def test_rejects_sequences_longer_than_maximum(self):
        position_embedding = LearnedPositionalEmbedding(d_model=8, max_seq_len=10)
        token_embeddings = torch.zeros(2, 11, 8)

        with self.assertRaises(AssertionError):
            position_embedding(token_embeddings)


if __name__ == "__main__":
    unittest.main()
