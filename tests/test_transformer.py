import unittest

import torch

from layers import Embedding
from models import Transformer, TransformerConfig


class TransformerTest(unittest.TestCase):
    def test_forward_pass(self):
        config = TransformerConfig(
            source_vocab_size=100,
            target_vocab_size=120,
            d_model=16,
            d_ff=32,
            n_heads=4,
            n_layers=2,
            max_seq_len=10,
        )
        model = Transformer(config)

        source_token_ids = torch.randint(0, 100, (3, 7))
        target_token_ids = torch.randint(0, 120, (3, 5))

        logits = model(source_token_ids, target_token_ids)

        self.assertEqual(logits.shape, (3, 5, 120))
        self.assertTrue(torch.isfinite(logits).all())

    def test_shared_embeddings_and_output_weights(self):
        shared_embedding = Embedding(vocab_size=100, d_model=16)
        config = TransformerConfig(
            source_vocab_size=100,
            target_vocab_size=100,
            d_model=16,
            d_ff=32,
            n_heads=4,
            n_layers=2,
            max_seq_len=10,
        )
        model = Transformer(
            config,
            encoder_embedding=shared_embedding,
            decoder_embedding=shared_embedding,
            tie_output_weights=True,
        )

        self.assertIs(model.encoder.embedding, model.decoder.embedding)
        self.assertIs(model.decoder.embedding.weight, shared_embedding.weight)

        source_token_ids = torch.randint(0, 100, (3, 7))
        target_token_ids = torch.randint(0, 100, (3, 5))
        logits = model(source_token_ids, target_token_ids)

        self.assertEqual(logits.shape, (3, 5, 100))
        self.assertTrue(torch.isfinite(logits).all())

    def test_identical_batch_items_have_identical_outputs(self):
        config = TransformerConfig(
            source_vocab_size=100,
            target_vocab_size=120,
            d_model=16,
            d_ff=32,
            n_heads=4,
            n_layers=2,
            max_seq_len=10,
        )
        model = Transformer(config)

        # put the model in eval mode to disable dropout effects
        model.eval()

        source_item = torch.randint(0, 100, (7,))
        target_item = torch.randint(0, 120, (5,))
        source_token_ids = source_item.unsqueeze(0).repeat(2, 1)
        target_token_ids = target_item.unsqueeze(0).repeat(2, 1)

        logits = model(source_token_ids, target_token_ids)

        self.assertTrue(torch.allclose(logits[0], logits[1], rtol=1e-5, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
