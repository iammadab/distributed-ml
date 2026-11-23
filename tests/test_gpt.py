import unittest

import torch
import torch.nn.functional as F

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
        model = GPT(config, Embedding(config.vocab_size, config.d_model))
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
        )
        embedding = Embedding(config.vocab_size, config.d_model)
        model = GPT(config, embedding=embedding)

        self.assertIs(model.embedding, embedding)

    def test_identical_batch_items_have_identical_outputs(self):
        config = GPTConfig(
            vocab_size=100,
            max_seq_len=10,
            d_model=16,
            d_ff=32,
            n_heads=4,
            n_layers=2,
        )
        model = GPT(config, Embedding(config.vocab_size, config.d_model))
        model.eval()
        token_item = torch.randint(0, 100, (7,))
        logits = model(token_item.unsqueeze(0).repeat(2, 1))

        self.assertTrue(torch.allclose(logits[0], logits[1], rtol=1e-5, atol=3e-6))

    def test_activation_recomputation_matches_training_update(self):
        config = GPTConfig(
            vocab_size=100,
            max_seq_len=10,
            d_model=16,
            d_ff=32,
            n_heads=4,
            n_layers=2,
            dropout=0.0,
        )
        baseline = GPT(config, Embedding(config.vocab_size, config.d_model))
        recomputed_config = GPTConfig(
            vocab_size=100,
            max_seq_len=10,
            d_model=16,
            d_ff=32,
            n_heads=4,
            n_layers=2,
            dropout=0.0,
            activation_recomputation="full",
        )
        recomputed = GPT(
            recomputed_config,
            Embedding(config.vocab_size, config.d_model),
        )
        recomputed.load_state_dict(baseline.state_dict())

        baseline_optimizer = torch.optim.Adam(baseline.parameters(), lr=1e-3)
        recomputed_optimizer = torch.optim.Adam(recomputed.parameters(), lr=1e-3)
        token_ids = torch.randint(0, config.vocab_size, (3, 7))
        targets = torch.randint(0, config.vocab_size, (3, 7))

        baseline_logits = baseline(token_ids)
        recomputed_logits = recomputed(token_ids)
        self.assertTrue(torch.allclose(baseline_logits, recomputed_logits))

        baseline_loss = F.cross_entropy(baseline_logits.reshape(-1, 100), targets.reshape(-1))
        recomputed_loss = F.cross_entropy(recomputed_logits.reshape(-1, 100), targets.reshape(-1))
        baseline_loss.backward()
        recomputed_loss.backward()

        for baseline_parameter, recomputed_parameter in zip(
            baseline.parameters(), recomputed.parameters()
        ):
            self.assertTrue(torch.allclose(baseline_parameter.grad, recomputed_parameter.grad))

        baseline_optimizer.step()
        recomputed_optimizer.step()

        for baseline_parameter, recomputed_parameter in zip(
            baseline.parameters(), recomputed.parameters()
        ):
            self.assertTrue(torch.allclose(baseline_parameter, recomputed_parameter))

    def test_selective_checkpointing_matches_forward_and_gradients(self):
        config = GPTConfig(
            vocab_size=100,
            max_seq_len=10,
            d_model=16,
            d_ff=32,
            n_heads=4,
            n_layers=2,
            dropout=0.0,
        )
        selective_config = GPTConfig(
            vocab_size=100,
            max_seq_len=10,
            d_model=16,
            d_ff=32,
            n_heads=4,
            n_layers=2,
            dropout=0.0,
            activation_recomputation="selective",
        )
        baseline = GPT(config, Embedding(config.vocab_size, config.d_model))
        selective = GPT(
            selective_config,
            Embedding(config.vocab_size, config.d_model),
        )
        selective.load_state_dict(baseline.state_dict())
        token_ids = torch.randint(0, config.vocab_size, (2, 7))
        targets = torch.randint(0, config.vocab_size, (2, 7))

        baseline_loss = F.cross_entropy(
            baseline(token_ids).reshape(-1, 100), targets.reshape(-1)
        )
        selective_loss = F.cross_entropy(
            selective(token_ids).reshape(-1, 100), targets.reshape(-1)
        )
        baseline_loss.backward()
        selective_loss.backward()

        self.assertTrue(torch.allclose(baseline_loss, selective_loss))
        for baseline_parameter, selective_parameter in zip(
            baseline.parameters(), selective.parameters()
        ):
            self.assertTrue(torch.allclose(baseline_parameter.grad, selective_parameter.grad))


if __name__ == "__main__":
    unittest.main()
