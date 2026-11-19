import unittest

import torch

from layers import FFN, GELUFFN


class FeedForwardTest(unittest.TestCase):
    def test_relu_ffn_shape(self):
        layer = FFN(d_model=8, d_ff=16)
        output = layer(torch.randn(2, 5, 8))
        self.assertEqual(output.shape, (2, 5, 8))

    def test_gelu_ffn_shape(self):
        layer = GELUFFN(d_model=8, d_ff=16)
        output = layer(torch.randn(2, 5, 8))
        self.assertEqual(output.shape, (2, 5, 8))


if __name__ == "__main__":
    unittest.main()
