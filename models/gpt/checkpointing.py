import torch
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts


_RECOMPUTED_OPS = {
    torch.ops.aten.mm.default,
    torch.ops.aten.bmm.default,
    torch.ops.aten.matmul.default,
    torch.ops.aten._softmax.default,
}


def selective_attention_context():
    """Recompute attention matrix operations while saving other block ops."""
    def policy(_context, op, *_args, **_kwargs):
        if op in _RECOMPUTED_OPS:
            return CheckpointPolicy.MUST_RECOMPUTE
        return CheckpointPolicy.MUST_SAVE

    return create_selective_checkpoint_contexts(policy)
