import unittest

import torch

from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2


class _NativeArgmaxHead:
    def __init__(self):
        self.weight = torch.empty((4, 2), dtype=torch.float32)
        self.calls: list[torch.Tensor] = []

    def compute_argmax_token(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.calls.append(hidden_states.clone())
        return hidden_states[:, 0].to(torch.long) + 100


class TestDFlashNativeArgmaxHead(unittest.TestCase):
    def test_prefers_native_tp_argmax_and_preserves_chunk_order(self):
        worker = object.__new__(DFlashWorkerV2)
        head = _NativeArgmaxHead()
        hidden_states = torch.tensor(
            [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]], dtype=torch.float16
        )

        result = worker._greedy_sample_from_vocab_parallel_head(
            hidden_states=hidden_states,
            lm_head=head,
            chunk_size=2,
        )

        self.assertEqual(result.tolist(), [100, 101, 102, 103, 104])
        self.assertEqual([call.shape[0] for call in head.calls], [2, 2, 1])
        self.assertTrue(all(call.dtype == torch.float32 for call in head.calls))

    def test_keeps_unsharded_fallback_for_regular_heads(self):
        worker = object.__new__(DFlashWorkerV2)
        head = torch.nn.Linear(2, 3, bias=False)
        head.weight.data.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
        )

        result = worker._greedy_sample_from_vocab_parallel_head(
            hidden_states=torch.tensor([[3.0, 1.0], [1.0, 4.0]]),
            lm_head=head,
            chunk_size=1,
        )

        self.assertEqual(result.tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
