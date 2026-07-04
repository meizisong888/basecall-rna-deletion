from __future__ import annotations

import torch

from basecall_deletion.evaluation.ctc import greedy_ctc_decode, greedy_decode_ids


def test_blank_removal_and_repeat_collapse() -> None:
    assert greedy_decode_ids([1, 1, 0, 1, 2, 0, 2, 3]) == "AACC G".replace(" ", "")


def test_greedy_ctc_decode_tensor() -> None:
    path = torch.tensor([1, 1, 0, 1, 2])
    log_probs = torch.full((len(path), 1, 5), -20.0)
    for t, cls in enumerate(path):
        log_probs[t, 0, int(cls)] = 0.0
    assert greedy_ctc_decode(log_probs, torch.tensor([len(path)])) == ["AAC"]

