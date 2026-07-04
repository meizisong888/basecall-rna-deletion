from __future__ import annotations

from typing import Iterable

import torch

ID_TO_BASE = {0: "_", 1: "A", 2: "C", 3: "G", 4: "T"}


def greedy_decode_ids(ids: Iterable[int], blank_idx: int = 0) -> str:
    decoded = []
    prev = None
    for item in ids:
        item = int(item)
        if item != blank_idx and item != prev:
            decoded.append(ID_TO_BASE.get(item, ""))
        prev = item
    return "".join(decoded)


def greedy_ctc_decode(log_probs: torch.Tensor, input_lens: torch.Tensor, blank_idx: int = 0) -> list[str]:
    argmax = log_probs.detach().cpu().argmax(dim=-1).transpose(0, 1)
    lengths = [int(x) for x in input_lens.detach().cpu().tolist()]
    return [greedy_decode_ids(row[:length].tolist(), blank_idx=blank_idx) for row, length in zip(argmax, lengths)]
