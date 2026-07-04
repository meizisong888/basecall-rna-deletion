from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def split_targets(labels_concat: torch.Tensor, label_lens: torch.Tensor) -> list[torch.Tensor]:
    chunks = []
    cursor = 0
    for length in [int(x) for x in label_lens.detach().cpu().tolist()]:
        chunks.append(labels_concat[cursor : cursor + length])
        cursor += length
    return chunks


def ctc_per_read_loss(log_probs: torch.Tensor, labels_concat: torch.Tensor, input_lens: torch.Tensor, label_lens: torch.Tensor, blank_idx: int = 0) -> torch.Tensor:
    return F.ctc_loss(log_probs, labels_concat, input_lens, label_lens, blank=blank_idx, reduction="none", zero_infinity=True)


def qw_ctc_loss(log_probs: torch.Tensor, labels_concat: torch.Tensor, input_lens: torch.Tensor, label_lens: torch.Tensor, read_weights: torch.Tensor, blank_idx: int = 0) -> torch.Tensor:
    losses = ctc_per_read_loss(log_probs, labels_concat, input_lens, label_lens, blank_idx=blank_idx)
    weights = read_weights.to(device=losses.device, dtype=losses.dtype).clamp(min=0.0)
    return (losses * weights).sum() / weights.sum().clamp(min=1e-8)


def blank_pressure_penalty(log_probs: torch.Tensor, input_lens: torch.Tensor, blank_idx: int = 0) -> torch.Tensor:
    probs = log_probs.exp()
    total = torch.zeros((), device=log_probs.device)
    count = torch.zeros((), device=log_probs.device)
    for batch_idx, length in enumerate(input_lens):
        length_i = int(length.detach().cpu())
        if length_i > 0:
            total = total + probs[:length_i, batch_idx, blank_idx].mean()
            count = count + 1.0
    return total / count.clamp(min=1.0)


def expected_nonblank_length(log_probs: torch.Tensor, input_lens: torch.Tensor, blank_idx: int = 0) -> torch.Tensor:
    probs = log_probs.exp()
    values = []
    for batch_idx, length in enumerate(input_lens):
        length_i = int(length.detach().cpu())
        if length_i <= 0:
            values.append(torch.zeros((), device=log_probs.device))
        else:
            values.append((1.0 - probs[:length_i, batch_idx, blank_idx]).sum())
    return torch.stack(values)


def lenreg_loss(log_probs: torch.Tensor, input_lens: torch.Tensor, label_lens: torch.Tensor, target_ratio: float = 1.0, blank_idx: int = 0) -> torch.Tensor:
    expected = expected_nonblank_length(log_probs, input_lens, blank_idx=blank_idx)
    ratio = expected / label_lens.to(device=log_probs.device, dtype=expected.dtype).clamp(min=1.0)
    target = torch.full_like(ratio, float(target_ratio))
    return F.smooth_l1_loss(ratio, target, reduction="mean")


def training_loss(batch: dict[str, Any], coefficients: dict[str, float]) -> tuple[torch.Tensor, dict[str, float]]:
    log_probs = batch["log_probs"]
    labels = batch["labels"]
    input_lens = batch["input_lens"]
    label_lens = batch["label_lens"]
    weights = batch["weights"]
    blank_idx = int(coefficients.get("blank_idx", 0))
    loss = qw_ctc_loss(log_probs, labels, input_lens, label_lens, weights, blank_idx=blank_idx)
    parts = {"qw_ctc": float(loss.detach().cpu())}
    bp = float(coefficients.get("blank_penalty", 0.0))
    if bp:
        bp_loss = blank_pressure_penalty(log_probs, input_lens, blank_idx=blank_idx)
        loss = loss + bp * bp_loss
        parts["blank_penalty"] = float(bp_loss.detach().cpu())
    len_coef = float(coefficients.get("lenreg", 0.0))
    if len_coef:
        length_loss = lenreg_loss(log_probs, input_lens, label_lens, target_ratio=float(coefficients.get("lenreg_target", 1.0)), blank_idx=blank_idx)
        loss = loss + len_coef * length_loss
        parts["lenreg"] = float(length_loss.detach().cpu())
    parts["total"] = float(loss.detach().cpu())
    return loss, parts

