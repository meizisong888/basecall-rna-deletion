from __future__ import annotations

from typing import Any

import torch
from torch import nn


def parse_int_list(value: str | list[int] | tuple[int, ...], default: list[int]) -> list[int]:
    if isinstance(value, (list, tuple)):
        parsed = [int(v) for v in value]
    else:
        text = str(value).strip()
        parsed = [int(part) for part in text.split(",") if part.strip()] if text else list(default)
    return parsed or list(default)


class MultiScaleTemporalAdapter(nn.Module):
    """Residual multi-scale temporal adapter over `[batch, time, channels]` features."""

    def __init__(
        self,
        hidden_dim: int,
        kernels: str | list[int] | tuple[int, ...] = "5,9,17,33",
        dilations: str | list[int] | tuple[int, ...] = "1,2,4,8",
        branch_mode: str = "kernels",
        dropout: float = 0.1,
        conv_norm: str = "group",
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.kernels = parse_int_list(kernels, [5, 9, 17, 33])
        self.dilations = parse_int_list(dilations, [1, 2, 4, 8])
        self.branch_mode = str(branch_mode)
        if self.branch_mode not in {"kernels", "dilated"}:
            raise ValueError("branch_mode must be `kernels` or `dilated`")
        specs = [(5, dilation) for dilation in self.dilations] if self.branch_mode == "dilated" else [(kernel, 1) for kernel in self.kernels]
        self.specs = [(kernel if kernel % 2 else kernel + 1, dilation) for kernel, dilation in specs]
        if not self.specs:
            raise ValueError("at least one temporal branch is required")

        branches = []
        for kernel, dilation in self.specs:
            padding = (kernel // 2) * dilation
            if conv_norm == "batch":
                norm = nn.BatchNorm1d(self.hidden_dim)
            else:
                groups = 8 if self.hidden_dim % 8 == 0 else 1
                norm = nn.GroupNorm(groups, self.hidden_dim)
            branches.append(
                nn.Sequential(
                    nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=kernel, padding=padding, dilation=dilation, groups=self.hidden_dim),
                    norm,
                    nn.SiLU(inplace=True),
                    nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                )
            )
        self.input_norm = nn.LayerNorm(self.hidden_dim)
        self.branches = nn.ModuleList(branches)
        self.output_norm = nn.LayerNorm(self.hidden_dim * len(self.branches))
        self.output = nn.Linear(self.hidden_dim * len(self.branches), self.hidden_dim)
        nn.init.normal_(self.output.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.output.bias)

    def mean_vector_norm(self, x: torch.Tensor) -> float:
        return 0.0 if x.numel() == 0 else float(x.detach().norm(dim=-1).mean().cpu())

    def forward(self, h: torch.Tensor, padding_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
        y = self.input_norm(h).transpose(1, 2)
        branch_outputs = []
        stats: dict[str, Any] = {}
        for idx, (branch, (kernel, dilation)) in enumerate(zip(self.branches, self.specs)):
            out = branch(y).transpose(1, 2)
            if padding_mask is not None:
                out = out.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            branch_outputs.append(out)
            stats[f"branch_{idx}_kernel"] = int(kernel)
            stats[f"branch_{idx}_dilation"] = int(dilation)
            stats[f"branch_{idx}_norm"] = self.mean_vector_norm(out)
        merged = torch.cat(branch_outputs, dim=-1)
        context = self.output(self.output_norm(merged))
        if padding_mask is not None:
            context = context.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        stats["context_norm"] = self.mean_vector_norm(context)
        return context, stats
