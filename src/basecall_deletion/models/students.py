from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .mta import MultiScaleTemporalAdapter


def group_norm(channels: int) -> nn.GroupNorm:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


def conv1d_out_lengths(lengths: torch.Tensor, kernel_size: int = 5, stride: int = 1, padding: int = 2, dilation: int = 1) -> torch.Tensor:
    return torch.div(lengths + 2 * padding - dilation * (kernel_size - 1) - 1, stride, rounding_mode="floor") + 1


def strides_for_downsample_factor(factor: int) -> list[int]:
    factor = int(factor)
    if factor == 1:
        return [1]
    if factor == 2:
        return [2]
    if factor == 4:
        return [2, 2]
    if factor == 8:
        return [2, 2, 2]
    raise ValueError("downsample_factor must be one of 1, 2, 4, 8")


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        kernel = int(kernel_size)
        if kernel % 2 == 0:
            kernel += 1
        padding = int(dilation) * (kernel - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel, padding=padding, dilation=int(dilation))
        self.norm1 = group_norm(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm2 = group_norm(channels)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.silu(self.norm1(self.conv1(x)))
        y = self.dropout(y)
        y = self.norm2(self.conv2(y))
        return F.silu(x + y)


class ConvFrontend(nn.Module):
    def __init__(self, input_channels: int, hidden_dim: int, downsample_factor: int) -> None:
        super().__init__()
        strides = strides_for_downsample_factor(downsample_factor)
        if len(strides) == 1:
            channels = [hidden_dim]
        elif len(strides) == 2:
            channels = [max(64, hidden_dim // 2), hidden_dim]
        else:
            channels = [max(64, hidden_dim // 4), max(128, hidden_dim // 2), hidden_dim]
        layers = []
        in_channels = int(input_channels)
        for out_channels, stride in zip(channels, strides):
            layers.extend([nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2), group_norm(out_channels), nn.SiLU(inplace=True)])
            in_channels = out_channels
        self.layers = nn.Sequential(*layers)
        self.downsample_factor = int(downsample_factor)

    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        out = lengths.long()
        for stride in strides_for_downsample_factor(self.downsample_factor):
            out = conv1d_out_lengths(out, stride=int(stride)).clamp(min=1)
        return out

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"expected [batch, time] or [batch, channels, time], got {tuple(x.shape)}")
        return self.layers(x).transpose(1, 2), self.output_lengths(lengths.to(x.device))


class LSTMStudentCTC(nn.Module):
    def __init__(self, input_channels: int = 1, hidden_dim: int = 384, lstm_hidden: int = 384, num_layers: int = 4, downsample_factor: int = 4, dropout: float = 0.1, vocab_size: int = 5) -> None:
        super().__init__()
        self.frontend = ConvFrontend(input_channels, hidden_dim, downsample_factor)
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=float(dropout) if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(lstm_hidden * 2, vocab_size)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        features, out_lengths = self.frontend(x, lengths)
        features, _ = self.lstm(features)
        logits = self.classifier(self.dropout(features))
        log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1).contiguous()
        return log_probs, out_lengths.clamp(max=log_probs.shape[0]), {"model": "lstm"}


class TCNStudentCTC(nn.Module):
    def __init__(self, input_channels: int = 1, hidden_dim: int = 512, num_layers: int = 10, downsample_factor: int = 4, conv_kernel: int = 5, dropout: float = 0.1, vocab_size: int = 5) -> None:
        super().__init__()
        self.frontend = ConvFrontend(input_channels, hidden_dim, downsample_factor)
        cycle = [1, 2, 4, 8, 16, 32]
        self.blocks = nn.ModuleList([ResidualConvBlock(hidden_dim, conv_kernel, cycle[idx % len(cycle)], dropout) for idx in range(num_layers)])
        self.final_norm = group_norm(hidden_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features, out_lengths = self.frontend(x, lengths)
        y = features.transpose(1, 2)
        for block in self.blocks:
            y = block(y)
        y = F.silu(self.final_norm(y)).transpose(1, 2)
        return y, out_lengths

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        features, out_lengths = self.encode(x, lengths)
        logits = self.classifier(self.dropout(features))
        log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1).contiguous()
        return log_probs, out_lengths.clamp(max=log_probs.shape[0]), {"model": "tcn"}


class TCNStudentWithMTA(TCNStudentCTC):
    def __init__(self, *args: Any, ms_kernels: str = "5,9,17,33", alpha_ms: float = 0.05, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        hidden_dim = self.classifier.in_features
        self.mta = MultiScaleTemporalAdapter(hidden_dim=hidden_dim, kernels=ms_kernels)
        self.alpha_ms = float(alpha_ms)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        features, out_lengths = self.encode(x, lengths)
        mask = torch.arange(features.shape[1], device=features.device).unsqueeze(0) >= out_lengths.to(features.device).unsqueeze(1)
        context, stats = self.mta(features, mask)
        features = features + self.alpha_ms * context
        logits = self.classifier(self.dropout(features))
        log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1).contiguous()
        return log_probs, out_lengths.clamp(max=log_probs.shape[0]), {"model": "tcn_mta", **stats}


class GCRTConformerCTC(nn.Module):
    """Conformer-style CTC student with a convolutional frontend and optional MTA."""

    def __init__(self, input_channels: int = 1, hidden_dim: int = 384, num_layers: int = 4, num_heads: int = 4, ff_dim: int = 1536, downsample_factor: int = 4, dropout: float = 0.1, vocab_size: int = 5, use_mta: bool = True, alpha_ms: float = 0.02) -> None:
        super().__init__()
        self.frontend = ConvFrontend(input_channels, hidden_dim, downsample_factor)
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.use_mta = bool(use_mta)
        self.mta = MultiScaleTemporalAdapter(hidden_dim=hidden_dim, kernels="3,7,15,31") if self.use_mta else None
        self.alpha_ms = float(alpha_ms)
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        features, out_lengths = self.frontend(x, lengths)
        mask = torch.arange(features.shape[1], device=features.device).unsqueeze(0) >= out_lengths.to(features.device).unsqueeze(1)
        features = self.encoder(features, src_key_padding_mask=mask)
        stats: dict[str, Any] = {"model": "gcrt_conformer"}
        if self.mta is not None:
            context, mta_stats = self.mta(features, mask)
            features = features + self.alpha_ms * context
            stats.update(mta_stats)
        logits = self.classifier(self.dropout(features))
        log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1).contiguous()
        return log_probs, out_lengths.clamp(max=log_probs.shape[0]), stats


def build_model(config: dict[str, Any]) -> nn.Module:
    model_type = str(config.get("model", "tcn_mta")).lower()
    params = dict(config.get("model_params", {}))
    if model_type == "lstm":
        return LSTMStudentCTC(**params)
    if model_type == "tcn":
        return TCNStudentCTC(**params)
    if model_type in {"tcn_mta", "tcn+ mta", "tcn_mta"}:
        return TCNStudentWithMTA(**params)
    if model_type in {"gcrt", "conformer", "gcrt_conformer"}:
        return GCRTConformerCTC(**params)
    raise ValueError(f"unknown model type: {model_type}")

