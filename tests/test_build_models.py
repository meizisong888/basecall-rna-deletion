from __future__ import annotations

import torch

from basecall_deletion.models import build_model


def smoke_forward(model_name: str, params: dict[str, object]) -> None:
    model = build_model({"model": model_name, "model_params": params})
    x = torch.randn(2, 64)
    log_probs, out_lens, aux = model(x, torch.tensor([64, 60]))
    assert log_probs.ndim == 3
    assert out_lens.shape == (2,)
    assert isinstance(aux, dict)


def test_build_documented_model_types() -> None:
    smoke_forward("lstm", {"hidden_dim": 16, "lstm_hidden": 16, "num_layers": 1})
    smoke_forward("lstm_mta", {"hidden_dim": 16, "lstm_hidden": 16, "num_layers": 1})
    smoke_forward("tcn", {"hidden_dim": 16, "num_layers": 1, "conv_kernel": 3})
    smoke_forward("tcn_mta", {"hidden_dim": 16, "num_layers": 1, "conv_kernel": 3})
    smoke_forward("gcrt_conformer", {"hidden_dim": 16, "num_layers": 1, "num_heads": 4, "ff_dim": 32})

