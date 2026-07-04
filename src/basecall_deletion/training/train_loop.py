from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from basecall_deletion.data.io import make_dataloader
from basecall_deletion.models.students import build_model
from basecall_deletion.training.losses import training_loss


def train_from_config(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    data_cfg = dict(config.get("data", {}))
    train_loader = make_dataloader(data_cfg["train_ctc_dir"], int(config.get("batch_size", 4)), weights_csv=data_cfg.get("read_weight_table"), shuffle=True)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 1e-4)), weight_decay=float(config.get("weight_decay", 1e-4)))
    coefficients = dict(config.get("loss", {}))
    history = []
    epochs = int(config.get("epochs", 1))
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        steps = 0
        for signals, labels, signal_lens, label_lens, read_ids, weights in train_loader:
            del read_ids
            signals = signals.to(device)
            labels = labels.to(device)
            signal_lens = signal_lens.to(device)
            label_lens = label_lens.to(device)
            weights = weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            log_probs, output_lens, aux = model(signals, signal_lens)
            del aux
            loss, parts = training_loss(
                {
                    "log_probs": log_probs,
                    "labels": labels,
                    "input_lens": output_lens,
                    "label_lens": label_lens,
                    "weights": weights,
                },
                coefficients,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip", 5.0)))
            optimizer.step()
            total += float(parts["total"])
            steps += 1
        row = {"epoch": epoch, "train_loss": total / max(steps, 1), "steps": steps}
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        if bool(config.get("save_each_epoch", False)):
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch}, out / f"checkpoint_epoch{epoch}.pth")
    checkpoint = out / "checkpoint_last.pth"
    torch.save({"model": model.state_dict(), "config": config, "epoch": epochs}, checkpoint)
    (out / "train_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return {"checkpoint": str(checkpoint), "epochs": epochs, "history": history}

