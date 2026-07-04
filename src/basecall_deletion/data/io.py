from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


BASE_TO_ID = {"A": 1, "C": 2, "G": 3, "T": 4, "U": 4}
ID_TO_BASE = {0: "_", 1: "A", 2: "C", 3: "G", 4: "T"}


def open_text(path: str | Path):
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, "rt", encoding="utf-8", errors="replace")
    return p.open("r", encoding="utf-8", errors="replace")


def read_fastq(path: str | Path) -> dict[str, str]:
    records: dict[str, str] = {}
    with open_text(path) as handle:
        while True:
            header = handle.readline()
            if not header:
                break
            seq = handle.readline()
            plus = handle.readline()
            qual = handle.readline()
            if not qual:
                break
            del plus
            read_id = header[1:].strip().split()[0] if header.startswith(">") or header.startswith(chr(64)) else header.strip().split()[0]
            records[read_id] = "".join(base for base in seq.strip().upper().replace("U", "T") if base in "ACGT")
    return records


def write_fastq(path: str | Path, records: dict[str, str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for read_id, seq in records.items():
            handle.write(f"{chr(64)}{read_id}\n{seq}\n+\n{'I' * len(seq)}\n")


def encode_sequence(seq: str) -> list[int]:
    return [BASE_TO_ID[base] for base in seq.upper().replace("U", "T") if base in BASE_TO_ID]


def decode_ids(ids: Iterable[int], blank_idx: int = 0) -> str:
    return "".join(ID_TO_BASE.get(int(item), "") for item in ids if int(item) != blank_idx)


def read_weight_table(path: str | Path | None) -> dict[str, float]:
    if not path:
        return {}
    weights: dict[str, float] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            read_id = str(row.get("read_id", "")).strip()
            if not read_id:
                continue
            try:
                weights[read_id] = float(row.get("weight", "1"))
            except ValueError:
                weights[read_id] = 1.0
    return weights


class ChunkedCTCDataset(Dataset):
    """Dataset for pre-materialized CTC chunks.

    Expected files are `chunks.npy`, `references.npy`, and `reference_lengths.npy`.
    Optional `chunk_meta.jsonl` rows may provide read identifiers.
    """

    def __init__(self, ctc_data_dir: str | Path, weights_csv: str | Path | None = None) -> None:
        self.root = Path(ctc_data_dir)
        self.chunks = np.load(self.root / "chunks.npy", mmap_mode="r")
        self.references = np.load(self.root / "references.npy", mmap_mode="r")
        self.reference_lengths = np.load(self.root / "reference_lengths.npy", mmap_mode="r")
        if self.chunks.shape[0] != self.references.shape[0]:
            raise ValueError("chunks.npy and references.npy have different first dimensions")
        self.meta = self._load_meta()
        self.weights = read_weight_table(weights_csv)

    def _load_meta(self) -> list[dict[str, object]]:
        path = self.root / "chunk_meta.jsonl"
        if not path.exists():
            return []
        rows: list[dict[str, object]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def __len__(self) -> int:
        return int(self.reference_lengths.shape[0])

    def __getitem__(self, index: int):
        signal = np.asarray(self.chunks[index], dtype=np.float32).reshape(-1)
        label_len = int(self.reference_lengths[index])
        target = np.asarray(self.references[index, :label_len], dtype=np.int64)
        read_id = f"chunk_{index}"
        if index < len(self.meta):
            read_id = str(self.meta[index].get("read_id", read_id))
        weight = float(self.weights.get(read_id, 1.0))
        return torch.from_numpy(signal[None, :]), torch.from_numpy(target), int(signal.shape[0]), label_len, read_id, weight


def collate_ctc_batch(batch):
    max_len = max(int(item[2]) for item in batch)
    batch_size = len(batch)
    channels = int(batch[0][0].shape[0])
    signals = torch.zeros((batch_size, channels, max_len), dtype=torch.float32)
    labels = []
    signal_lens = []
    label_lens = []
    read_ids = []
    weights = []
    for idx, (signal, label, signal_len, label_len, read_id, weight) in enumerate(batch):
        signals[idx, :, :signal_len] = signal
        labels.append(label)
        signal_lens.append(signal_len)
        label_lens.append(label_len)
        read_ids.append(read_id)
        weights.append(weight)
    return (
        signals,
        torch.cat(labels),
        torch.tensor(signal_lens, dtype=torch.long),
        torch.tensor(label_lens, dtype=torch.long),
        read_ids,
        torch.tensor(weights, dtype=torch.float32),
    )


def make_dataloader(ctc_data_dir: str | Path, batch_size: int, weights_csv: str | Path | None = None, shuffle: bool = True) -> DataLoader:
    dataset = ChunkedCTCDataset(ctc_data_dir, weights_csv=weights_csv)
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=shuffle, collate_fn=collate_ctc_batch)
