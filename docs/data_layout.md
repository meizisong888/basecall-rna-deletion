# Data Layout

This repository is code-only. Raw signal files, pseudo-labels, references, read-weight tables, and train/validation/test splits must be prepared externally.

Expected external layout:

```text
<DATA_ROOT>/
  signals/
  pseudo_labels/
  reference/
  read_weights/
  splits/
  ctc_chunks/
    train/
      chunks.npy
      references.npy
      reference_lengths.npy
      chunk_meta.jsonl
```

The `chunks.npy` file stores signal chunks as float arrays. The `references.npy` and `reference_lengths.npy` files store CTC targets derived from fixed Dorado teacher pseudo-labels. The optional `chunk_meta.jsonl` file may include `read_id` values used to join fixed read weights.

No sequencing data, alignment files, or reference files are included in this release.

The QW-CTC read-weight table is expected to contain at least `read_id`, `identity`, and `weight` columns. The `identity` column is the offline teacher-to-reference confidence \(q_i\), and the default mapping is 1.0 for \(q_i \geq 0.97\), 0.8 for \(0.95 \leq q_i < 0.97\), 0.5 for \(0.90 \leq q_i < 0.95\), and 0.2 otherwise.
