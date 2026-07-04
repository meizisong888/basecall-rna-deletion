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

