# Reducing Teacher-Relative Deletion Errors in Direct RNA Nanopore Student Basecalling

This repository contains cleaned implementation components for deletion-aware direct RNA nanopore student basecalling experiments, including QW-CTC, the Multi-scale Temporal Adapter, teacher-relative Del/CER evaluation, paired statistical testing, and table/figure utilities.

## Double-blind note

This repository is prepared for de-anonymized release. The anonymous submission should not include a personal GitHub link.

## What is included

- QW-CTC training code
- Multi-scale Temporal Adapter code
- LSTM, TCN, and GCRT/Conformer-style student model components, including QW-CTC+MTA instantiation paths
- teacher-relative evaluation scripts
- paired bootstrap and sign-flip analysis
- table and figure helper scripts for externally supplied metric CSV files
- smoke-test and paper-style config templates

## What is not included

- raw GTGSEQ RNA004 data
- Dorado pseudo-label FASTQ files
- precomputed CTC chunks and read-weight tables
- BAM/SAM alignment files
- reference transcriptome files
- trained checkpoints
- large experiment outputs
- private cluster logs
- paper PDF or LaTeX source

This repository does not include completed paper result archives. The provided scripts are implementation components and wrappers; they do not regenerate Tables I--IV without the external data, pseudo-labels, CTC chunks, read-weight tables, decoded outputs, and metric CSV files used by the paper.

## Expected data layout

External data should be prepared outside this repository using placeholder paths such as:

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

The training wrapper expects pre-materialized CTC chunks. The expected target alphabet is blank plus A/C/G/T, with U converted to T for RNA teacher pseudo-label evaluation.

## Minimal installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

Core dependencies include PyTorch, numpy, pandas, scipy, matplotlib, tqdm, edlib, biopython, pyyaml, pod5, and h5py. The `pod5` and `h5py` packages are only needed when external preprocessing code reads raw signal files.

## Example commands

```bash
python scripts/prepare_confidence_weights.py \
  --metadata-csv <DATA_ROOT>/read_weights/reference_alignment_confidence.csv \
  --output-csv <READ_WEIGHT_TABLE>
```

```bash
python scripts/train_qw_ctc.py \
  --config configs/example_smoke_qw_ctc_mta.yaml \
  --data-root <DATA_ROOT> \
  --pseudo-label-fastq <PSEUDOLABEL_FASTQ> \
  --read-weight-table <READ_WEIGHT_TABLE> \
  --output-dir <OUTPUT_ROOT>/train_qw_ctc_mta
```

```bash
python scripts/evaluate_teacher_relative_errors.py \
  --pred-fastq <PRED_FASTQ> \
  --teacher-fastq <PSEUDOLABEL_FASTQ> \
  --output-dir <OUTPUT_ROOT>/eval
```

```bash
python scripts/run_paired_bootstrap.py \
  --control-per-read <OUTPUT_ROOT>/control/per_read_teacher_relative_metrics.csv \
  --method-per-read <OUTPUT_ROOT>/method/per_read_teacher_relative_metrics.csv \
  --output-csv <OUTPUT_ROOT>/bootstrap/paired_bootstrap.csv
```

## Reproducibility notes

- Del and CER are teacher-relative to fixed Dorado SUP pseudo-labels.
- Evaluation uses greedy CTC decoding and post-decoding alignment.
- QW-CTC read weights are fixed offline from identity-derived teacher-to-reference confidence \(q_i\).
- The paper uses a fixed 3,276 / 409 / 411 student split.
- The 402-read public-baseline subset is only for auxiliary baseline comparison.
- This code release does not include external data, pseudo-labels, checkpoints, or completed result archives.

## Citation

Citation will be added after publication.
