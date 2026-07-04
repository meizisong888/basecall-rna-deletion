# Mapping Between Paper Components and Repository Code

| Paper component | Repository entry point |
| --- | --- |
| Table I, core QW-CTC plus MTA comparison | `scripts/train_qw_ctc.py`, `scripts/evaluate_teacher_relative_errors.py`, `scripts/make_paper_tables.py` with external completed metric CSV files |
| Table II, public-baseline evaluation | `scripts/evaluate_teacher_relative_errors.py` with public-baseline decoded FASTQ inputs |
| Table III, paired bootstrap and sign-flip testing | `scripts/run_paired_bootstrap.py` using read-level Del count/rate metrics |
| Table IV, confidence-source ablation | `scripts/prepare_confidence_weights.py` plus the same training and evaluation wrappers |
| Method MTA component | `src/basecall_deletion/models/mta.py` |
| Teacher-relative Del/CER/Len/Ins/Mis evaluation | `src/basecall_deletion/evaluation/teacher_relative.py` |
| Calibration probes BP and LenReg | `src/basecall_deletion/training/losses.py` and `configs/example_smoke_calibration_probe.yaml` |
| Paper summary figures | `scripts/make_paper_figures.py` |

The repository contains code paths and templates, not completed experiment outputs. It therefore supports re-running the workflow when the external data and intermediate artifacts are supplied; it does not by itself regenerate the paper tables.
