# Reproducibility Notes

The paper evaluation is teacher-relative. The fixed Dorado SUP teacher provides pseudo-labels and reference-alignment confidence values. Student models are trained and compared under a fixed split, fixed greedy CTC decoder, and fixed post-decoding alignment protocol.

Key protocol points:

- The main method is QW-CTC plus MTA.
- QW-CTC uses fixed read weights derived offline from identity-derived teacher-to-reference confidence \(q_i\).
- MTA is a representation-side temporal adapter before the CTC head.
- BP and LenReg are calibration probes rather than primary architecture components.
- Del, CER, Len, Ins, and Mis are computed after decoding by aligning the student sequence to the fixed teacher pseudo-label.
- The fixed student split is 3,276 training reads, 409 validation reads, and 411 paired test reads.
- The 402-read public-baseline subset is used only for auxiliary baseline comparison.

Raw data, pseudo-labels, precomputed CTC chunks, read-weight tables, decoded FASTQ files, and completed metric CSV files must be obtained or generated externally before running these scripts.

The `paper_*.yaml` files are paper-style templates. They record the intended optimizer and selection protocol, but exact end-to-end reproduction also requires the external preprocessing and completed evaluation artifacts that are not distributed in this code-only release.
