# Excluded Files

The release excludes the following classes of files:

- raw sequencing data such as POD5, FAST5, FASTQ, BAM, SAM, CRAM, FASTA, GTF, and index files
- trained checkpoints and model artifacts
- completed run directories, large result archives, and generated figures
- private logs, scheduler outputs, environment files, and credential-like files
- paper PDFs, LaTeX source, and LaTeX build artifacts
- unrelated exploratory prototypes and old experiment wrappers not needed for the final paper claims

Some original project scripts were not copied because they contained hard-coded local paths, old experiment names, or result-collection logic tied to private run directories. Their reusable logic was represented here through clean modules and wrapper scripts using placeholder paths.

