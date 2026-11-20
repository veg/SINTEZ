# SINTEZ-SIGNALS/1.0 Contract Specification

This directory contains deep-learning structural gene signals exported by **SINTEZ** (`sintez export-signals`) for downstream integrative decoding (e.g., **KOSHKA** signal `S5`) and standalone downstream analysis.

## Directory Layout

- `docs/SIGNALS.md`: Contract marker and format specification (must exist for `S5` detection).
- `manifest.json`: Top-level JSON metadata (`schema: "SINTEZ-SIGNALS/1.0"`, `contigs: {seqid: {"length": int, "track_file": "tracks/<safe_seqid>.npz"}}`).
  - `seqid` strictly follows `header[1:].split()[0]` from the input FASTA.
- `tracks/<safe_seqid>.npz`: Per-contig compressed NumPy archive of `float16` arrays of shape `(L,)` for both strands (`fwd_*` for `+`, `rev_*` for `-`).
- `exon_scores.tsv`: Sparse candidate exon log-likelihood ratios (`seqid\tstrand\tstart0\tend\tllr[\ttype]`).
- `intron_scores.tsv`: Sparse candidate intron log-likelihood ratios (`seqid\tstrand\tstart0\tend\tllr`).
- `genes_sintez.gff3`: Decoded SINTEZ v3.2 gene models with `posterior` and `tier` attributes (`high` >= 0.85, `medium` >= 0.55, `low` < 0.55), when decoder execution is enabled.

## Coordinate & Orientation Conventions

1. **Genomic TSV Coordinates (`exon_scores.tsv`, `intron_scores.tsv`)**:
   - All coordinates `[start0, end)` are **0-based, half-open** on the forward genomic contig `[0, L)`.
   - An oriented `-` strand exon or intron `(a_rc, b_rc)` on the reverse-complement sequence of length `L` maps to genomic coordinates `[L - b_rc, L - a_rc)`.
   - Lines beginning with `#` are comment/header lines and ignored by readers.

2. **Dense `float16` Track Orientation (`tracks/<safe_seqid>.npz`)**:
   - All `fwd_*` arrays have length `L` in genomic 5'->3' (`+`) coordinates (`0 .. L-1`).
   - All `rev_*` arrays have length `L` in reverse-complement 5'->3' (`-`) coordinates (`0 .. L-1`), where index `i` on `rev_*` corresponds to genomic base `L - 1 - i`.
   - For a `-` strand chunk covering genomic interval `[offset, offset + n)`, the oriented chunk slice is `[L - (offset + n) : L - offset]`.

3. **Exact 1-bp Boundary Indexing (Oriented 5'->3' Coordinates `[a, b)`)**:
   - `start[a]`: Start codon initiation signal aligned to the `A` of `ATG` (first coding base of CDS `[a, b)`).
   - `stop[b - 3]`: Stop codon termination signal aligned to the `T` of `TAA`/`TAG`/`TGA` (first base of the terminal stop codon inside `[a, b)`).
   - `donor[b]`: 5' splice donor signal aligned to the first intronic base (`G` of `GT`/`GC`, `A` of `AT`) for an exon ending at `b`.
   - `acceptor[a - 1]`: 3' splice acceptor signal aligned to the last intronic base (`G` of `AG`, `C` of `AC`) immediately preceding an exon starting at `a`.
   - `coding`, `intron`, `intergenic`: Per-base 3-state posterior probabilities (`span_probs[:, 2/1/0, :]`).
   - `foldability`: Per-base neural coding quality / structural foldability probability.
