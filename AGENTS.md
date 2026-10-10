# AGENTS.md — Autonomous Agent & Machine Contributor Guide

Welcome, machine agent. This document defines the engineering contracts, architectural invariants, coordinate conventions, and testing protocols governing the **СИНТЕЗ (SINTEZ)** repository.

---

## 1. Prime Invariants & Non-Negotiable Rules

When modifying, extending, or testing code in this repository, you **MUST** strictly adhere to the following core principles:

1. **Zero Synthetic Mockery & Authentic Computation:**
   - **Never** inject synthetic placeholders, hardcoded thresholds for specific taxa, parametric logistic curves, or mock evaluation metrics.
   - Every metric reported in documentation, benchmark artifacts, or papers **must** be the authentic computational output of code executed on verifiable NCBI RefSeq data.
2. **Taxon-Agnostic Design (Zero Test-Set Tuning):**
   - **Never** implement hardcoded conditionals for specific organisms (e.g., `if organism == "plasmodium": threshold = 0.8`).
   - All improvements to the model must be grounded in **general physical or information-theoretic principles** (e.g., Jacobson-Stockmayer polymer looping, Shannon sequence entropy, local composition pooling).
3. **Dual-Space Architecture Separation:**
   - Keep DNA-space features (nucleotide sequence, splice boundaries, compositional normalization) cleanly separated from Protein-space features (ESM-2 embeddings, amino acid sequences, foldability energies).
   - The protein oracle (`ProteinEnergyOracle`) must remain **frozen** and strictly evaluated on translated products produced by `SplicedTranslationOperator`.

---

## 2. Coordinate System & Boundary Invariants

All modules inside `sintez/` must follow strict coordinate conventions:

- **Internal Python Representations:**
  - **Always 0-based, half-open intervals:** `[start, end)`.
  - Length is always `end - start`.
  - Exon 1 ends at `d`; intron runs from `d` to `a`; Exon 2 starts at `a`.
  - Spliced transcript sequence is formed by concatenating `seq[s:e]` for each `(s, e)` span.
- **External GFF3 Serializer (`sintez.cli.write_gff3`):**
  - **Always 1-based, inclusive intervals:** `[start + 1, end]`.
  - The conversion happens **only** at serialization time. Never use 1-based coordinates inside DAG decoding or dynamic programming algorithms.
- **Strand Conventions:**
  - Forward strand (`+`): Coordinates reference the 5' $\to$ 3' forward genomic sequence.
  - Reverse strand (`-`): Candidate coordinates decoded on the reverse-complement sequence of length $`L`$ must be projected back to genomic forward coordinates using:
    `genomic_spans = [(L - e, L - s) for s, e in reversed(spans)]`
- **Splice Phase Tracking:**
  - Donor phase: `d_phase = (last_phase + (d - last_end)) % 3`.
  - Offset to the first complete in-frame codon after acceptor `a`: `offset = (3 - d_phase) % 3`.
  - The split codon is formed by joining the trailing `d_phase` nucleotides of the donor exon with the leading `offset` nucleotides of the acceptor exon.

---

## 3. Directory Layout & Module Responsibilities

```
SINTEZ/
├── sintez/
│   ├── __init__.py               # Package metadata and public exports
│   ├── dna_backbone.py           # CompositionNormalizedBackbone (CNN + BiGRU, 238k params)
│   ├── protein_oracle.py         # ProteinEnergyOracle (Frozen ESM-2 + Shannon entropy)
│   ├── translation_operator.py   # SplicedTranslationOperator & SplicedTranscript
│   ├── dag_decoder.py            # TopologicalDAGDecoder (Pareto NMS + Polymer prior)
│   ├── dual_strand.py            # DualStrandResolver (Reciprocal overlap arbitration)
│   ├── trainer.py                # CounterfactualContrastiveTrainer engine
│   └── cli.py                    # Command-line interface (`sintez predict`)
├── models/
│   └── sintez_backbone_multiphyla.pt  # Pretrained backbone weights (953 KB)
├── tests/
│   ├── test_backbone.py          # Tensor shape, parameter count, and normalization tests
│   ├── test_oracle.py            # Shannon entropy, PTC penalties, and ESM energy tests
│   ├── test_translation.py       # Single-exon, split-codon, and reverse strand tests
│   ├── test_decoder.py           # DAG search, polymer looping, and Pareto logic tests
│   ├── test_dual_strand.py       # Antisense conflict arbitration tests
│   └── test_cli.py               # Argument parsing and GFF3 formatting tests
├── pyproject.toml                # Build configuration and dependencies
├── requirements.txt              # Pip dependencies
├── README.md                     # Human-facing documentation & benchmark tables
├── AGENTS.md                     # Machine contributor guide (this file)
├── ROADMAP.md                    # Future architectural development plan
└── LICENSE                       # MIT License
```

---

## 4. Verification Protocol

Before submitting or committing any change, execute the test suite:

```bash
# 1. Run all unit tests
pytest tests/ -v

# 2. Check CLI entrypoint
python -m sintez.cli --help

# 3. Parameter count assertion
# The backbone parameter count must remain exactly 238,407 unless an architectural
# upgrade is explicitly documented and tested.
```

---

## 5. Benchmarking Against Authentic Ground Truth

To benchmark against an NCBI RefSeq chromosome:
1. Obtain the authentic genomic FASTA and NCBI Feature Table (`.ft`).
2. Run windowed inference and decoding:
   ```bash
   sintez predict --fasta /path/to/chrom.fa --out predictions.gff3 --chrom_id NC_XXXXXX.X
   ```
3. Extract CDS intervals from `.ft` (0-based half-open) and calculate:
   - **Base-level Confusion Matrix:** $`TP, FP, FN, TN`$.
   - **Base MCC:**

```math
\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
```

   - **Exact Gene Matches:** 100% identity of all internal exon-intron boundaries and phase.
   - **Recovered Loci:** $\ge 50\%$ reciprocal overlap with ground truth.
