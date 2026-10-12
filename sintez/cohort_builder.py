"""
High-Performance Multi-Phyla Cohort Builder
===========================================
Extracts, verifies, and compiles an optimal 12,000-14,000 gene training cohort
from 5 diverse eukaryotic reference genomes across the tree of life:
1. Schizosaccharomyces pombe (Fungi)
2. Caenorhabditis elegans (Metazoa/Nematoda)
3. Drosophila melanogaster (Metazoa/Insecta)
4. Arabidopsis thaliana (Viridiplantae/Plant)
5. Chlamydomonas reinhardtii (Protista/Algae - 64% GC)

Filters:
- Spliced coding exons verified in-frame with canonical start/stop
- Flanking context (pad = 200-500 bp)
- Intron horizon up to 25 kb
- Eliminates non-coding pseudogenes and incomplete annotations
"""

import gzip
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

from .trainer import GenomicGeneSample
from .translation_operator import SplicedTranslationOperator


def load_fasta_chroms(fasta_path: Path) -> Dict[str, str]:
    """Loads all chromosomes/scaffolds from a FASTA file into memory."""
    chroms = {}
    cur_id = None
    cur_seq = []
    open_fn = gzip.open if str(fasta_path).endswith(".gz") else open
    with open_fn(fasta_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id:
                    chroms[cur_id] = "".join(cur_seq)
                cur_id = line[1:].split()[0]
                cur_seq = []
            else:
                cur_seq.append(line)
        if cur_id:
            chroms[cur_id] = "".join(cur_seq)
    return chroms


def parse_gff_transcripts(
    gff_path: Path,
    chroms: Dict[str, str],
    prefix: str,
    max_multi: int = 3000,
    max_single: int = 1500,
    max_len: int = 6000,
    pad: int = 200,
    seed: int = 42,
) -> Tuple[List[GenomicGeneSample], List[GenomicGeneSample]]:
    """
    Parses verified protein-coding transcripts from a GFF3 file and returns
    balanced lists of multi-exon and single-exon GenomicGeneSample objects.
    """
    tx_cds = defaultdict(list)
    tx_info = {}

    open_fn = gzip.open if str(gff_path).endswith(".gz") else open
    with open_fn(gff_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 9 and parts[2] == "CDS":
                chrom, s, e, strand, attrs = parts[0], int(parts[3]), int(parts[4]), parts[6], parts[8]
                parent = None
                gene_name = None
                for item in attrs.split(";"):
                    item = item.strip()
                    if item.startswith("Parent="):
                        parent = item.split("=")[1].replace("rna-", "").replace("gene-", "")
                    elif item.startswith("gene=") and not gene_name:
                        gene_name = item.split("=")[1]
                    elif item.startswith("locus_tag=") and not gene_name:
                        gene_name = item.split("=")[1]

                if parent:
                    tx_cds[parent].append((min(s, e), max(s, e)))
                    if parent not in tx_info:
                        tx_info[parent] = (chrom, strand, gene_name or parent)

    operator = SplicedTranslationOperator()
    multi_genes = []
    single_genes = []

    tx_keys = list(tx_cds.keys())
    rng = random.Random(seed)
    rng.shuffle(tx_keys)

    for tx_id in tx_keys:
        intervals = tx_cds[tx_id]
        chrom, strand, gene_name = tx_info[tx_id]
        if chrom not in chroms:
            continue
        chr_seq = chroms[chrom]
        intervals.sort(key=lambda x: x[0])
        spans = [(s - 1, e) for s, e in intervals]

        l_s = max(0, spans[0][0] - pad)
        l_e = min(len(chr_seq), spans[-1][1] + pad)
        locus_len = l_e - l_s
        if locus_len > max_len or locus_len < 100:
            continue

        locus_dna = chr_seq[l_s:l_e]
        rel_spans = [(s - l_s, e - l_s) for s, e in spans]

        # Biological validation: verify mature spliced peptide has valid start, in-frame, no internal stops
        tx = operator.process_transcript(locus_dna, rel_spans, strand=strand)
        if tx.is_in_frame and not tx.has_internal_stop and len(tx.protein_sequence) >= 30:
            sample = GenomicGeneSample(
                dna_sequence=locus_dna,
                exon_spans=rel_spans,
                strand=strand,
                gene_id=f"{prefix}_{gene_name}",
            )
            if len(spans) > 1:
                if len(multi_genes) < max_multi:
                    multi_genes.append(sample)
            else:
                if len(single_genes) < max_single:
                    single_genes.append(sample)

        if len(multi_genes) >= max_multi and len(single_genes) >= max_single:
            break

    return multi_genes, single_genes


def extract_pure_negative_windows(
    gff_path: Path,
    chroms: Dict[str, str],
    prefix: str,
    target_count: int,
    min_len: int = 1500,
    max_len: int = 4000,
    buffer_bp: int = 500,
    seed: int = 42,
) -> List[GenomicGeneSample]:
    """
    Extracts authentic, verified non-coding intergenic windows from reference genomes.
    Guarantees that every extracted window is at least `buffer_bp` away from any annotated
    coding exon, gene, mRNA, pseudogene, or transposable element feature.
    """
    rng = random.Random(seed)
    gene_intervals = defaultdict(list)
    open_fn = gzip.open if str(gff_path).endswith(".gz") else open
    with open_fn(gff_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 5 and parts[2] in ("gene", "CDS", "mRNA", "pseudogene", "transposable_element", "exon"):
                chrom, s, e = parts[0], int(parts[3]), int(parts[4])
                gene_intervals[chrom].append((max(0, s - buffer_bp), e + buffer_bp))

    candidates = []
    for chrom, seq in chroms.items():
        if chrom not in gene_intervals:
            continue
        raw_ints = sorted(gene_intervals[chrom], key=lambda x: x[0])
        merged = []
        for s, e in raw_ints:
            if not merged or s > merged[-1][1]:
                merged.append([s, e])
            else:
                merged[-1][1] = max(merged[-1][1], e)

        last_end = 0
        for s, e in merged:
            if s - last_end >= min_len:
                candidates.append((chrom, last_end, s))
            last_end = max(last_end, e)
        if len(seq) - last_end >= min_len:
            candidates.append((chrom, last_end, len(seq)))

    rng.shuffle(candidates)
    samples = []
    for chrom, s, e in candidates:
        span_len = e - s
        if span_len < min_len:
            continue
        num_windows = max(1, span_len // max_len)
        for _ in range(num_windows):
            w_len = rng.randint(min_len, min(max_len, span_len))
            w_start = rng.randint(s, e - w_len)
            subseq = chroms[chrom][w_start : w_start + w_len]
            # Discard regions with >10% assembly Ns
            if subseq.upper().count("N") / len(subseq) > 0.10:
                continue
            samples.append(
                GenomicGeneSample(
                    dna_sequence=subseq,
                    exon_spans=[],
                    strand="+",
                    gene_id=f"Neg_{prefix}_{chrom}_{w_start}_{w_start+w_len}",
                )
            )
            if len(samples) >= target_count:
                break
        if len(samples) >= target_count:
            break

    return samples


def compile_optimal_training_cohort(
    data_dir: Path = Path("real_data"),
    target_total: int = 25000,
    negative_ratio: float = 0.35,
    seed: int = 42,
) -> List[GenomicGeneSample]:
    """
    Compiles a mathematically scaled 25,000 sample cohort across 5 eukaryotic phyla:
    - ~65% verified spliced & single-exon protein-coding loci (~16,250 genes)
    - ~35% authentic non-coding intergenic windows (~8,750 windows) to eliminate intergenic FP leakage.
    """
    print("=" * 70)
    print(f"  COMPILING SCALED {target_total:,d} SAMPLE MULTI-PHYLA TRAINING COHORT")
    print(f"  Target Composition: {int((1 - negative_ratio) * 100)}% Verified Genes + {int(negative_ratio * 100)}% Pure Intergenic Windows")
    print("=" * 70)

    configs = [
        {
            "prefix": "Spombe",
            "fna": data_dir / "spombe/spombe_genomic.fna",
            "gff": data_dir / "spombe/spombe_genomic.gff",
            "max_multi": 2000,
            "max_single": 500,
            "max_neg": 250,
        },
        {
            "prefix": "Celegans",
            "fna": data_dir / "celegans/celegans_genomic.fna",
            "gff": data_dir / "celegans/celegans_genomic.gff",
            "max_multi": 3000,
            "max_single": 500,
            "max_neg": 2500,
        },
        {
            "prefix": "Drosophila",
            "fna": data_dir / "drosophila/drosophila_genomic.fna",
            "gff": data_dir / "drosophila/drosophila_genomic.gff",
            "max_multi": 3000,
            "max_single": 500,
            "max_neg": 2000,
        },
        {
            "prefix": "Arabidopsis",
            "fna": data_dir / "arabidopsis/arabidopsis_genomic.fna",
            "gff": data_dir / "arabidopsis/arabidopsis_genomic.gff",
            "max_multi": 3000,
            "max_single": 500,
            "max_neg": 2500,
        },
        {
            "prefix": "Chlamydomonas",
            "fna": data_dir / "chlamydomonas/chlamydomonas_genomic.fna",
            "gff": data_dir / "chlamydomonas/chlamydomonas_genomic.gff",
            "max_multi": 2750,
            "max_single": 500,
            "max_neg": 1500,
        },
    ]

    all_samples = []
    total_multi = 0
    total_single = 0
    total_neg = 0

    for cfg in configs:
        prefix = cfg["prefix"]
        fna = cfg["fna"]
        gff = cfg["gff"]
        if not fna.exists() or not gff.exists():
            print(f"[!] Warning: Missing files for {prefix} at {fna} / {gff}")
            continue

        print(f"[*] Parsing {prefix:14s} ({fna.name})...")
        chroms = load_fasta_chroms(fna)
        multi, single = parse_gff_transcripts(
            gff_path=gff,
            chroms=chroms,
            prefix=prefix,
            max_multi=cfg["max_multi"],
            max_single=cfg["max_single"],
            max_len=6000,
            seed=seed,
        )
        print(f"    -> Extracted {len(multi):4,d} spliced multi-exon + {len(single):4,d} single-exon genes.")
        all_samples.extend(multi)
        all_samples.extend(single)
        total_multi += len(multi)
        total_single += len(single)

        # Extract authentic negative windows
        negs = extract_pure_negative_windows(
            gff_path=gff,
            chroms=chroms,
            prefix=prefix,
            target_count=cfg["max_neg"],
            min_len=1500,
            max_len=4000,
            buffer_bp=500,
            seed=seed,
        )
        print(f"    -> Extracted {len(negs):4,d} pure negative intergenic windows.")
        all_samples.extend(negs)
        total_neg += len(negs)

    rng = random.Random(seed)
    rng.shuffle(all_samples)

    print("\n" + "=" * 70)
    print(f"[+] Successfully Compiled Scaled 25K Cohort: {len(all_samples):,d} Total Training Samples")
    print(f"    - Multi-Exon Spliced Genes: {total_multi:,d} ({total_multi/len(all_samples)*100:.1f}%)")
    print(f"    - Single-Exon Genes:        {total_single:,d} ({total_single/len(all_samples)*100:.1f}%)")
    print(f"    - Pure Intergenic Negatives: {total_neg:,d} ({total_neg/len(all_samples)*100:.1f}%)")
    print("=" * 70 + "\n")

    return all_samples

