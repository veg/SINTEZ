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
    max_len: int = 10000,
    pad: int = 200,
    random_pad_range: Optional[Tuple[int, int]] = (400, 1800),
    seed: int = 42,
) -> Tuple[List[GenomicGeneSample], List[GenomicGeneSample]]:
    """
    Parses verified protein-coding transcripts from a GFF3 file and returns
    balanced lists of multi-exon and single-exon GenomicGeneSample objects.

    Key Training Improvements:
    1. Random asymmetric flanking context (`random_pad_range=(400, 1800)` bp) eliminates
       the fixed `pad=200` start/stop positional shortcut and ensures the 2,001 bp
       composition normalization window sees authentic genomic context.
    2. Labels ALL co-directional verified coding genes (`multi_gene_spans`) that fall inside
       each extracted genomic window so neighboring genes in gene-dense genomes are not
       falsely penalized as intergenic background.
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
    rng = random.Random(seed)

    # First pass: index all biologically valid transcripts per (chrom, strand) for multi-gene window labeling
    valid_by_chrom_strand: Dict[Tuple[str, str], List[Tuple[int, int, str, List[Tuple[int, int]]]]] = defaultdict(list)
    for tx_id, intervals in tx_cds.items():
        chrom, strand, gene_name = tx_info[tx_id]
        if chrom not in chroms:
            continue
        chr_seq = chroms[chrom]
        intervals.sort(key=lambda x: x[0])
        spans = [(s - 1, e) for s, e in intervals]
        g_s, g_e = spans[0][0], spans[-1][1]
        if g_e - g_s > max_len or g_e - g_s < 90 or g_s < 0 or g_e > len(chr_seq):
            continue
        tx = operator.process_transcript(chr_seq[g_s:g_e], [(s - g_s, e - g_s) for s, e in spans], strand=strand)
        if tx.is_in_frame and not tx.has_internal_stop and len(tx.protein_sequence) >= 30:
            valid_by_chrom_strand[(chrom, strand)].append((g_s, g_e, gene_name, spans))

    # Deduplicate overlapping isoforms per locus (keep longest coding isoform per locus)
    dedup_by_chrom_strand: Dict[Tuple[str, str], List[Tuple[int, int, str, List[Tuple[int, int]]]]] = defaultdict(list)
    for key, items in valid_by_chrom_strand.items():
        items.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        kept = []
        for g_s, g_e, g_name, spans in items:
            if kept and max(0, min(g_e, kept[-1][1]) - max(g_s, kept[-1][0])) > 0.30 * min(g_e - g_s, kept[-1][1] - kept[-1][0]):
                continue
            kept.append((g_s, g_e, g_name, spans))
        dedup_by_chrom_strand[key] = kept

    all_loci = []
    for (chrom, strand), items in dedup_by_chrom_strand.items():
        for g_s, g_e, g_name, spans in items:
            all_loci.append((chrom, strand, g_s, g_e, g_name, spans))
    rng.shuffle(all_loci)

    multi_genes = []
    single_genes = []

    for chrom, strand, g_s, g_e, gene_name, spans in all_loci:
        is_multi = len(spans) > 1
        if is_multi and len(multi_genes) >= max_multi:
            continue
        if not is_multi and len(single_genes) >= max_single:
            continue

        chr_seq = chroms[chrom]
        if random_pad_range is not None:
            pad_left = rng.randint(random_pad_range[0], random_pad_range[1])
            pad_right = rng.randint(random_pad_range[0], random_pad_range[1])
        else:
            pad_left = pad_right = pad

        l_s = max(0, g_s - pad_left)
        l_e = min(len(chr_seq), g_e + pad_right)
        locus_len = l_e - l_s
        if locus_len > max_len or locus_len < 150:
            continue

        locus_dna = chr_seq[l_s:l_e]
        rel_spans = [(s - l_s, e - l_s) for s, e in spans]

        # Collect all co-directional valid genes completely contained inside [l_s, l_e]
        co_genes_spans = []
        for o_s, o_e, _, o_spans in dedup_by_chrom_strand[(chrom, strand)]:
            if o_s >= l_s and o_e <= l_e:
                co_genes_spans.append([(s - l_s, e - l_s) for s, e in o_spans])
        if not co_genes_spans:
            co_genes_spans = [rel_spans]

        sample = GenomicGeneSample(
            dna_sequence=locus_dna,
            exon_spans=rel_spans,
            strand=strand,
            gene_id=f"{prefix}_{gene_name}",
            multi_gene_spans=co_genes_spans,
        )
        if is_multi:
            multi_genes.append(sample)
        else:
            single_genes.append(sample)

        if len(multi_genes) >= max_multi and len(single_genes) >= max_single:
            break

    return multi_genes, single_genes


def extract_antisense_negative_windows(
    gff_path: Path,
    chroms: Dict[str, str],
    prefix: str,
    target_count: int,
    pad_range: Tuple[int, int] = (300, 1200),
    max_len: int = 8000,
    seed: int = 42,
) -> List[GenomicGeneSample]:
    """
    Extracts hard-negative antisense windows: genomic windows containing a verified protein-coding
    gene on the OPPOSITE strand and zero coding exons on the queried forward strand.
    Training on these windows directly suppresses reverse-complement 'shadow ORF' false positives
    (driven by 3-base wobble periodicity leaking across strands).
    """
    rng = random.Random(seed + 101)
    strand_intervals = {"+": defaultdict(list), "-": defaultdict(list)}
    open_fn = gzip.open if str(gff_path).endswith(".gz") else open
    with open_fn(gff_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 7 and parts[2] in ("gene", "CDS", "mRNA", "exon"):
                chrom, s, e, strand = parts[0], int(parts[3]) - 1, int(parts[4]), parts[6]
                if strand in ("+", "-"):
                    strand_intervals[strand][chrom].append((s, e))

    def merge_iv(iv_list: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not iv_list:
            return []
        s_iv = sorted(iv_list, key=lambda x: x[0])
        m = [[s_iv[0][0], s_iv[0][1]]]
        for s, e in s_iv[1:]:
            if s <= m[-1][1]:
                m[-1][1] = max(m[-1][1], e)
            else:
                m.append([s, e])
        return [(x[0], x[1]) for x in m]

    merged_plus = {c: merge_iv(ivs) for c, ivs in strand_intervals["+"].items()}
    merged_minus = {c: merge_iv(ivs) for c, ivs in strand_intervals["-"].items()}

    operator = SplicedTranslationOperator()
    candidates = []
    # Collect '-' strand genes that have zero overlap with any '+' strand gene -> query on '+' strand
    for chrom, ivs in merged_minus.items():
        if chrom not in chroms:
            continue
        p_ivs = merged_plus.get(chrom, [])
        for s, e in ivs:
            if 500 <= (e - s) <= max_len - 1000:
                has_plus = any(min(e + 400, pe) > max(s - 400, ps) for ps, pe in p_ivs)
                if not has_plus:
                    candidates.append((chrom, s, e, "+"))

    # Collect '+' strand genes that have zero overlap with any '-' strand gene -> query on '-' strand (reverse complement)
    for chrom, ivs in merged_plus.items():
        if chrom not in chroms:
            continue
        m_ivs = merged_minus.get(chrom, [])
        for s, e in ivs:
            if 500 <= (e - s) <= max_len - 1000:
                has_minus = any(min(e + 400, me) > max(s - 400, ms) for ms, me in m_ivs)
                if not has_minus:
                    candidates.append((chrom, s, e, "-"))

    rng.shuffle(candidates)
    samples = []
    for chrom, s, e, query_strand in candidates:
        seq = chroms[chrom]
        w_s = max(0, s - rng.randint(pad_range[0], pad_range[1]))
        w_e = min(len(seq), e + rng.randint(pad_range[0], pad_range[1]))
        if w_e - w_s > max_len or w_e - w_s < 800:
            continue
        subseq = seq[w_s:w_e]
        if subseq.upper().count("N") / len(subseq) > 0.05:
            continue
        if query_strand == "-":
            subseq = operator.code.reverse_complement(subseq)
        samples.append(
            GenomicGeneSample(
                dna_sequence=subseq,
                exon_spans=[],
                strand="+",
                gene_id=f"AntisenseNeg_{prefix}_{chrom}_{w_s}_{w_e}",
                multi_gene_spans=[],
            )
        )
        if len(samples) >= target_count:
            break

    return samples


def extract_pure_negative_windows(
    gff_path: Path,
    chroms: Dict[str, str],
    prefix: str,
    target_count: int,
    min_len: int = 1500,
    max_len: int = 5000,
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
                    multi_gene_spans=[],
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
    - ~65% verified spliced & single-exon protein-coding windows (with wide random flanking context & multi-gene labeling)
    - ~20% pure non-coding intergenic windows
    - ~15% antisense hard-negative windows (real gene on opposite strand, zero coding bases on forward strand)
    """
    print("=" * 70)
    print(f"  COMPILING SCALED {target_total:,d} SAMPLE MULTI-PHYLA TRAINING COHORT")
    print(f"  Target Composition: {int((1 - negative_ratio) * 100)}% Multi-Gene Context Windows + {int(negative_ratio * 100)}% Intergenic & Antisense Negatives")
    print("=" * 70)

    configs = [
        {
            "prefix": "Spombe",
            "fna": data_dir / "spombe/spombe_genomic.fna",
            "gff": data_dir / "spombe/spombe_genomic.gff",
            "max_multi": 2000,
            "max_single": 500,
            "max_neg": 150,
            "max_antisense": 150,
        },
        {
            "prefix": "Celegans",
            "fna": data_dir / "celegans/celegans_genomic.fna",
            "gff": data_dir / "celegans/celegans_genomic.gff",
            "max_multi": 3000,
            "max_single": 500,
            "max_neg": 1500,
            "max_antisense": 1000,
        },
        {
            "prefix": "Drosophila",
            "fna": data_dir / "drosophila/drosophila_genomic.fna",
            "gff": data_dir / "drosophila/drosophila_genomic.gff",
            "max_multi": 3000,
            "max_single": 500,
            "max_neg": 1200,
            "max_antisense": 800,
        },
        {
            "prefix": "Arabidopsis",
            "fna": data_dir / "arabidopsis/arabidopsis_genomic.fna",
            "gff": data_dir / "arabidopsis/arabidopsis_genomic.gff",
            "max_multi": 3000,
            "max_single": 500,
            "max_neg": 1500,
            "max_antisense": 1000,
        },
        {
            "prefix": "Chlamydomonas",
            "fna": data_dir / "chlamydomonas/chlamydomonas_genomic.fna",
            "gff": data_dir / "chlamydomonas/chlamydomonas_genomic.gff",
            "max_multi": 2750,
            "max_single": 500,
            "max_neg": 900,
            "max_antisense": 600,
        },
    ]

    all_samples = []
    total_multi = 0
    total_single = 0
    total_neg = 0
    total_antisense = 0

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
            max_len=10000,
            random_pad_range=(400, 1800),
            seed=seed,
        )
        print(f"    -> Extracted {len(multi):4,d} spliced multi-exon + {len(single):4,d} single-exon wide-context windows.")
        all_samples.extend(multi)
        all_samples.extend(single)
        total_multi += len(multi)
        total_single += len(single)

        # Extract authentic pure intergenic negative windows
        negs = extract_pure_negative_windows(
            gff_path=gff,
            chroms=chroms,
            prefix=prefix,
            target_count=cfg["max_neg"],
            min_len=1500,
            max_len=5000,
            buffer_bp=500,
            seed=seed,
        )
        # Extract hard-negative antisense shadow-ORF windows
        anti_negs = extract_antisense_negative_windows(
            gff_path=gff,
            chroms=chroms,
            prefix=prefix,
            target_count=cfg["max_antisense"],
            seed=seed,
        )
        print(f"    -> Extracted {len(negs):4,d} pure intergenic + {len(anti_negs):4,d} antisense shadow-ORF negative windows.")
        all_samples.extend(negs)
        all_samples.extend(anti_negs)
        total_neg += len(negs)
        total_antisense += len(anti_negs)

    rng = random.Random(seed)
    rng.shuffle(all_samples)

    print("\n" + "=" * 70)
    print(f"[+] Successfully Compiled Scaled Cohort: {len(all_samples):,d} Total Training Samples")
    if all_samples:
        print(f"    - Multi-Exon Spliced Windows: {total_multi:,d} ({total_multi/len(all_samples)*100:.1f}%)")
        print(f"    - Single-Exon Windows:        {total_single:,d} ({total_single/len(all_samples)*100:.1f}%)")
        print(f"    - Pure Intergenic Negatives:  {total_neg:,d} ({total_neg/len(all_samples)*100:.1f}%)")
        print(f"    - Antisense Shadow Negatives: {total_antisense:,d} ({total_antisense/len(all_samples)*100:.1f}%)")
    print("=" * 70 + "\n")

    return all_samples

