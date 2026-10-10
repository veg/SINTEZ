"""
Transposable Element & Mobile Element Classifier (TEClassifier)
==============================================================
Provides stratified classification of gene predictions into host cellular genes
vs. transposable elements and mobile element derivatives.

Multi-Signal Architecture:
1. Curated Catalytic Signatures:
   - Reverse Transcriptase (RVT) catalytic triad & primer-binding domains.
   - Retroviral / Retrotransposon Integrase (rve) & Aspartyl Protease (RVP).
   - Retroviral Nucleocapsid Zinc Knuckles (CX2CX4HX4C).
   - Class II DNA Transposases (DDE, Mariner/Tc1, P-element, 1360/hoppel).
   - Non-LTR LINE / Endonuclease domains (CR1, INE-1).
2. Autonomous Genomic Copy-Number Indexing:
   - Fast zero-dependency k-mer depth profiling.
   - Discovers high-copy multi-locus mobile elements without external databases.
3. Native Soft-Masking Awareness:
   - Automatically detects RepeatMasker / Dfam lowercase soft-masked segments.
"""

from __future__ import annotations
import re
from typing import List, Dict, Optional, Tuple
from collections import Counter
import numpy as np


class TransposonClassifier:
    """
    Zero-dependency multi-signal transposable element classifier for eukaryotic genomes.
    """

    # Curated regex signatures for core TE catalytic centers (bounded, non-backtracking)
    CATALYTIC_SIGNATURES = {
        "rvt_catalytic_triad": (
            re.compile(r"[FYWL][LIVM][GSA][DN][DN]|[FY]MDD[LIVM]"),
            "LTR_or_LINE/ReverseTranscriptase",
        ),
        "rvt_primer_binding": (
            re.compile(r"LG[ILV]P[QE]G|[ILVM]G[FY].[FY]R"),
            "LTR_or_LINE/ReverseTranscriptase",
        ),
        "retroviral_protease": (
            re.compile(r"[IVL][DSH][STG]G[SA]"),
            "LTR_Retrotransposon/Protease",
        ),
        "retroviral_zinc_knuckle": (
            re.compile(r"C.{2}C.{4}H.{4}C"),
            "LTR_Retrotransposon/Gag",
        ),
        "transposase_1360_hoppel": (
            re.compile(r"MRNAIRFF|TRFFGRGSR|SRIFVRERESE"),
            "DNA_Transposon/1360_hoppel",
        ),
        "ine1_endonuclease": (
            re.compile(r"TFMYLYLLY|LCLYNLNLN|VGVATWVNK"),
            "LINE_Retrotransposon/INE-1",
        ),
        "mcclintock_ltr": (
            re.compile(r"DPQTLYTFIS|EIAQKLNSS|YRIAIDYRYL"),
            "LTR_Retrotransposon/McClintock",
        ),
        "mariner_tc1_transposase": (
            re.compile(r"W[RK].[DE].{10,25}[DE]"),
            "DNA_Transposon/Mariner_Tc1",
        ),
    }

    def __init__(
        self,
        genome_seq: Optional[str] = None,
        k: int = 17,
        min_repeat_depth: float = 3.0,
        min_masked_fraction: float = 0.50,
    ):
        self.k = k
        self.min_repeat_depth = min_repeat_depth
        self.min_masked_fraction = min_masked_fraction
        self.kmer_index: Optional[Dict[str, int]] = None

        if genome_seq:
            self.index_genome(genome_seq)

    def index_genome(self, genome_seq: str) -> None:
        """Indexes genome sequence into k-mers for fast copy-number estimation."""
        counts = {}
        seq_u = genome_seq.upper()
        L = len(seq_u)
        k = self.k
        for i in range(0, L - k + 1):
            km = seq_u[i:i+k]
            if "N" not in km:
                counts[km] = counts.get(km, 0) + 1
        self.kmer_index = counts

    def classify_protein(
        self,
        pep: str,
        spans: List[Tuple[int, int]],
        genome_seq: str,
    ) -> Dict:
        """
        Classifies a single protein candidate into host vs. transposable element.
        """
        matched_signatures = []
        inferred_family = "none"

        # 1. Signature Motif Scan
        for sig_name, (pattern, fam_desc) in self.CATALYTIC_SIGNATURES.items():
            if pattern.search(pep):
                matched_signatures.append(sig_name)
                if inferred_family == "none":
                    inferred_family = fam_desc

        # 2. Genomic K-mer Depth
        repeat_depth = 1.0
        if self.kmer_index:
            depths = []
            seq_u = genome_seq.upper()
            k = self.k
            for s, e in spans:
                sub = seq_u[s:e]
                for i in range(len(sub) - k + 1):
                    km = sub[i:i+k]
                    if km in self.kmer_index:
                        depths.append(self.kmer_index[km])
            if depths:
                repeat_depth = float(np.median(depths))

        # 3. Soft-Masking Check
        total_len = sum(e - s for s, e in spans)
        lower_cnt = sum(sum(1 for c in genome_seq[s:e] if c.islower()) for s, e in spans)
        masked_fraction = (lower_cnt / total_len) if total_len > 0 else 0.0

        is_te = False
        decision_reason = "host_protein"

        if matched_signatures:
            is_te = True
            decision_reason = f"catalytic_signature:{matched_signatures[0]}"
        elif repeat_depth >= self.min_repeat_depth:
            is_te = True
            decision_reason = f"high_copy_repeat:{repeat_depth:.1f}x"
            if inferred_family == "none":
                inferred_family = f"RepeatFamily/{repeat_depth:.0f}x"
        elif masked_fraction >= self.min_masked_fraction:
            is_te = True
            decision_reason = f"soft_masked_repeat:{masked_fraction*100:.1f}%"
            if inferred_family == "none":
                inferred_family = "SoftMaskedRepeat"

        return {
            "is_transposon": is_te,
            "biotype": "transposon_derived" if is_te else "protein_coding",
            "gene_type": "mobile_element" if is_te else "host",
            "te_family": inferred_family,
            "te_signatures": matched_signatures,
            "repeat_depth": repeat_depth,
            "masked_fraction": masked_fraction,
            "decision_reason": decision_reason,
        }

    def stratify_predictions(
        self,
        predictions: List[Dict],
        genome_seq: str,
        translate_fn=None,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Classifies all predictions and annotates each dict with TE metadata.
        Returns tuple of (host_predictions, transposon_predictions).
        """
        host_preds = []
        te_preds = []

        for p in predictions:
            spans = p["spans"]
            strand = p.get("strand", "+")

            # Extract peptide if not already present
            pep = p.get("peptide")
            if not pep:
                cds_dna = "".join(genome_seq[s:e] for s, e in spans)
                if translate_fn:
                    pep = translate_fn(cds_dna, strand)
                else:
                    from Bio.Seq import Seq
                    sobj = Seq(cds_dna)
                    if strand == "-":
                        sobj = sobj.reverse_complement()
                    pep = str(sobj.translate(to_stop=False))

            classification = self.classify_protein(pep, spans, genome_seq)
            p.update(classification)

            if classification["is_transposon"]:
                te_preds.append(p)
            else:
                host_preds.append(p)

        return host_preds, te_preds
