"""
Dual-Strand Conflict Resolver
==============================
Resolves overlapping candidate predictions on opposing DNA strands using
constrained conflict graph optimization and reciprocal overlap thresholds.
"""

from __future__ import annotations
from typing import List, Dict


class DualStrandResolver:
    """Resolves collisions between opposing forward and reverse strand gene models."""

    @staticmethod
    def resolve_conflicts(
        predictions: List[Dict], max_reciprocal_overlap: float = 0.15
    ) -> List[Dict]:
        """
        Suppresses lower-scoring shadow decoys while preserving genuine convergent,
        divergent, or non-overlapping gene pairs across opposing strands.
        """
        def _strand_priority(c: Dict) -> float:
            base_sc = float(c.get("score", 0.0))
            # Boost genuine multi-exon / high-boundary / high-coding genes during cross-strand arbitration
            # so a 1-exon antisense ORF inside a GC-rich exon cannot kill a spliced true gene
            n_ex = len(c.get("spans", ()))
            min_jp = float(c.get("min_junc_p", 1.0))
            term_s = float(c.get("term_sum", 0.5))
            spliced_bonus = 6.0 * (n_ex - 1) * min_jp if n_ex > 1 and min_jp >= 0.45 else 0.0
            return base_sc + spliced_bonus + 4.0 * term_s

        predictions.sort(key=_strand_priority, reverse=True)
        retained: List[Dict] = []

        for cand in predictions:
            c_s = cand["spans"][0][0]
            c_e = cand["spans"][-1][1]
            c_span_len = max(1, c_e - c_s)
            c_len = sum(e - s for s, e in cand["spans"])
            c_strand = cand["strand"]
            c_cod_ratio = float(cand.get("cod_ratio", 1.0))
            c_min_junc = float(cand.get("min_junc_p", 1.0))
            c_term_sum = float(cand.get("term_sum", 1.0))

            clash = False
            for kept in retained:
                if kept["strand"] != c_strand:
                    k_s = kept["spans"][0][0]
                    k_e = kept["spans"][-1][1]
                    k_len = sum(e - s for s, e in kept["spans"])

                    # 1. Direct coding exon collision across opposing strands
                    coding_overlap = sum(
                        max(0, min(e1, e2) - max(s1, s2))
                        for s1, e1 in cand["spans"]
                        for s2, e2 in kept["spans"]
                    )
                    eff_max_ov = max_reciprocal_overlap
                    # Fix 9B: Protect genuine convergent/overlapping coding genes from cross-strand suppression
                    if (
                        c_len >= 240
                        and c_cod_ratio >= 0.72
                        and c_term_sum >= 0.45
                        and float(cand.get("mean_coding", 0.0)) >= 0.55
                    ):
                        eff_max_ov = max(eff_max_ov, 0.35)
                    if coding_overlap > eff_max_ov * min(c_len, k_len):
                        clash = True
                        break

                    # 2. Fix 5B: Antisense Intronic-Span Containment Suppression
                    # Suppress short, low-confidence antisense decoys nested inside the genomic span of a dominant gene
                    span_overlap = max(0, min(c_e, k_e) - max(c_s, k_s))
                    if (
                        span_overlap >= 0.45 * c_span_len
                        and c_len < 600
                        and k_len > c_len * 1.35
                        and float(kept.get("mean_coding", 1.0)) >= float(cand.get("mean_coding", 0.0)) - 0.05
                        and (c_cod_ratio < 0.72 or (c_min_junc < 0.55 and c_term_sum < 0.55))
                    ):
                        clash = True
                        break

            if not clash:
                retained.append(cand)

        retained.sort(key=lambda x: x["spans"][0][0])
        return retained
