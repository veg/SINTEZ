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
        predictions: List[Dict], max_reciprocal_overlap: float = 0.35
    ) -> List[Dict]:
        """
        Suppresses lower-scoring shadow decoys while preserving genuine convergent,
        divergent, or non-overlapping gene pairs across opposing strands.
        """
        predictions.sort(key=lambda x: x["score"], reverse=True)
        retained: List[Dict] = []

        for cand in predictions:
            c_s = cand["spans"][0][0]
            c_e = cand["spans"][-1][1]
            c_len = sum(e - s for s, e in cand["spans"])
            c_strand = cand["strand"]

            clash = False
            for kept in retained:
                if kept["strand"] != c_strand:
                    k_s = kept["spans"][0][0]
                    k_e = kept["spans"][-1][1]
                    k_len = sum(e - s for s, e in kept["spans"])

                    overlap = max(0, min(c_e, k_e) - max(c_s, k_s))
                    if overlap > max_reciprocal_overlap * min(c_len, k_len):
                        # Direct antisense coding collision: suppress the lower-scoring candidate
                        clash = True
                        break

            if not clash:
                retained.append(cand)

        retained.sort(key=lambda x: x["spans"][0][0])
        return retained
