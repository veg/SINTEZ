"""
Topological Splicing DAG Decoder
================================
Implements dynamic programming beam search over directed acyclic gene graphs (DAGs)
with:
1. Pareto-frontier multi-objective candidate retention across exon topologies.
2. Jacobson-Stockmayer physical polymer loop closure penalty for intron excision.
3. Two-stage hierarchical pruning (DNA graph pruning before biophysical oracle evaluation).
"""

from __future__ import annotations
import time
import math
from typing import List, Tuple, Dict, Optional, Set

import numpy as np
import torch

from .translation_operator import SplicedTranslationOperator
from .protein_oracle import ProteinEnergyOracle


class TopologicalDAGDecoder:
    """Decodes spliced eukaryotic gene models using dual-space DAG search."""

    def __init__(
        self,
        min_single_len: int = 240,
        min_multi_len: int = 240,
        max_intron_len: int = 1500,
        max_exons: int = 8,
        lambda_intron: float = 4.5,
        min_donor_prob: float = 0.005,
        min_accept_prob: float = 0.005,
        energy_cutoff: float = 13.0,
    ):
        self.min_single_len = min_single_len
        self.min_multi_len = min_multi_len
        self.max_intron_len = max_intron_len
        self.max_exons = max_exons
        self.lambda_intron = lambda_intron
        self.min_donor_prob = min_donor_prob
        self.min_accept_prob = min_accept_prob
        self.energy_cutoff = energy_cutoff
        self.operator = SplicedTranslationOperator()

    def decode_strand(
        self,
        seq: str,
        preds: Dict[str, np.ndarray],
        oracle: ProteinEnergyOracle,
        strand_label: str = "+",
    ) -> List[Dict]:
        """
        Decodes gene models on a 5'->3' sequence using Pareto-frontier DAG search
        and biophysical ESM-2 oracle scoring.
        """
        t0 = time.time()
        L = len(seq)
        dna_upper = seq.upper()

        start_p = preds["start"]
        stop_p = preds["stop"]
        donor_p = preds["donor"]
        accept_p = preds["acceptor"]
        coding_p = preds["coding"]

        th_start = float(np.percentile(start_p, 90.0))
        cand_starts = [i for i in range(L - 3) if dna_upper[i:i+3] == "ATG" and start_p[i] >= th_start]
        cand_acceptors = set(i + 1 for i in range(1, L - 1) if dna_upper[i-1:i+1] == "AG" and accept_p[i] >= self.min_accept_prob)
        stops_set = {"TAA", "TAG", "TGA"}

        all_best_per_start = []

        for s in cand_starts:
            active_paths = [([], s, 0, 1, 0, 0.0)]
            completed_for_s = []

            for depth in range(1, self.max_exons + 1):
                next_paths = []
                for spans, last_end, phase, exon_count, total_len, junc_score in active_paths:
                    offset = (3 - phase) % 3
                    first_codon = last_end + offset
                    first_stop = None
                    for pos in range(first_codon, min(L - 2, last_end + 15000), 3):
                        if dna_upper[pos:pos+3] in stops_set:
                            first_stop = pos
                            break
                    if first_stop is None:
                        continue

                    e = first_stop + 3
                    ex_len = e - last_end
                    t_len = total_len + ex_len

                    # Option A: In-frame stop codon termination
                    if t_len >= (self.min_single_len if depth == 1 else self.min_multi_len):
                        cand_spans = spans + [(last_end, e)]
                        mean_cod = float(np.mean([coding_p[es:ee].mean() for es, ee in cand_spans]))
                        evidence_len = math.log(t_len) * 2.0 + 1.5 * min(t_len / 300.0, 5.0)
                        dna_score = evidence_len + mean_cod * 2.0 + junc_score
                        completed_for_s.append({
                            "spans": cand_spans,
                            "len": t_len,
                            "exons": len(cand_spans),
                            "mean_coding": mean_cod,
                            "junc_score": junc_score,
                            "dna_score": dna_score,
                        })

                    # Option B: Extend to donor near stop codon and splice
                    if depth < self.max_exons and (first_stop - last_end) >= 30:
                        donor_start = max(last_end + 30, first_stop - 400)
                        valid_donors = []
                        for d in range(donor_start, first_stop + 1):
                            if dna_upper[d:d+2] in {"GT", "GC"} and donor_p[d] >= self.min_donor_prob:
                                valid_donors.append((d, float(donor_p[d])))
                        valid_donors.sort(key=lambda x: x[1], reverse=True)
                        for d, d_score in valid_donors[:3]:
                            d_phase = (phase + (d - last_end)) % 3
                            valid_a = [(a, float(accept_p[a-1])) for a in range(d + 40, min(L, d + self.max_intron_len)) if a in cand_acceptors]
                            valid_a.sort(key=lambda x: x[1], reverse=True)
                            for a, a_score in valid_a[:2]:
                                next_spans = spans + [(last_end, d)]
                                next_len = total_len + (d - last_end)
                                intron_len = a - d
                                # Jacobson-Stockmayer polymer looping penalty
                                loop_penalty = 1.5 * max(0.0, math.log(max(1.0, intron_len / 80.0)))
                                intron_gain = math.log(1.0 + 100.0 * d_score) + math.log(1.0 + 100.0 * a_score) - self.lambda_intron - loop_penalty
                                if intron_gain > 0.0:
                                    next_paths.append((next_spans, a, d_phase, exon_count + 1, next_len, junc_score + intron_gain))

                next_paths.sort(key=lambda x: x[5], reverse=True)
                active_paths = next_paths[:4]

            # Pareto Principle: Group by exon count and preserve top candidate of EACH topology
            if completed_for_s:
                by_exon = {}
                for c in completed_for_s:
                    k = c["exons"]
                    if k not in by_exon or c["dna_score"] > by_exon[k]["dna_score"]:
                        by_exon[k] = c
                for c in by_exon.values():
                    all_best_per_start.append(c)

        # Pareto-Frontier NMS in DNA space (only suppress within same exon count)
        all_best_per_start.sort(key=lambda x: x["dna_score"], reverse=True)
        dna_retained: List[Dict] = []
        for c in all_best_per_start:
            cs, ce = c["spans"][0][0], c["spans"][-1][1]
            ov = False
            for r in dna_retained:
                rs, re = r["spans"][0][0], r["spans"][-1][1]
                overlap = max(0, min(ce, re) - max(cs, rs))
                if overlap > 0.35 * min(ce - cs, re - rs) and c["exons"] == r["exons"]:
                    ov = True
                    break
            if not ov:
                dna_retained.append(c)

        # Stage 2: ESM-2 Biophysical Scoring (with Information-Theoretic Complexity Normalization)
        t_esm = time.time()
        scored: List[Dict] = []
        for c in dna_retained:
            tx = self.operator.process_transcript(seq, c["spans"], strand="+")
            if tx.is_in_frame and not tx.has_internal_stop:
                ed = oracle.score_transcript(tx)
                energy = ed["total_energy"]
                if energy <= self.energy_cutoff:
                    c["energy"] = energy
                    c["esm_cross_entropy"] = ed["esm_cross_entropy"]
                    c["protein_len"] = len(tx.protein_sequence)
                    c["score"] = c["dna_score"] - energy * 0.2
                    scored.append(c)

        # Final Intra-Strand NMS (ranked by joint dual-space score)
        scored.sort(key=lambda x: x["score"], reverse=True)
        final_retained: List[Dict] = []
        for c in scored:
            cs, ce = c["spans"][0][0], c["spans"][-1][1]
            ov = False
            for f in final_retained:
                fs, fe = f["spans"][0][0], f["spans"][-1][1]
                overlap = max(0, min(ce, fe) - max(cs, fs))
                if overlap > 0.35 * min(ce - cs, fe - fs):
                    ov = True
                    break
            if not ov:
                final_retained.append(c)

        return final_retained
