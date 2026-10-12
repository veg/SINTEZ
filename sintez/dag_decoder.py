"""
Topological Splicing DAG Decoder
================================
Principled dual-space dynamic programming decoder for eukaryotic gene structures.

Core Mathematical Formulations:
1. Log-Odds Coding Likelihood Prefix-Sum Integration:
   O(1) exact evaluation of sequence coding likelihood ratio vs. non-coding background.
   Penalizes spurious intronic pseudo-exons exponentially (e.g. -1.73 per bp when P < 0.15).
2. Full-Horizon Donor Splice Search:
   Scans canonical GT/GC splice donors across the entire valid open reading frame
   interval [last_end + min_exon_len, first_stop], ranked by joint donor probability
   and exon coding density, eliminating the 400 bp ORF truncation bottleneck.
3. Parametric Spliceosome Intron Length Prior:
   P_len(|i|) models the biophysical polymer loop penalty and spliceosome assembly
   constraints across 35 <= |i| <= max_intron_len.
4. Reading Frame Phase-Decoupled Dynamic Programming Beam:
   Maintains top-K candidate paths independently per reading frame phase (0, 1, 2),
   preventing phase monopolization and beam starvation in complex multi-exon loci.
5. Soft-Margin Splice Junction Bridging:
   Supports multi-exon loci by allowing strong flanking coding density
   to bridge marginal canonical splice junctions without premature model termination.
6. ESM-2 Biophysical Foldability Verification:
   Translates candidate transcripts in-frame and rejects non-viable topologies
   using ESM-2 pseudo-perplexity / pseudo-energy scoring.
"""

from __future__ import annotations
import time
import math
import bisect
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Set

import numpy as np
import torch

from .translation_operator import SplicedTranslationOperator
from .protein_oracle import ProteinEnergyOracle


def compute_coding_overlap(spans1: List[Tuple[int, int]], spans2: List[Tuple[int, int]]) -> int:
    """Computes exact base-pair overlap between two sets of coding exons."""
    total_ov = 0
    for s1, e1 in spans1:
        for s2, e2 in spans2:
            if e2 <= s1:
                continue
            if s2 >= e1:
                break
            ov = min(e1, e2) - max(s1, s2)
            if ov > 0:
                total_ov += ov
    return total_ov


class TopologicalDAGDecoder:
    """Decodes spliced eukaryotic gene models using dual-space DAG search."""

    def __init__(
        self,
        min_single_len: int = 240,
        min_multi_len: int = 240,
        max_intron_len: int = 1500,
        min_internal_exon_len: int = 15,
        max_exons: int = 16,
        beam_width: int = 8,
        beam_width_per_phase: Optional[int] = None,
        lambda_intron: float = 4.0,
        min_intron_gain: float = -1.5,
        min_donor_prob: float = 0.008,
        min_accept_prob: float = 0.008,
        min_start_prob: float = 0.001,
        energy_cutoff: float = 14.0,
    ):
        self.min_single_len = min_single_len
        self.min_multi_len = min_multi_len
        self.max_intron_len = max_intron_len
        self.min_internal_exon_len = min_internal_exon_len
        self.max_exons = max_exons
        self.beam_width = beam_width
        self.beam_width_per_phase = beam_width_per_phase or max(4, beam_width // 2)
        self.lambda_intron = lambda_intron
        self.min_intron_gain = min_intron_gain
        self.min_donor_prob = min_donor_prob
        self.min_accept_prob = min_accept_prob
        self.min_start_prob = min_start_prob
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
        Decodes gene models on a 5'->3' sequence using principled log-odds DAG search
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

        # 1. Prefix-sum arrays for O(1) log-odds and coding probability integration
        # Empirical background calibration: adapts neutral point dynamically
        p0_est = float(np.percentile(coding_p, 25.0)) if L > 1000 else 0.10
        p0 = max(0.08, min(p0_est, 0.20))
        llr_offset = math.log(p0 / (1.0 - p0))

        eps = 1e-4
        clamped_cod = np.clip(coding_p, eps, 1.0 - eps)
        log_odds = np.log(clamped_cod / (1.0 - clamped_cod)) - llr_offset

        cumsum_llr = np.zeros(L + 1, dtype=np.float64)
        cumsum_llr[1:] = np.cumsum(log_odds)

        cumsum_cod = np.zeros(L + 1, dtype=np.float64)
        cumsum_cod[1:] = np.cumsum(coding_p)

        def get_exon_metrics(s: int, e: int) -> Tuple[float, float]:
            if e <= s:
                return 0.0, 0.0
            tot_llr = cumsum_llr[e] - cumsum_llr[s]
            mean_p = (cumsum_cod[e] - cumsum_cod[s]) / (e - s)
            return float(tot_llr), float(mean_p)

        # 2. Pre-index stop codons by reading frame phase for O(log N) lookup
        stops_set = {"TAA", "TAG", "TGA"}
        stop_positions = {0: [], 1: [], 2: []}
        for pos in range(0, L - 2):
            if dna_upper[pos:pos+3] in stops_set:
                stop_positions[pos % 3].append(pos)

        # 3. Pre-filter and sort candidate donor and acceptor sites
        raw_donors = np.where(donor_p >= self.min_donor_prob)[0]
        donor_list = [int(d) for d in raw_donors if d < L - 2 and dna_upper[d:d+2] in {"GT", "GC"}]

        raw_acceptors = np.where(accept_p >= self.min_accept_prob)[0]
        acceptor_list = [int(a + 1) for a in raw_acceptors if 0 < a < L - 1 and dna_upper[a-1:a+1] == "AG"]

        # 4. Candidate Starts calibrated to distribution of ATG scores
        atg_indices = [i for i in range(L - 2) if dna_upper[i:i+3] == "ATG"]
        if atg_indices:
            atg_scores = start_p[atg_indices]
            p80 = float(np.percentile(atg_scores, 80.0))
            th_start = max(self.min_start_prob, min(p80, 0.05))
            cand_starts = [i for i in atg_indices if start_p[i] >= th_start]
        else:
            cand_starts = []

        all_best_per_start = []
        min_cod_th = max(0.25, p0 * 1.5)

        for s in cand_starts:
            init_signal = math.log(1.0 + 50.0 * float(start_p[s]))
            # Active paths store: (spans, last_end, phase, exon_count, total_len, cum_signal, cum_llr)
            active_paths = [([], s, 0, 1, 0, init_signal, 0.0)]
            completed_for_s = []

            for depth in range(1, self.max_exons + 1):
                next_paths = []
                for spans, last_end, phase, exon_count, total_len, cum_signal, cum_llr in active_paths:
                    offset = (3 - phase) % 3
                    first_codon = last_end + offset
                    r = first_codon % 3
                    stop_idx = bisect.bisect_left(stop_positions[r], first_codon)
                    if stop_idx >= len(stop_positions[r]):
                        continue
                    first_stop = stop_positions[r][stop_idx]
                    if first_stop > last_end + 25000:
                        continue

                    e = first_stop + 3
                    ex_len = e - last_end
                    t_len = total_len + ex_len

                    # Option A: In-frame Stop Codon Termination
                    if t_len >= (self.min_single_len if depth == 1 else self.min_multi_len):
                        cand_spans = spans + [(last_end, e)]
                        tot_llr = 0.0
                        tot_p_sum = 0.0
                        for es, ee in cand_spans:
                            s_llr, s_mp = get_exon_metrics(es, ee)
                            tot_llr += s_llr
                            tot_p_sum += s_mp * (ee - es)
                        mean_cod = tot_p_sum / max(1, t_len)

                        stop_score = math.log(1.0 + 50.0 * float(stop_p[first_stop]))
                        len_evidence = 2.0 * math.log(t_len) - max(0.0, (240 - t_len) / 20.0)

                        total_signal = cum_signal + stop_score
                        norm_llr = tot_llr / math.sqrt(max(1, t_len))
                        dna_score = total_signal + norm_llr * 0.5 + mean_cod * 6.0 + len_evidence

                        # Filter candidates with net positive coding evidence and sufficient coding density
                        if tot_llr > 0.0 and mean_cod >= min_cod_th:
                            completed_for_s.append({
                                "spans": cand_spans,
                                "len": t_len,
                                "exons": len(cand_spans),
                                "mean_coding": mean_cod,
                                "dna_score": dna_score,
                            })

                    # Option B: Spliced Extension via Full-Horizon Donor Search (Binary Search Indexed)
                    if depth < self.max_exons and (first_stop - last_end) >= self.min_internal_exon_len:
                        d_start = last_end + self.min_internal_exon_len
                        d_i0 = bisect.bisect_left(donor_list, d_start)
                        d_i1 = bisect.bisect_right(donor_list, first_stop)
                        valid_donors = []
                        for idx in range(d_i0, d_i1):
                            d = donor_list[idx]
                            d_p = float(donor_p[d])
                            e_llr, e_mp = get_exon_metrics(last_end, d)
                            saliency = math.log(1.0 + 100.0 * d_p) + 2.0 * e_mp
                            if e_mp >= max(0.20, p0 * 1.5):
                                valid_donors.append((d, d_p, e_llr, e_mp, saliency))

                        valid_donors.sort(key=lambda x: x[4], reverse=True)

                        for d, d_score, e_llr, e_mp, _ in valid_donors[:6]:
                            d_phase = (phase + (d - last_end)) % 3
                            next_len = total_len + (d - last_end)
                            next_spans = spans + [(last_end, d)]

                            a_min = d + 35
                            a_max = min(L, d + self.max_intron_len)
                            a_i0 = bisect.bisect_left(acceptor_list, a_min)
                            a_i1 = bisect.bisect_right(acceptor_list, a_max)
                            valid_a = []
                            for idx in range(a_i0, a_i1):
                                a = acceptor_list[idx]
                                valid_a.append((a, float(accept_p[a-1])))

                            valid_a.sort(key=lambda x: x[1], reverse=True)

                            for a, a_score in valid_a[:4]:
                                intron_len = a - d
                                loop_penalty = 1.0 * max(0.0, math.log(max(1.0, intron_len / 80.0)))
                                intron_gain = math.log(1.0 + 50.0 * d_score) + math.log(1.0 + 50.0 * a_score) - self.lambda_intron - loop_penalty

                                # Soft-margin bridging for high coding regions
                                if e_mp >= max(0.30, p0 * 2.0):
                                    intron_gain += 0.5

                                if intron_gain > self.min_intron_gain:
                                    new_cum_signal = cum_signal + intron_gain
                                    new_cum_llr = cum_llr + e_llr
                                    beam_score = new_cum_signal + (new_cum_llr / math.sqrt(max(1, next_len))) * 0.5
                                    next_paths.append((next_spans, a, d_phase, exon_count + 1, next_len, new_cum_signal, new_cum_llr, beam_score))

                # Phase-decoupled beam retention: top K per phase
                by_phase = defaultdict(list)
                for p_item in next_paths:
                    by_phase[p_item[2]].append(p_item)

                active_paths = []
                for ph in (0, 1, 2):
                    by_phase[ph].sort(key=lambda x: x[7], reverse=True)
                    for item in by_phase[ph][:self.beam_width_per_phase]:
                        active_paths.append(item[:7])

                if not active_paths:
                    break

            # Pareto Principle: Group by exon count and preserve top candidate of EACH topology
            if completed_for_s:
                by_exon = {}
                for c in completed_for_s:
                    k = c["exons"]
                    if k not in by_exon or c["dna_score"] > by_exon[k]["dna_score"]:
                        by_exon[k] = c
                for c in by_exon.values():
                    all_best_per_start.append(c)

        # Pareto-Frontier NMS in DNA space (only suppress within same exon count using locus span overlap)
        all_best_per_start.sort(key=lambda x: x["dna_score"], reverse=True)
        dna_retained: List[Dict] = []
        for c in all_best_per_start:
            cs, ce = c["spans"][0][0], c["spans"][-1][1]
            ov = False
            for r in dna_retained:
                if c["exons"] == r["exons"]:
                    rs, re = r["spans"][0][0], r["spans"][-1][1]
                    s_ov = max(0, min(ce, re) - max(cs, rs))
                    if s_ov > 0.35 * min(ce - cs, re - rs):
                        ov = True
                        break
            if not ov:
                dna_retained.append(c)

        # Stage 2: ESM-2 Biophysical Scoring
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
                    c["score"] = c["dna_score"] - energy * 0.25
                    scored.append(c)

        # Final Intra-Strand NMS (ranked by joint dual-space score, suppressing nested/overlapping loci)
        scored.sort(key=lambda x: x["score"], reverse=True)
        final_retained: List[Dict] = []
        for c in scored:
            cs, ce = c["spans"][0][0], c["spans"][-1][1]
            c_len = sum(e - s for s, e in c["spans"])
            ov = False
            for f in final_retained:
                fs, fe = f["spans"][0][0], f["spans"][-1][1]
                s_ov = max(0, min(ce, fe) - max(cs, fs))
                # Suppress if genomic span overlap > 35% of either candidate locus
                if s_ov > 0.35 * min(ce - cs, fe - fs):
                    ov = True
                    break
                # Also suppress on direct coding collision > 20%
                f_len = sum(e - s for s, e in f["spans"])
                c_ov = compute_coding_overlap(c["spans"], f["spans"])
                if c_ov > 0.20 * min(c_len, f_len):
                    ov = True
                    break
            if not ov:
                final_retained.append(c)

        return final_retained
