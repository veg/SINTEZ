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
        # Empirical background calibration: adapts neutral point dynamically to GC regime
        p0_est = float(np.percentile(coding_p, 30.0)) if L > 1000 else 0.15
        p0 = max(0.08, min(p0_est, 0.38))
        llr_offset = math.log(p0 / (1.0 - p0))

        eps = 1e-4
        clamped_cod = np.clip(coding_p, eps, 1.0 - eps)
        log_odds = np.log(clamped_cod / (1.0 - clamped_cod)) - llr_offset

        cumsum_llr = np.zeros(L + 1, dtype=np.float64)
        cumsum_llr[1:] = np.cumsum(log_odds)

        cumsum_cod = np.zeros(L + 1, dtype=np.float64)
        cumsum_cod[1:] = np.cumsum(coding_p)

        # Optional 3-frame phase-specific prefix sums from 7-class backbone
        has_phase = "phase" in preds and preds["phase"] is not None
        cumsum_frame = np.zeros((3, L + 1), dtype=np.float64)
        if has_phase:
            phase_p = preds["phase"]  # shape (3, L): codon positions 0, 1, 2
            idx_arr = np.arange(L)
            for f_idx in (0, 1, 2):
                cumsum_frame[f_idx, 1:] = np.cumsum(phase_p[(idx_arr - f_idx) % 3, idx_arr])

        # Optional neural foldability / coding quality prefix sum (Foundation v3)
        has_fold = "foldability" in preds and preds["foldability"] is not None
        cumsum_fold = np.zeros(L + 1, dtype=np.float64)
        if has_fold:
            cumsum_fold[1:] = np.cumsum(preds["foldability"])

        def get_exon_metrics(s: int, e: int) -> Tuple[float, float]:
            if e <= s:
                return 0.0, 0.0
            tot_llr = cumsum_llr[e] - cumsum_llr[s]
            mean_p = (cumsum_cod[e] - cumsum_cod[s]) / (e - s)
            return float(tot_llr), float(mean_p)

        def get_frame_mean(f_idx: int, s: int, e: int) -> float:
            if not has_phase or e <= s:
                return 0.0
            return float(cumsum_frame[f_idx, e] - cumsum_frame[f_idx, s]) / (e - s)

        def get_fold_mean(s: int, e: int) -> float:
            if not has_fold or e <= s:
                return 0.0
            return float(cumsum_fold[e] - cumsum_fold[s]) / (e - s)

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

        # 4. Candidate Starts: Primary threshold + secondary relative peak support for non-canonical contexts
        atg_indices = [i for i in range(L - 2) if dna_upper[i:i+3] == "ATG"]
        cand_starts_set = set()
        if atg_indices:
            atg_scores = start_p[atg_indices]
            p80 = float(np.percentile(atg_scores, 80.0))
            th_start = max(self.min_start_prob, min(p80, 0.05))
            for i in atg_indices:
                if start_p[i] >= th_start:
                    cand_starts_set.add(i)

            # Secondary relative peaks for non-canonical/GC-rich Kozak contexts
            if len(cand_starts_set) < max(50, L // 3000):
                for idx, i in enumerate(atg_indices):
                    if start_p[i] >= 0.0003:
                        _, dn_cod = get_exon_metrics(i, min(L, i + 120))
                        if dn_cod >= max(0.40, p0 * 2.0):
                            cand_starts_set.add(i)

        cand_starts = sorted(list(cand_starts_set))
        all_best_per_start = []
        min_cod_th = max(0.30, p0 * 1.35)

        for s in cand_starts:
            init_signal = math.log(1.0 + 50.0 * max(float(start_p[s]), 0.003))
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
                    is_single_exon = (depth == 1)
                    min_len_req = self.min_single_len if is_single_exon else self.min_multi_len
                    if t_len >= min_len_req:
                        cand_spans = spans + [(last_end, e)]
                        tot_llr = 0.0
                        tot_p_sum = 0.0
                        tot_frame_p_sum = 0.0
                        tot_fold_sum = 0.0
                        cur_ph = 0
                        for es, ee in cand_spans:
                            s_llr, s_mp = get_exon_metrics(es, ee)
                            tot_llr += s_llr
                            tot_p_sum += s_mp * (ee - es)
                            if has_phase:
                                f_ex = (es - cur_ph) % 3
                                tot_frame_p_sum += get_frame_mean(f_ex, es, ee) * (ee - es)
                                cur_ph = (cur_ph + (ee - es)) % 3
                            if has_fold:
                                tot_fold_sum += get_fold_mean(es, ee) * (ee - es)
                        mean_cod = tot_p_sum / max(1, t_len)
                        mean_frame_cod = (tot_frame_p_sum / max(1, t_len)) if has_phase else mean_cod
                        frame_purity = (mean_frame_cod / max(1e-4, mean_cod)) if has_phase else 1.0
                        mean_fold = (tot_fold_sum / max(1, t_len)) if has_fold else mean_cod

                        # Single-exon genes lack splicing signals, requiring stricter coding purity
                        min_cod_req = max(0.48, p0 * 1.7) if is_single_exon else min_cod_th
                        min_llr_req = 1.0 if is_single_exon else 0.0

                        stop_score = math.log(1.0 + 50.0 * float(stop_p[first_stop]))
                        len_evidence = 2.0 * math.log(t_len) - max(0.0, (240 - t_len) / 20.0)

                        total_signal = cum_signal + stop_score
                        norm_llr = tot_llr / math.sqrt(max(1, t_len))
                        frame_bonus = (2.5 * mean_frame_cod) if has_phase else 0.0
                        fold_bonus = (2.0 * mean_fold) if has_fold else 0.0
                        dna_score = total_signal + norm_llr * 0.5 + mean_cod * 6.0 + frame_bonus + fold_bonus + len_evidence

                        # Filter candidates with net positive coding evidence, sufficient coding density, and reading-frame coherence
                        if tot_llr > min_llr_req and mean_cod >= min_cod_req and frame_purity >= 0.25 and (not has_fold or mean_fold >= 0.25):
                            completed_for_s.append({
                                "spans": cand_spans,
                                "len": t_len,
                                "exons": len(cand_spans),
                                "mean_coding": mean_cod,
                                "dna_score": dna_score,
                            })

                    # Option B: Spliced Extension via Boundary Step-Derivative Donor Search
                    if depth < self.max_exons and (first_stop - last_end) >= self.min_internal_exon_len:
                        d_start = last_end + self.min_internal_exon_len
                        d_i0 = bisect.bisect_left(donor_list, d_start)
                        d_i1 = bisect.bisect_right(donor_list, first_stop)
                        valid_donors = []
                        f_ex = (last_end - phase) % 3
                        for idx in range(d_i0, d_i1):
                            d = donor_list[idx]
                            d_p = float(donor_p[d])
                            e_llr, e_mp = get_exon_metrics(last_end, d)
                            e_f_mp = get_frame_mean(f_ex, last_end, d)
                            
                            # Boundary Step-Derivative: Exon-exit drop into intron
                            up_s = max(last_end, d - 30)
                            _, up_mp = get_exon_metrics(up_s, d)
                            dn_e = min(L, d + 30)
                            _, dn_mp = get_exon_metrics(d, dn_e)
                            coding_drop = max(0.0, up_mp - dn_mp)

                            # Saliency combines point sensor, coding step-down, and reading-frame coherence
                            saliency = (
                                math.log(1.0 + 100.0 * d_p)
                                + 8.0 * d_p
                                + 3.0 * coding_drop
                                + 2.0 * e_mp
                                + (2.5 * e_f_mp if has_phase else 0.0)
                            )
                            if e_mp >= max(0.20, p0 * 1.5) and (not has_phase or (e_f_mp / max(1e-4, e_mp)) >= 0.20):
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
                                ap = float(accept_p[a-1])
                                # PPT (polypyrimidine tract) density in [-25, -4] upstream window
                                ppt_win = dna_upper[max(0, a - 25):max(0, a - 4)]
                                ppt_ratio = (sum(1 for c in ppt_win if c in {'C', 'T'}) / len(ppt_win)) if len(ppt_win) >= 10 else 0.5
                                
                                # Intron purity & Acceptor Boundary Step-Derivative + Reading-Frame Lock
                                _, int_mp = get_exon_metrics(d, a)
                                up_a_s = max(d, a - 30)
                                _, up_a_mp = get_exon_metrics(up_a_s, a)
                                dn_a_e = min(L, a + 45)
                                _, dn_a_mp = get_exon_metrics(a, dn_a_e)
                                coding_rise = max(0.0, dn_a_mp - up_a_mp)
                                next_f = (a - d_phase) % 3
                                dn_f_mp = get_frame_mean(next_f, a, dn_a_e)

                                a_saliency = (
                                    (math.log(1.0 + 100.0 * ap) + 8.0 * ap) * (0.5 + ppt_ratio)
                                    + 2.0 * coding_rise
                                    + (3.0 * dn_f_mp if has_phase else 0.0)
                                    - 3.5 * int_mp
                                )
                                valid_a.append((a, ap, a_saliency, dn_a_mp, dn_f_mp))

                            valid_a.sort(key=lambda x: x[2], reverse=True)

                            for a, a_score, _, dn_a_mp, dn_f_mp in valid_a[:4]:
                                # Immediate Split-Codon Phase Pruning: Reject lethal in-frame nonsense stops
                                if d_phase == 1 and d >= 1 and a + 2 <= L:
                                    split_codon = dna_upper[d - 1] + dna_upper[a:a + 2]
                                    if split_codon in {"TAA", "TAG", "TGA"}:
                                        continue
                                elif d_phase == 2 and d >= 2 and a + 1 <= L:
                                    split_codon = dna_upper[d - 2:d] + dna_upper[a]
                                    if split_codon in {"TAA", "TAG", "TGA"}:
                                        continue

                                intron_len = a - d
                                loop_penalty = 1.0 * max(0.0, math.log(max(1.0, intron_len / 80.0)))
                                
                                # Coding-excision penalty prevents excising high-scoring coding sequence
                                int_llr, _ = get_exon_metrics(d, a)
                                coding_excision_penalty = max(0.0, int_llr * 0.5)

                                intron_gain = (
                                    math.log(1.0 + 50.0 * d_score)
                                    + math.log(1.0 + 50.0 * a_score)
                                    - self.lambda_intron
                                    - loop_penalty
                                    - coding_excision_penalty
                                )

                                # Phase-lock adjustment on downstream exon entry
                                if has_phase:
                                    if dn_a_mp > 0.20 and (dn_f_mp / dn_a_mp) < 0.22:
                                        intron_gain -= 2.0
                                    elif dn_f_mp >= max(0.25, p0 * 1.2):
                                        intron_gain += 0.8 * dn_f_mp

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

            # Pareto Principle: Group by exon count and preserve top topologies per start codon
            if completed_for_s:
                by_exon = {}
                for c in completed_for_s:
                    k = c["exons"]
                    if k not in by_exon or c["dna_score"] > by_exon[k]["dna_score"]:
                        by_exon[k] = c
                top_for_s = sorted(by_exon.values(), key=lambda x: x["dna_score"], reverse=True)
                kept_s = top_for_s[:4]
                if 1 in by_exon and by_exon[1] not in kept_s:
                    kept_s.append(by_exon[1])
                all_best_per_start.extend(kept_s)

        # Pareto-Frontier NMS in DNA space (suppress redundant same-topology or near-identical coding loci)
        all_best_per_start.sort(key=lambda x: x["dna_score"], reverse=True)
        dna_retained: List[Dict] = []
        for c in all_best_per_start:
            cs, ce = c["spans"][0][0], c["spans"][-1][1]
            ov = False
            for r in dna_retained:
                rs, re = r["spans"][0][0], r["spans"][-1][1]
                s_ov = max(0, min(ce, re) - max(cs, rs))
                if s_ov <= 0:
                    continue
                if c["exons"] == r["exons"] and s_ov > 0.35 * min(ce - cs, re - rs):
                    ov = True
                    break
                if s_ov > 0.80 * min(ce - cs, re - rs):
                    c_ov = compute_coding_overlap(c["spans"], r["spans"])
                    if c_ov > 0.85 * max(c["len"], r["len"]):
                        ov = True
                        break
            if not ov:
                dna_retained.append(c)

        # Stage 2: Batched ESM-2 Biophysical Scoring
        valid_candidates = []
        for c in dna_retained:
            tx = self.operator.process_transcript(seq, c["spans"], strand="+")
            if tx.is_in_frame and not tx.has_internal_stop:
                valid_candidates.append((c, tx))

        if valid_candidates and hasattr(oracle, "score_batch"):
            oracle.score_batch([tx.protein_sequence for _, tx in valid_candidates], batch_size=16)

        scored: List[Dict] = []
        for c, tx in valid_candidates:
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

