"""
Protein Foundation Energy Oracle
=================================
Evaluates the biophysical viability and evolutionary likelihood of translated
protein candidates using pre-trained Protein Language Models (ESM-2) combined
with biophysical foldability potentials.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import math
import torch
import torch.nn as nn
import esm


class ProteinEnergyOracle:
    """
    Computes an evolutionary and biophysical Energy Function E(Protein).
    Lower energy indicates a natural, stable, foldable protein.
    Higher energy indicates an unviable translation, frameshift, or non-coding junk.
    """
    def __init__(
        self,
        model_name: str = "esm2_t6_8M_UR50D",
        device: Optional[str] = None,
        ptc_penalty: float = 100.0,
        min_length: int = 25,
        short_penalty: float = 2.0,
    ):
        """
        Args:
            model_name: Pretrained ESM-2 model to load.
            device: 'cuda', 'mps', or 'cpu'. Automatically resolved if None.
            ptc_penalty: Energy penalty added for premature termination codons.
            min_length: Minimum expected amino acid length for viable domain.
            short_penalty: Multiplier per missing amino acid below min_length.
        """
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                # CPU is 7.2 ms/peptide on Apple Silicon and eliminates MPSGraph dynamic-length JIT compilation leaks
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"[ProteinOracle] Loading {model_name} on {self.device}...")
        self.esm_model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.esm_model.to(self.device)
        self.esm_model.eval()
        
        # Freeze ESM parameters - it acts as an immutable physical energy surface
        for p in self.esm_model.parameters():
            p.requires_grad = False

        self.batch_converter = self.alphabet.get_batch_converter()
        self.ptc_penalty = ptc_penalty
        self.min_length = min_length
        self.short_penalty = short_penalty
        self._cache: Dict[str, Dict[str, float]] = {}

    @staticmethod
    def calculate_shannon_entropy(seq: str) -> float:
        """
        Calculates the Shannon information entropy of an amino acid sequence in bits.
        Measures structural alphabet complexity (max log2(20) ~ 4.32 bits).
        Natural globular proteins typically range from 3.6 to 4.2 bits.
        Low-complexity repeats (e.g. TAREs, (TAT)n slippage) drop below 2.5 bits.
        """
        if not seq:
            return 0.0
        from collections import Counter
        counts = Counter(seq)
        n = len(seq)
        return -sum((c / n) * math.log2(c / n) for c in counts.values())

    @staticmethod
    def low_complexity_mask(seq: str, window: int = 12, threshold: float = 2.2) -> List[bool]:
        """
        Identifies local low-complexity amino-acid regions using a sliding-window Shannon entropy threshold.
        Returns a per-residue boolean list matching len(seq) where True marks low-complexity positions.
        """
        n = len(seq)
        if n == 0:
            return []
        if n < window:
            is_lc = ProteinEnergyOracle.calculate_shannon_entropy(seq) < threshold
            return [is_lc] * n

        mask = [False] * n
        counts: Dict[str, int] = {}
        for ch in seq[:window]:
            counts[ch] = counts.get(ch, 0) + 1

        inv_w = 1.0 / window
        for i in range(n - window + 1):
            h = -sum((c * inv_w) * math.log2(c * inv_w) for c in counts.values())
            if h < threshold:
                for k in range(i, i + window):
                    mask[k] = True
            if i + window < n:
                out_ch = seq[i]
                if counts[out_ch] == 1:
                    del counts[out_ch]
                else:
                    counts[out_ch] -= 1
                in_ch = seq[i + window]
                counts[in_ch] = counts.get(in_ch, 0) + 1
        return mask

    @staticmethod
    def globular_core_entropy(seq: str, window: int = 45) -> float:
        """
        Returns the maximum Shannon entropy observed across sliding windows of `window` residues.
        Captures whether a sequence with low-complexity tails/linkers still contains a complex globular core.
        """
        n = len(seq)
        if n == 0:
            return 0.0
        if n <= window:
            return ProteinEnergyOracle.calculate_shannon_entropy(seq)

        counts: Dict[str, int] = {}
        for ch in seq[:window]:
            counts[ch] = counts.get(ch, 0) + 1

        inv_w = 1.0 / window
        best_h = 0.0
        for i in range(n - window + 1):
            h = -sum((c * inv_w) * math.log2(c * inv_w) for c in counts.values())
            if h > best_h:
                best_h = h
            if i + window < n:
                out_ch = seq[i]
                if counts[out_ch] == 1:
                    del counts[out_ch]
                else:
                    counts[out_ch] -= 1
                in_ch = seq[i + window]
                counts[in_ch] = counts.get(in_ch, 0) + 1
        return float(best_h)

    def _finalize_sequence_energy(
        self,
        clean_seq: str,
        esm_loss: float,
        peak_12_nll: float,
        n_term_nll: float,
        loss_vec: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Combines ESM-2 perplexity, ORF validity, and low-complexity-aware entropy into a cached summary dict."""
        has_internal_stop = "*" in clean_seq
        ptc_cost = self.ptc_penalty if has_internal_stop else 0.0
        length = len(clean_seq)
        len_cost = float(max(0, self.min_length - length)) * self.short_penalty

        shannon_h = self.calculate_shannon_entropy(clean_seq)
        lc_mask = self.low_complexity_mask(clean_seq, window=12, threshold=2.2)
        n_lc = sum(lc_mask)
        f_lc = float(n_lc) / float(length) if length > 0 else 1.0
        non_lc_residues = length - n_lc
        core_h = self.globular_core_entropy(clean_seq, window=45)

        has_globular_core = (core_h >= 3.25) and (non_lc_residues >= 45) and (f_lc <= 0.70)
        entropy_cost = 0.0 if has_globular_core else float(max(0.0, 3.2 - shannon_h) * 8.0)

        globular_esm_loss = float(esm_loss)
        eff_esm_loss = float(esm_loss)
        if has_globular_core and loss_vec is not None and len(loss_vec) > 0:
            esm_len = int(loss_vec.shape[0])
            non_lc_indices = [idx for idx in range(esm_len) if not lc_mask[idx]]
            if len(non_lc_indices) >= 15 and n_lc > 0:
                idx_t = torch.tensor(non_lc_indices, dtype=torch.long, device=loss_vec.device)
                globular_esm_loss = float(loss_vec.index_select(0, idx_t).mean().item())
                blend = 0.5 * f_lc
                eff_esm_loss = min(
                    float(esm_loss),
                    (1.0 - blend) * float(esm_loss) + blend * globular_esm_loss,
                )

        total_energy = eff_esm_loss + ptc_cost + len_cost + entropy_cost
        return {
            "total_energy": float(total_energy),
            "esm_cross_entropy": float(esm_loss),
            "globular_esm_nll": float(globular_esm_loss),
            "peak_window_nll": float(peak_12_nll),
            "n_term_nll": float(n_term_nll),
            "shannon_entropy": float(shannon_h),
            "core_entropy": float(core_h),
            "f_lc": float(f_lc),
            "has_globular_core": bool(has_globular_core),
            "entropy_penalty": float(entropy_cost),
            "ptc_penalty": float(ptc_cost),
            "length_penalty": float(len_cost),
            "is_valid_orf": (not has_internal_stop) and (length >= self.min_length),
            "length": length,
            "oracle_schema": 2,
        }

    @torch.no_grad()
    def score_sequence(self, raw_protein_seq: str) -> Dict[str, float]:
        """
        Evaluates a single protein sequence and returns its energy breakdown.
        
        Args:
            raw_protein_seq: String of single-letter amino acid codes.
        
        Returns:
            Dict containing:
                - 'total_energy': Composite energy (lower is better).
                - 'esm_cross_entropy': Average per-residue perplexity / cross-entropy.
                - 'ptc_penalty': Premature termination codon penalty.
                - 'length_penalty': Length deficit penalty.
                - 'is_valid_orf': Boolean whether sequence contains no internal stops.
        """
        cached = self._cache.get(raw_protein_seq)
        if cached is not None and cached.get("oracle_schema") == 2:
            return cached

        clean_seq = raw_protein_seq.rstrip("*")
        
        # 1. Check for Internal Premature Termination Codons (PTCs)
        has_internal_stop = "*" in clean_seq
        ptc_cost = self.ptc_penalty if has_internal_stop else 0.0
        
        # 2. Length check
        length = len(clean_seq)
        length_deficit = max(0, self.min_length - length)
        len_cost = float(length_deficit) * self.short_penalty
        
        if length == 0:
            res = {
                "total_energy": 500.0,
                "esm_cross_entropy": 50.0,
                "globular_esm_nll": 50.0,
                "peak_window_nll": 50.0,
                "n_term_nll": 50.0,
                "shannon_entropy": 0.0,
                "core_entropy": 0.0,
                "f_lc": 1.0,
                "has_globular_core": False,
                "entropy_penalty": 0.0,
                "ptc_penalty": ptc_cost,
                "length_penalty": len_cost,
                "is_valid_orf": False,
                "length": 0,
                "oracle_schema": 2,
            }
            self._cache[raw_protein_seq] = res
            return res

        # Remove any internal asterisks and cap at ESM-2 1022-residue context limit
        esm_seq = clean_seq.replace("*", "X")[:1022]
        data = [("candidate", esm_seq)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        # 3. Compute ESM-2 Evolutionary Pseudo-Log-Likelihood & Local Windowed Profile
        loss_vec: Optional[torch.Tensor] = None
        try:
            results = self.esm_model(batch_tokens, return_contacts=False)
            logits = results["logits"]  # (1, seq_len + 2, alphabet_size)
            
            # Extract target tokens (skipping <cls> at pos 0 and <eos> at pos -1)
            target = batch_tokens[0, 1 : 1 + len(esm_seq)]
            pred_logits = logits[0, 1 : 1 + len(esm_seq), :]
            
            loss_vec = nn.CrossEntropyLoss(reduction="none")(pred_logits, target)
            esm_loss = float(loss_vec.mean().item())
            if len(esm_seq) >= 12:
                w_losses = loss_vec.unfold(0, 12, 1).mean(dim=1)
                peak_12_nll = float(w_losses.max().item())
            else:
                peak_12_nll = esm_loss
            n_term_nll = float(loss_vec[: min(15, len(esm_seq))].mean().item())
        except Exception:
            esm_loss = 25.0
            peak_12_nll = 25.0
            n_term_nll = 25.0

        res = self._finalize_sequence_energy(
            clean_seq, esm_loss, peak_12_nll, n_term_nll, loss_vec=loss_vec
        )
        self._cache[raw_protein_seq] = res
        return res

    @torch.no_grad()
    def score_masked_profile(
        self,
        raw_protein_seq: str,
        junction_aa_indices: Optional[List[int]] = None,
        disputed_aa_ranges: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict[str, float]:
        """
        Computes leakage-free Masked Language Model NLL over splice-junction collars
        (+-4 AA around each splice boundary) and any disputed sequence regions (e.g.
        retained introns or N-terminal extensions) in a single 2-row interleaved forward pass.
        """
        clean_seq = raw_protein_seq.rstrip("*")
        esm_seq = clean_seq.replace("*", "X")[:1022]
        L = len(esm_seq)
        if L == 0:
            return {
                "masked_junc_nll": 25.0,
                "masked_peak_nll": 25.0,
            }

        target_positions = set()
        if junction_aa_indices:
            for j in junction_aa_indices:
                for pos in range(max(0, j - 4), min(L, j + 4)):
                    target_positions.add(pos)
        if disputed_aa_ranges:
            for r_s, r_e in disputed_aa_ranges:
                for pos in range(max(0, r_s), min(L, r_e)):
                    target_positions.add(pos)

        if not target_positions:
            base = self.score_sequence(raw_protein_seq)
            return {
                "masked_junc_nll": base["esm_cross_entropy"],
                "masked_peak_nll": base.get("peak_window_nll", base["esm_cross_entropy"]),
            }

        pos_list = sorted(target_positions)
        cache_key = (esm_seq, tuple(pos_list))
        if not hasattr(self, "_collar_cache"):
            self._collar_cache: Dict[Tuple[str, Tuple[int, ...]], Dict[str, float]] = {}
        if cache_key in self._collar_cache:
            return self._collar_cache[cache_key]

        try:
            _, _, tok = self.batch_converter([("p", esm_seq)])
            tok = tok.to(self.device)
            mask_idx = self.alphabet.mask_idx
            # 2-row interleaved mask so adjacent residues inside the collar remain unmasked context
            batch = tok.repeat(2, 1)
            for idx_p, pos in enumerate(pos_list):
                batch[idx_p % 2, 1 + pos] = mask_idx
            logits = self.esm_model(batch, return_contacts=False)["logits"]
            loss_fn = nn.CrossEntropyLoss(reduction="none")
            vals = []
            for idx_p, pos in enumerate(pos_list):
                r = idx_p % 2
                v = float(loss_fn(logits[r, 1 + pos : 2 + pos, :], tok[0, 1 + pos : 2 + pos]).item())
                vals.append(v)
            mean_collar = sum(vals) / len(vals)
            if len(vals) >= 6:
                peak_collar = max(sum(vals[k : k + 6]) / 6.0 for k in range(len(vals) - 5))
            else:
                peak_collar = max(vals)
            res = {
                "masked_junc_nll": float(mean_collar),
                "masked_peak_nll": float(peak_collar),
            }
        except Exception:
            res = {"masked_junc_nll": 3.5, "masked_peak_nll": 4.0}

        self._collar_cache[cache_key] = res
        return res

    @torch.no_grad()
    def score_masked_profiles_batch(
        self,
        requests: List[Tuple[str, Optional[List[int]], Optional[List[Tuple[int, int]]]]],
        batch_size: int = 8,
    ) -> List[Dict[str, float]]:
        """
        Pre-warms `_collar_cache` using length-sorted batched 2-row interleaved ESM-2
        forward passes, then returns the exact `score_masked_profile` results.
        """
        if not hasattr(self, "_collar_cache"):
            self._collar_cache = {}

        uncached = []
        seen_keys = set()
        for raw_protein_seq, junction_aa_indices, disputed_aa_ranges in requests:
            clean_seq = raw_protein_seq.rstrip("*")
            esm_seq = clean_seq.replace("*", "X")[:1022]
            L = len(esm_seq)
            if L == 0:
                continue
            target_positions = set()
            if junction_aa_indices:
                for j in junction_aa_indices:
                    for pos in range(max(0, j - 4), min(L, j + 4)):
                        target_positions.add(pos)
            if disputed_aa_ranges:
                for r_s, r_e in disputed_aa_ranges:
                    for pos in range(max(0, r_s), min(L, r_e)):
                        target_positions.add(pos)
            if not target_positions:
                continue
            pos_tuple = tuple(sorted(target_positions))
            cache_key = (esm_seq, pos_tuple)
            if cache_key not in self._collar_cache and cache_key not in seen_keys:
                seen_keys.add(cache_key)
                uncached.append((esm_seq, pos_tuple, cache_key))

        if uncached:
            uncached.sort(key=lambda x: len(x[0]))
            mask_idx = self.alphabet.mask_idx
            loss_fn = nn.CrossEntropyLoss(reduction="none")

            for i in range(0, len(uncached), batch_size):
                chunk = uncached[i : i + batch_size]
                data = [(f"m_{j}", item[0]) for j, item in enumerate(chunk)]
                try:
                    _, _, tok_batch = self.batch_converter(data)
                    tok_batch = tok_batch.to(self.device)
                    batch = tok_batch.repeat_interleave(2, dim=0)

                    row_indices = []
                    col_indices = []
                    target_toks = []
                    item_slices = []
                    cur_off = 0

                    for j, (_, pos_tuple, _) in enumerate(chunk):
                        base_r = 2 * j
                        for idx_p, pos in enumerate(pos_tuple):
                            r = base_r + (idx_p % 2)
                            c = 1 + pos
                            batch[r, c] = mask_idx
                            row_indices.append(r)
                            col_indices.append(c)
                            target_toks.append(tok_batch[j, c])
                        next_off = cur_off + len(pos_tuple)
                        item_slices.append((cur_off, next_off))
                        cur_off = next_off

                    logits = self.esm_model(batch, return_contacts=False)["logits"]
                    r_t = torch.tensor(row_indices, dtype=torch.long, device=self.device)
                    c_t = torch.tensor(col_indices, dtype=torch.long, device=self.device)
                    tgt_t = torch.stack(target_toks)
                    all_losses = loss_fn(logits[r_t, c_t, :], tgt_t).detach().cpu().tolist()

                    for j, (_, _, cache_key) in enumerate(chunk):
                        s_i, e_i = item_slices[j]
                        vals = all_losses[s_i:e_i]
                        mean_collar = sum(vals) / len(vals)
                        if len(vals) >= 6:
                            peak_collar = max(sum(vals[k : k + 6]) / 6.0 for k in range(len(vals) - 5))
                        else:
                            peak_collar = max(vals)
                        self._collar_cache[cache_key] = {
                            "masked_junc_nll": float(mean_collar),
                            "masked_peak_nll": float(peak_collar),
                        }
                except Exception:
                    # Individual calls will populate cache lazily via fallback
                    pass

        return [
            self.score_masked_profile(raw_seq, j_aa, d_rng)
            for raw_seq, j_aa, d_rng in requests
        ]

    @torch.no_grad()
    def score_transcript(self, tx) -> Dict[str, float]:
        """
        Evaluates a complete SplicedTranscript object, taking into account
        triplet reading frame divisibility, start/stop validity, Shannon complexity,
        and ESM-2 perplexity.
        """
        base_scores = self.score_sequence(tx.protein_sequence)
        
        frame_cost = 100.0 if not tx.is_in_frame else 0.0
        start_cost = 50.0 if not tx.has_valid_start else 0.0
        stop_cost = 50.0 if not tx.has_valid_stop else 0.0
        internal_stop_cost = self.ptc_penalty if tx.has_internal_stop else 0.0
        entropy_cost = base_scores.get("entropy_penalty", 0.0)
        
        total_energy = (
            base_scores["esm_cross_entropy"]
            + frame_cost
            + start_cost
            + stop_cost
            + internal_stop_cost
            + base_scores["length_penalty"]
            + entropy_cost
        )
        
        return {
            "total_energy": float(total_energy),
            "esm_cross_entropy": base_scores["esm_cross_entropy"],
            "peak_window_nll": base_scores.get("peak_window_nll", base_scores["esm_cross_entropy"]),
            "n_term_nll": base_scores.get("n_term_nll", base_scores["esm_cross_entropy"]),
            "entropy_penalty": float(entropy_cost),
            "frame_penalty": frame_cost,
            "start_penalty": start_cost,
            "stop_penalty": stop_cost,
            "ptc_penalty": internal_stop_cost,
            "length_penalty": base_scores["length_penalty"],
            "is_valid_orf": tx.is_in_frame and tx.has_valid_start and tx.has_valid_stop and (not tx.has_internal_stop),
            "length": base_scores["length"],
        }

    @torch.no_grad()
    def score_batch(self, sequences: List[str], batch_size: int = 16) -> List[Dict[str, float]]:
        """
        Scores multiple candidate protein sequences using length-sorted mini-batches
        to minimize padding overhead and maximize SIMD/GPU throughput.
        """
        uncached = []
        seen = set()
        for seq in sequences:
            cached = self._cache.get(seq)
            if (cached is None or cached.get("oracle_schema") != 2) and seq not in seen:
                seen.add(seq)
                clean = seq.rstrip("*")
                if len(clean) == 0:
                    self.score_sequence(seq)
                else:
                    uncached.append((seq, clean, clean.replace("*", "X")[:1022]))

        # Sort by ESM-2 sequence length to minimize batch padding waste
        uncached.sort(key=lambda x: len(x[2]))
        loss_fn = nn.CrossEntropyLoss(reduction="none")

        for i in range(0, len(uncached), batch_size):
            chunk = uncached[i : i + batch_size]
            data = [(f"cand_{j}", item[2]) for j, item in enumerate(chunk)]
            try:
                _, _, batch_tokens = self.batch_converter(data)
                batch_tokens = batch_tokens.to(self.device)
                results = self.esm_model(batch_tokens, return_contacts=False)
                logits = results["logits"]  # (B, max_L + 2, V)

                for j, (raw_seq, clean_seq, esm_seq) in enumerate(chunk):
                    seq_len = len(esm_seq)
                    target_j = batch_tokens[j, 1 : 1 + seq_len]
                    logits_j = logits[j, 1 : 1 + seq_len, :]
                    loss_vec = loss_fn(logits_j, target_j)
                    esm_loss = float(loss_vec.mean().item())
                    if seq_len >= 12:
                        peak_12_nll = float(loss_vec.unfold(0, 12, 1).mean(dim=1).max().item())
                    else:
                        peak_12_nll = esm_loss
                    n_term_nll = float(loss_vec[: min(15, seq_len)].mean().item())

                    self._cache[raw_seq] = self._finalize_sequence_energy(
                        clean_seq, esm_loss, peak_12_nll, n_term_nll, loss_vec=loss_vec
                    )
            except Exception:
                # Fallback to single-sequence evaluation on error
                for raw_seq, _, _ in chunk:
                    self.score_sequence(raw_seq)

        return [self.score_sequence(seq) for seq in sequences]

    @torch.no_grad()
    def score_transcripts_batch(self, transcripts: List, batch_size: int = 16) -> List[Dict[str, float]]:
        """Pre-warms cache with batched ESM-2 inference and returns transcript energy dicts."""
        self.score_batch([tx.protein_sequence for tx in transcripts], batch_size=batch_size)
        return [self.score_transcript(tx) for tx in transcripts]


