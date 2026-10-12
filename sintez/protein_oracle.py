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
        if raw_protein_seq in self._cache:
            return self._cache[raw_protein_seq]

        clean_seq = raw_protein_seq.rstrip("*")
        
        # 1. Check for Internal Premature Termination Codons (PTCs)
        has_internal_stop = "*" in clean_seq
        ptc_cost = self.ptc_penalty if has_internal_stop else 0.0
        
        # 2. Length check
        length = len(clean_seq)
        length_deficit = max(0, self.min_length - length)
        len_cost = float(length_deficit) * self.short_penalty
        
        if length == 0:
            return {
                "total_energy": 500.0,
                "esm_cross_entropy": 50.0,
                "ptc_penalty": ptc_cost,
                "length_penalty": len_cost,
                "is_valid_orf": False,
                "length": 0,
            }

        # Remove any internal asterisks before passing to ESM (which expects standard AAs)
        esm_seq = clean_seq.replace("*", "X")
        data = [("candidate", esm_seq)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        # 3. Compute ESM-2 Evolutionary Pseudo-Log-Likelihood
        try:
            results = self.esm_model(batch_tokens, return_contacts=False)
            logits = results["logits"]  # (1, seq_len + 2, alphabet_size)
            
            # Extract target tokens (skipping <cls> at pos 0 and <eos> at pos -1)
            target = batch_tokens[:, 1:-1]
            pred_logits = logits[:, 1:-1, :]
            
            loss_fn = nn.CrossEntropyLoss(reduction="mean")
            esm_loss = loss_fn(pred_logits.reshape(-1, pred_logits.size(-1)), target.reshape(-1)).item()
        except Exception as e:
            # Fallback for unexpected edge cases
            esm_loss = 25.0

        # 3. Information-Theoretic Sequence Complexity
        shannon_h = self.calculate_shannon_entropy(clean_seq)
        complexity_deficit = max(0.0, 3.2 - shannon_h)
        entropy_cost = float(complexity_deficit * 8.0)

        total_energy = esm_loss + ptc_cost + len_cost + entropy_cost

        res = {
            "total_energy": float(total_energy),
            "esm_cross_entropy": float(esm_loss),
            "shannon_entropy": float(shannon_h),
            "entropy_penalty": float(entropy_cost),
            "ptc_penalty": float(ptc_cost),
            "length_penalty": float(len_cost),
            "is_valid_orf": (not has_internal_stop) and (length >= self.min_length),
            "length": length,
        }
        self._cache[raw_protein_seq] = res
        return res

    @torch.no_grad()
    def score_transcript(self, tx) -> Dict[str, float]:
        """
        Evaluates a complete SplicedTranscript object, taking into account
        triplet reading frame divisibility, start/stop validity, and ESM-2 perplexity.
        """
        base_scores = self.score_sequence(tx.protein_sequence)
        
        frame_cost = 100.0 if not tx.is_in_frame else 0.0
        start_cost = 50.0 if not tx.has_valid_start else 0.0
        stop_cost = 50.0 if not tx.has_valid_stop else 0.0
        internal_stop_cost = self.ptc_penalty if tx.has_internal_stop else 0.0
        
        total_energy = (
            base_scores["esm_cross_entropy"]
            + frame_cost
            + start_cost
            + stop_cost
            + internal_stop_cost
            + base_scores["length_penalty"]
        )
        
        return {
            "total_energy": float(total_energy),
            "esm_cross_entropy": base_scores["esm_cross_entropy"],
            "frame_penalty": frame_cost,
            "start_penalty": start_cost,
            "stop_penalty": stop_cost,
            "ptc_penalty": internal_stop_cost,
            "length_penalty": base_scores["length_penalty"],
            "is_valid_orf": tx.is_in_frame and tx.has_valid_start and tx.has_valid_stop and (not tx.has_internal_stop),
            "length": base_scores["length"],
        }

    def score_batch(self, sequences: List[str]) -> List[Dict[str, float]]:
        """Score multiple candidate protein sequences."""
        return [self.score_sequence(seq) for seq in sequences]

