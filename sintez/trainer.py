"""
Counterfactual Contrastive Training Engine
==========================================
Trains the composition-invariant DNA backbone using margin-based contrastive
protein energy objectives paired with sparse junction boundary losses.

Features:
- Differentiable dual-space path-ranking loss with ESM-2 foldability margin
- Alpha-balanced Focal Loss for extreme splice junction class imbalance
- 7-class frame/phase-aware exon segmentation targets
- GPU/MPS batched collation with sequence padding
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .dna_backbone import CompositionNormalizedBackbone
from .translation_operator import SplicedTranslationOperator, SplicedTranscript
from .protein_oracle import ProteinEnergyOracle


@dataclass
class GenomicGeneSample:
    """A training sample consisting of genomic DNA and its true exon intervals."""
    dna_sequence: str
    exon_spans: List[Tuple[int, int]]
    strand: str = "+"
    gene_id: str = "gene_1"


class FocalLoss(nn.Module):
    """
    Focal Loss for binary junction prediction (donor, acceptor, start, stop)
    to mitigate extreme class imbalance (~1:500).
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean", eps: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.clamp(prob, self.eps, 1.0 - self.eps)
        bce = -(target * torch.log(p) + (1.0 - target) * torch.log(1.0 - p))
        pt = p * target + (1.0 - p) * (1.0 - target)
        w = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        focal_weight = w * torch.pow(1.0 - pt, self.gamma)
        loss = focal_weight * bce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def apply_gaussian_smoothing(tensor: torch.Tensor, center: int, sigma: float = 1.0, radius: int = 2):
    """Applies a smooth Gaussian window around junction/codon coordinates."""
    L = tensor.size(0)
    for offset in range(-radius, radius + 1):
        pos = center + offset
        if 0 <= pos < L:
            weight = math.exp(-0.5 * (offset / sigma) ** 2)
            tensor[pos] = max(tensor[pos].item(), weight)


class CounterfactualDataset(Dataset):
    """
    Dataset that pairs genomic sequences with true exon structures and
    synthesizes on-the-fly hard counterfactual frame perturbations.
    """
    def __init__(self, samples: List[GenomicGeneSample], generate_negatives: bool = False):
        self.samples = samples
        self.operator = SplicedTranslationOperator()
        self.generate_negatives = generate_negatives

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        dna_str = sample.dna_sequence
        spans = list(sample.exon_spans)
        strand = sample.strand

        # Canonicalize to 5' -> 3' transcription orientation
        if strand == "-":
            L = len(dna_str)
            dna_str = self.operator.code.reverse_complement(dna_str)
            spans = [(L - e, L - s) for s, e in reversed(spans)]
            strand = "+"

        # 1. Process True Positive Transcript
        true_transcript = self.operator.process_transcript(dna_str, spans, strand="+")

        # 2. Synthesize Hard Counterfactual Negatives (if requested)
        negatives = self.operator.generate_counterfactual_negatives(dna_str, true_transcript) if self.generate_negatives else []

        # 3. Create Ground Truth Target Masks for DNA Backbone
        L = len(dna_str)
        target_donor = torch.zeros(L, dtype=torch.float32)
        target_acceptor = torch.zeros(L, dtype=torch.float32)
        target_start = torch.zeros(L, dtype=torch.float32)
        target_stop = torch.zeros(L, dtype=torch.float32)
        target_span = torch.zeros(L, dtype=torch.long)          # 0: Intergenic, 1: Intron, 2: Exon
        target_span_7class = torch.zeros(L, dtype=torch.long)   # 0: Inter, 1: Intron, 2-4: Phase 0/1/2, 5: Start, 6: Stop

        # Mark introns and splice junctions with Gaussian soft-window targets
        for i in range(len(spans) - 1):
            intron_start = spans[i][1]
            intron_end = spans[i+1][0]
            target_span[intron_start:intron_end] = 1
            target_span_7class[intron_start:intron_end] = 1
            apply_gaussian_smoothing(target_donor, intron_start, sigma=1.0, radius=2)
            apply_gaussian_smoothing(target_acceptor, intron_end - 1, sigma=1.0, radius=2)

        # Mark exons and phases
        cum_cds = 0
        for s, e in spans:
            target_span[s:e] = 2
            for pos in range(s, e):
                phase = cum_cds % 3
                target_span_7class[pos] = 2 + phase
                cum_cds += 1

        # Mark start and stop codons with Gaussian soft-window targets
        if spans:
            s_first = spans[0][0]
            e_last = spans[-1][1]
            apply_gaussian_smoothing(target_start, s_first, sigma=1.0, radius=2)
            apply_gaussian_smoothing(target_stop, e_last - 3, sigma=1.0, radius=2)

            # 7-class start/stop regions
            if 0 <= s_first and s_first + 3 <= L:
                target_span_7class[s_first:s_first + 3] = 5
            if 0 <= e_last - 3 and e_last <= L:
                target_span_7class[e_last - 3:e_last] = 6

        one_hot_dna = CompositionNormalizedBackbone.sequence_to_one_hot(dna_str).squeeze(0)

        return {
            "dna_str": dna_str,
            "one_hot_dna": one_hot_dna,
            "target_donor": target_donor,
            "target_acceptor": target_acceptor,
            "target_start": target_start,
            "target_stop": target_stop,
            "target_span": target_span,
            "target_span_7class": target_span_7class,
            "true_transcript": true_transcript,
            "negatives": negatives,
        }


def collate_genomic_batch(batch: List[Dict]) -> Dict:
    """Collates variable-length genomic samples into a padded batch for GPU/MPS."""
    max_len = max(b["target_span"].shape[0] for b in batch)
    B = len(batch)

    padded_one_hot = torch.zeros(B, 4, max_len, dtype=torch.float32)
    padded_one_hot[:, :, :] = 0.25  # Uniform background for padding

    padded_donor = torch.zeros(B, max_len, dtype=torch.float32)
    padded_acceptor = torch.zeros(B, max_len, dtype=torch.float32)
    padded_start = torch.zeros(B, max_len, dtype=torch.float32)
    padded_stop = torch.zeros(B, max_len, dtype=torch.float32)
    padded_span = torch.zeros(B, max_len, dtype=torch.long)
    padded_span_7class = torch.zeros(B, max_len, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.bool)

    for i, b in enumerate(batch):
        L = b["target_span"].shape[0]
        padded_one_hot[i, :, :L] = b["one_hot_dna"]
        padded_donor[i, :L] = b["target_donor"]
        padded_acceptor[i, :L] = b["target_acceptor"]
        padded_start[i, :L] = b["target_start"]
        padded_stop[i, :L] = b["target_stop"]
        padded_span[i, :L] = b["target_span"]
        padded_span_7class[i, :L] = b["target_span_7class"]
        mask[i, :L] = True

    return {
        "one_hot_dna": padded_one_hot,
        "target_donor": padded_donor,
        "target_acceptor": padded_acceptor,
        "target_start": padded_start,
        "target_stop": padded_stop,
        "target_span": padded_span,
        "target_span_7class": padded_span_7class,
        "mask": mask,
        "samples": batch,
    }


class ContrastiveEnergyTrainer:
    """
    Orchestrates the training of the DNA backbone using the Dual-Space
    Translation-Energy objective.
    """
    def __init__(
        self,
        backbone: CompositionNormalizedBackbone,
        oracle: Optional[ProteinEnergyOracle] = None,
        learning_rate: float = 5e-4,
        margin: float = 10.0,
        lambda_junc: float = 3.0,
        lambda_span: float = 1.0,
        lambda_contrastive: float = 0.50,
        loss_type: str = "focal",
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        use_phase_aware_span: bool = False,
    ):
        self.backbone = backbone
        self.oracle = oracle
        self.margin = margin
        self.lambda_junc = lambda_junc
        self.lambda_span = lambda_span
        self.lambda_contrastive = lambda_contrastive
        self.loss_type = loss_type
        self.use_phase_aware_span = use_phase_aware_span
        self.operator = SplicedTranslationOperator()

        self.optimizer = torch.optim.AdamW(self.backbone.parameters(), lr=learning_rate, weight_decay=1e-4)

        if loss_type == "focal":
            self.junc_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.junc_loss_fn = nn.BCELoss()

        self.span_loss_fn = nn.CrossEntropyLoss()

    def compute_transcript_log_prob(
        self,
        preds: Dict[str, torch.Tensor],
        spans: List[Tuple[int, int]],
        L: int,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Differentiably scores a candidate transcript path using backbone output logits/probabilities.
        Returns a scalar torch.Tensor with non-zero gradients connecting to backbone weights.
        """
        if not spans:
            return torch.tensor(-100.0, device=preds["donor_prob"].device)

        device = preds["donor_prob"].device
        log_prob = torch.tensor(0.0, device=device)

        # 1. Start codon
        s0 = spans[0][0]
        if 0 <= s0 < L:
            log_prob = log_prob + torch.log(preds["start_prob"].squeeze()[s0] + eps)

        # 2. Stop codon
        e_last = spans[-1][1]
        if 0 <= e_last - 3 < L:
            log_prob = log_prob + torch.log(preds["stop_prob"].squeeze()[e_last - 3] + eps)

        # 3. Splice junctions
        for i in range(len(spans) - 1):
            donor_pos = spans[i][1]
            acceptor_pos = spans[i+1][0] - 1
            if 0 <= donor_pos < L:
                log_prob = log_prob + torch.log(preds["donor_prob"].squeeze()[donor_pos] + eps)
            if 0 <= acceptor_pos < L:
                log_prob = log_prob + torch.log(preds["acceptor_prob"].squeeze()[acceptor_pos] + eps)

        # 4. Exon span coding probability
        exon_indices = []
        for s, e in spans:
            exon_indices.extend(range(max(0, s), min(L, e)))

        if exon_indices:
            idx_tensor = torch.tensor(exon_indices, dtype=torch.long, device=device)
            # Class 2 is Exon (or sum of Phase 0, 1, 2 in 7-class mode)
            exon_probs = preds["span_probs"].squeeze(0)[2, idx_tensor]
            mean_exon_log_prob = torch.log(exon_probs + eps).mean()
            log_prob = log_prob + mean_exon_log_prob

        return log_prob

    def train_step(self, batch: Dict, compute_energy: bool = False) -> Dict[str, float]:
        """Executes a single contrastive optimization step on one sample."""
        self.backbone.train()
        self.optimizer.zero_grad()

        one_hot = batch["one_hot_dna"].unsqueeze(0)
        device = next(self.backbone.parameters()).device
        one_hot = one_hot.to(device)

        t_donor = batch["target_donor"].to(device)
        t_acceptor = batch["target_acceptor"].to(device)
        t_start = batch["target_start"].to(device)
        t_stop = batch["target_stop"].to(device)

        if self.use_phase_aware_span and "target_span_7class" in batch:
            t_span = batch["target_span_7class"].to(device)
        else:
            t_span = batch["target_span"].to(device)

        # 1. Forward Pass
        preds = self.backbone(one_hot)
        L = one_hot.shape[-1]

        # 2. Junction Boundary Losses (Focal or BCE)
        loss_donor = self.junc_loss_fn(preds["donor_prob"].squeeze(), t_donor)
        loss_acceptor = self.junc_loss_fn(preds["acceptor_prob"].squeeze(), t_acceptor)
        loss_start = self.junc_loss_fn(preds["start_prob"].squeeze(), t_start)
        loss_stop = self.junc_loss_fn(preds["stop_prob"].squeeze(), t_stop)
        loss_junctions = loss_donor + loss_acceptor + loss_start + loss_stop

        # 3. Span Segmentation Loss
        if self.use_phase_aware_span and "span_probs_raw" in preds:
            loss_span = self.span_loss_fn(preds["span_logits"].squeeze(0).permute(1, 0), t_span)
        else:
            loss_span = self.span_loss_fn(preds["span_probs"].squeeze(0).permute(1, 0), t_span)

        # 4. Differentiable Dual-Space Path-Ranking Loss with Oracle Energy Margin
        contrastive_loss = torch.tensor(0.0, device=device)
        contrastive_cost = 0.0
        e_pos = 0.0
        is_valid = True

        if compute_energy and self.oracle is not None:
            true_tx = batch["true_transcript"]
            e_pos_dict = self.oracle.score_sequence(true_tx.protein_sequence)
            e_pos = e_pos_dict["total_energy"]
            is_valid = e_pos_dict["is_valid_orf"]

            # Differentiable positive path score from backbone
            score_pos = self.compute_transcript_log_prob(preds, true_tx.exon_spans, L)

            negatives = batch.get("negatives")
            if not negatives:
                negatives = self.operator.generate_counterfactual_negatives(batch["dna_str"], true_tx)

            ranking_losses = []
            for neg_name, neg_tx in negatives:
                e_neg_dict = self.oracle.score_sequence(neg_tx.protein_sequence)
                e_neg = e_neg_dict["total_energy"]

                # Energy margin: larger foldability discrepancy demands larger score gap
                energy_margin = max(self.margin, (e_pos - e_neg) + self.margin)

                # Differentiable negative path score from backbone
                score_neg = self.compute_transcript_log_prob(preds, neg_tx.exon_spans, L)

                # Ranking hinge loss: score_neg should be less than score_pos - energy_margin
                # i.e., score_neg - score_pos + energy_margin <= 0
                violation = F.relu(score_neg - score_pos + energy_margin)
                ranking_losses.append(violation)

            if ranking_losses:
                contrastive_loss = torch.stack(ranking_losses).mean()
                contrastive_cost = contrastive_loss.item()

        # 5. Composite Objective (Fully Differentiable)
        total_loss = (
            self.lambda_junc * loss_junctions
            + self.lambda_span * loss_span
            + self.lambda_contrastive * contrastive_loss
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), max_norm=1.0)
        self.optimizer.step()

        if device.type == "mps":
            torch.mps.empty_cache()

        return {
            "total_loss": total_loss.item(),
            "loss_junctions": loss_junctions.item(),
            "loss_span": loss_span.item(),
            "contrastive_loss": contrastive_cost,
            "e_pos": e_pos,
            "is_true_valid": is_valid,
        }
