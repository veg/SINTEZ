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
    """
    A training sample consisting of genomic DNA and its true exon intervals.
    Supports both single-gene windows (`exon_spans`) and multi-gene contiguous genomic windows
    (`multi_gene_spans`: list of exon-span lists for all forward-strand genes in the window).
    """
    dna_sequence: str
    exon_spans: List[Tuple[int, int]]
    strand: str = "+"
    gene_id: str = "gene_1"
    multi_gene_spans: Optional[List[List[Tuple[int, int]]]] = None


class FocalLoss(nn.Module):
    """
    Focal Loss for binary junction prediction (donor, acceptor, start, stop)
    to mitigate extreme class imbalance (~1:500).
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Supports:
    - `mask`: optional float/bool tensor of shape matching `prob` to zero out padded positions
      without changing tensor shapes (preventing MPSGraph/XLA backward-pass recompilation).
    - `"pos_norm"` reduction (RetinaNet/FCOS style) which normalizes total focal loss
      by the number of positive boundary sites (plus a small background floor) rather than sequence
      length L, preventing gradient dilution on long 15-25 kb windows.
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "pos_norm", eps: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        prob: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        p = torch.clamp(prob, self.eps, 1.0 - self.eps)
        t = torch.clamp(target, 0.0, 1.0)
        bce = -(t * torch.log(p) + (1.0 - t) * torch.log(1.0 - p))
        pt = p * t + (1.0 - p) * (1.0 - t)
        w = self.alpha * t + (1.0 - self.alpha) * (1.0 - t)
        focal_weight = w * torch.pow(1.0 - pt, self.gamma)
        loss = focal_weight * bce

        if mask is not None:
            m = mask.to(loss.dtype)
            loss = loss * m
            t_masked = t * m
        else:
            t_masked = t

        if self.reduction == "pos_norm":
            batch_scale = float(prob.shape[0]) if prob.ndim >= 2 else 1.0
            pos_mass = torch.clamp(t_masked.sum(), min=10.0 * batch_scale)
            return loss.sum() / (pos_mass * 20.0)
        elif self.reduction == "mean":
            if mask is not None:
                return loss.sum() / torch.clamp(mask.to(loss.dtype).sum(), min=1.0)
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
    Dataset that pairs genomic sequences with true exon structures (single or multi-gene windows)
    and synthesizes on-the-fly hard counterfactual frame perturbations.
    """
    def __init__(
        self,
        samples: List[GenomicGeneSample],
        generate_negatives: bool = False,
        bucket_step: Optional[int] = None,
    ):
        self.samples = samples
        self.operator = SplicedTranslationOperator()
        self.generate_negatives = generate_negatives
        self.bucket_step = bucket_step

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _compute_codon_foldability_track(dna_str: str, all_genes_spans: List[List[Tuple[int, int]]]) -> torch.Tensor:
        """
        Computes a per-base biophysical coding quality / foldability target in [0, 1]:
        - 0.0 in non-coding introns, intergenic DNA, and antisense shadow ORFs
        - In true coding exons: scaled by local 15-aa window Shannon sequence complexity
          (H / 4.0 bits, clamped to [0.5, 1.0]) so globular coding domains receive ~0.95-1.0
          and low-complexity repeats receive lower foldability targets.
        """
        L = len(dna_str)
        target_fold = torch.zeros(L, dtype=torch.float32)
        for spans in all_genes_spans:
            if not spans:
                continue
            spliced_dna = "".join(dna_str[s:e] for s, e in spans if 0 <= s < e <= L)
            if len(spliced_dna) < 9:
                continue
            for s, e in spans:
                s_c = max(0, s)
                e_c = min(L, e)
                if e_c <= s_c:
                    continue
                sub = dna_str[s_c:e_c].upper()
                n = len(sub)
                counts = [sub.count(b) for b in "ACGT"]
                ent = -sum((c / n) * math.log2(c / n) for c in counts if c > 0)
                score = min(1.0, max(0.60, 0.50 + 0.25 * ent))
                target_fold[s_c:e_c] = score
        return target_fold

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        dna_str = sample.dna_sequence
        spans = list(sample.exon_spans)
        strand = sample.strand
        L = len(dna_str)

        # Canonicalize to 5' -> 3' transcription orientation
        if strand == "-":
            dna_str = self.operator.code.reverse_complement(dna_str)
            spans = [(L - e, L - s) for s, e in reversed(spans)]
            if sample.multi_gene_spans:
                all_genes_spans = [
                    [(L - e, L - s) for s, e in reversed(g_spans)]
                    for g_spans in sample.multi_gene_spans
                ]
            else:
                all_genes_spans = [spans] if spans else []
            strand = "+"
        else:
            if sample.multi_gene_spans is not None:
                all_genes_spans = [list(g_spans) for g_spans in sample.multi_gene_spans if g_spans]
            else:
                all_genes_spans = [spans] if spans else []

        # 1. Process Primary True Positive Transcript
        true_transcript = self.operator.process_transcript(dna_str, spans, strand="+")

        # 2. Synthesize Hard Counterfactual Negatives (if requested)
        negatives = self.operator.generate_counterfactual_negatives(dna_str, true_transcript) if self.generate_negatives else []

        # 3. Create Ground Truth Target Masks for DNA Backbone across ALL forward-strand genes in window
        target_donor = torch.zeros(L, dtype=torch.float32)
        target_acceptor = torch.zeros(L, dtype=torch.float32)
        target_start = torch.zeros(L, dtype=torch.float32)
        target_stop = torch.zeros(L, dtype=torch.float32)
        target_span = torch.zeros(L, dtype=torch.long)          # 0: Intergenic, 1: Intron, 2: Exon
        target_span_7class = torch.zeros(L, dtype=torch.long)   # 0: Inter, 1: Intron, 2-4: Phase 0/1/2, 5: Start, 6: Stop

        for g_spans in all_genes_spans:
            if not g_spans:
                continue
            for i in range(len(g_spans) - 1):
                intron_start = g_spans[i][1]
                intron_end = g_spans[i + 1][0]
                if 0 <= intron_start < intron_end <= L:
                    target_span[intron_start:intron_end] = 1
                    target_span_7class[intron_start:intron_end] = 1
                apply_gaussian_smoothing(target_donor, intron_start, sigma=1.0, radius=2)
                apply_gaussian_smoothing(target_acceptor, intron_end - 1, sigma=1.0, radius=2)

            cum_cds = 0
            for s, e in g_spans:
                s_c = max(0, s)
                e_c = min(L, e)
                target_span[s_c:e_c] = 2
                for pos in range(s, e):
                    if 0 <= pos < L:
                        phase = cum_cds % 3
                        target_span_7class[pos] = 2 + phase
                    cum_cds += 1

            s_first = g_spans[0][0]
            e_last = g_spans[-1][1]
            apply_gaussian_smoothing(target_start, s_first, sigma=1.0, radius=2)
            apply_gaussian_smoothing(target_stop, e_last - 3, sigma=1.0, radius=2)

            if 0 <= s_first and s_first + 3 <= L:
                target_span_7class[s_first : s_first + 3] = 5
            if 0 <= e_last - 3 and e_last <= L:
                target_span_7class[e_last - 3 : e_last] = 6

        target_foldability = self._compute_codon_foldability_track(dna_str, all_genes_spans)

        one_hot_dna = CompositionNormalizedBackbone.sequence_to_one_hot(dna_str).squeeze(0)

        valid_len = L
        if self.bucket_step is not None and self.bucket_step > 0:
            rem = L % self.bucket_step
            pad_len = (self.bucket_step - rem) if rem != 0 else 0
            if pad_len > 0:
                one_hot_padded = torch.full((4, L + pad_len), 0.25, dtype=torch.float32)
                one_hot_padded[:, :L] = one_hot_dna
                one_hot_dna = one_hot_padded

        return {
            "dna_str": dna_str,
            "one_hot_dna": one_hot_dna,
            "target_donor": target_donor,
            "target_acceptor": target_acceptor,
            "target_start": target_start,
            "target_stop": target_stop,
            "target_span": target_span,
            "target_span_7class": target_span_7class,
            "target_foldability": target_foldability,
            "true_transcript": true_transcript,
            "negatives": negatives,
            "valid_len": valid_len,
        }


def collate_genomic_batch(batch: List[Dict], bucket_step: Optional[int] = 256) -> Dict:
    """
    Collates variable-length genomic samples into a static-bucket padded batch `(B, ..., max_len)`
    for GPU/MPS/TPU execution. Pads `target_span` and `target_span_7class` with `-100`
    (`ignore_index=-100`) and populates `mask` of shape `(B, max_len)` so both forward and
    backward passes execute strictly on bucket-aligned tensor shapes without dynamic slicing.
    """
    raw_max = max(b["one_hot_dna"].shape[-1] for b in batch)
    if bucket_step is not None and bucket_step > 0:
        rem = raw_max % bucket_step
        max_len = raw_max + ((bucket_step - rem) if rem != 0 else 0)
    else:
        max_len = raw_max

    B = len(batch)

    padded_one_hot = torch.full((B, 4, max_len), 0.25, dtype=torch.float32)
    padded_donor = torch.zeros(B, max_len, dtype=torch.float32)
    padded_acceptor = torch.zeros(B, max_len, dtype=torch.float32)
    padded_start = torch.zeros(B, max_len, dtype=torch.float32)
    padded_stop = torch.zeros(B, max_len, dtype=torch.float32)
    padded_span = torch.full((B, max_len), -100, dtype=torch.long)
    padded_span_7class = torch.full((B, max_len), -100, dtype=torch.long)
    padded_foldability = torch.zeros(B, max_len, dtype=torch.float32)
    mask = torch.zeros(B, max_len, dtype=torch.bool)

    for i, b in enumerate(batch):
        L = b["target_span"].shape[0]
        padded_one_hot[i, :, :L] = b["one_hot_dna"][:, :L]
        padded_donor[i, :L] = b["target_donor"]
        padded_acceptor[i, :L] = b["target_acceptor"]
        padded_start[i, :L] = b["target_start"]
        padded_stop[i, :L] = b["target_stop"]
        padded_span[i, :L] = b["target_span"]
        padded_span_7class[i, :L] = b["target_span_7class"]
        if "target_foldability" in b:
            padded_foldability[i, :L] = b["target_foldability"]
        mask[i, :L] = True

    return {
        "one_hot_dna": padded_one_hot,
        "target_donor": padded_donor,
        "target_acceptor": padded_acceptor,
        "target_start": padded_start,
        "target_stop": padded_stop,
        "target_span": padded_span,
        "target_span_7class": padded_span_7class,
        "target_foldability": padded_foldability,
        "mask": mask,
        "samples": batch,
    }


class BucketBatchSampler:
    """
    Groups dataset indices of similar sequence lengths into mini-batches of size `batch_size`
    with minimal padding overhead, then shuffles the mini-batches each epoch.
    """
    def __init__(self, lengths: List[int], batch_size: int = 16, shuffle: bool = True, seed: int = 42):
        self.lengths = lengths
        self.batch_size = max(1, batch_size)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        import random
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        # Add small jitter to lengths before sorting so batches diversify across epochs while staying within ~1 bucket
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            indices.sort(key=lambda i: self.lengths[i] + rng.randint(-128, 128))
        else:
            indices.sort(key=lambda i: self.lengths[i])

        batches = [
            indices[i : i + self.batch_size]
            for i in range(0, len(indices), self.batch_size)
        ]
        if self.shuffle:
            rng.shuffle(batches)
        for b in batches:
            yield b


class ContrastiveEnergyTrainer:
    """
    Orchestrates the training of the DNA backbone using the Dual-Space
    Translation-Energy objective, positive-count-normalized Focal Loss,
    class-weighted 7-class phase segmentation, phase-transition consistency,
    and neural foldability distillation.
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
        lambda_phase_trans: float = 0.35,
        lambda_foldability: float = 0.50,
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
        self.lambda_phase_trans = lambda_phase_trans
        self.lambda_foldability = lambda_foldability
        self.loss_type = loss_type
        self.use_phase_aware_span = use_phase_aware_span
        self.operator = SplicedTranslationOperator()

        self.optimizer = torch.optim.AdamW(self.backbone.parameters(), lr=learning_rate, weight_decay=1e-4)

        if loss_type == "focal":
            self.junc_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction="pos_norm")
        else:
            self.junc_loss_fn = nn.BCELoss()

        # Class-weighted CrossEntropy for 7-class mode with ignore_index=-100 for bucket padding:
        # 0: Intergenic (1.0), 1: Intron (1.4), 2-4: Exon Phase 0/1/2 (1.8), 5: Start (6.0), 6: Stop (6.0)
        device = next(self.backbone.parameters()).device
        if use_phase_aware_span:
            weights_7 = torch.tensor([1.0, 1.4, 1.8, 1.8, 1.8, 6.0, 6.0], dtype=torch.float32, device=device)
            self.span_loss_fn = nn.CrossEntropyLoss(weight=weights_7, ignore_index=-100)
        else:
            self.span_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    @staticmethod
    def compute_phase_transition_loss(span_probs_raw: torch.Tensor, target_span_7class: torch.Tensor) -> torch.Tensor:
        """
        Static-shape, boolean-indexing-free phase-transition consistency loss across `(B, 7, L)`
        or `(1, 7, L)` tensors. Penalizes illegal reading-frame transitions inside interior
        coding exons (`target in {2, 3, 4}`) using pure float mask multiplication so MPSGraph/XLA
        never recompiles on dynamic boolean masks.
        """
        if span_probs_raw.shape[1] < 7 or span_probs_raw.shape[-1] < 2:
            return torch.tensor(0.0, device=span_probs_raw.device)

        if target_span_7class.ndim == 1:
            t_7 = target_span_7class.unsqueeze(0)
        else:
            t_7 = target_span_7class

        t_cur = t_7[:, :-1]
        t_nxt = t_7[:, 1:]
        valid_mask = ((t_cur >= 2) & (t_cur <= 4) & (t_nxt >= 2) & (t_nxt <= 4)).to(span_probs_raw.dtype)

        # Extract phase channels (2: Phase 0, 3: Phase 1, 4: Phase 2) -> shape (B, 3, L)
        p_phase = span_probs_raw[:, 2:5, :]
        p_t = p_phase[:, :, :-1]  # (B, 3, L-1)
        # Cyclic shift: P_expected(Phase 0, 1, 2 at t+1) = [P(Phase 2 at t), P(Phase 0 at t), P(Phase 1 at t)]
        p_expected_nxt = torch.stack([p_t[:, 2, :], p_t[:, 0, :], p_t[:, 1, :]], dim=1)
        p_actual_nxt = p_phase[:, :, 1:]

        diff = torch.abs(p_actual_nxt - p_expected_nxt).mean(dim=1)  # (B, L-1)
        return (diff * valid_mask).sum() / torch.clamp(valid_mask.sum(), min=1.0)

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

        s0 = spans[0][0]
        if 0 <= s0 < L:
            log_prob = log_prob + torch.log(preds["start_prob"].squeeze()[s0] + eps)

        e_last = spans[-1][1]
        if 0 <= e_last - 3 < L:
            log_prob = log_prob + torch.log(preds["stop_prob"].squeeze()[e_last - 3] + eps)

        for i in range(len(spans) - 1):
            donor_pos = spans[i][1]
            acceptor_pos = spans[i+1][0] - 1
            if 0 <= donor_pos < L:
                log_prob = log_prob + torch.log(preds["donor_prob"].squeeze()[donor_pos] + eps)
            if 0 <= acceptor_pos < L:
                log_prob = log_prob + torch.log(preds["acceptor_prob"].squeeze()[acceptor_pos] + eps)

        exon_indices = []
        for s, e in spans:
            exon_indices.extend(range(max(0, s), min(L, e)))

        if exon_indices:
            idx_tensor = torch.tensor(exon_indices, dtype=torch.long, device=device)
            exon_probs = preds["span_probs"].squeeze(0)[2, idx_tensor]
            mean_exon_log_prob = torch.log(exon_probs + eps).mean()
            log_prob = log_prob + mean_exon_log_prob

        return log_prob

    def train_batch_step(
        self,
        collated_batch: Dict,
        accum_steps: int = 1,
        step_optimizer: bool = True,
        zero_grad: bool = True,
    ) -> Dict[str, float]:
        """
        Executes a true batched `(B, 4, L_bucket)` forward and backward pass where both
        inputs and targets are padded to static bucket multiples (`L_bucket`), completely
        eliminating per-sample MPSGraph/XLA backward-pass recompilation.
        """
        self.backbone.train()
        if zero_grad:
            self.optimizer.zero_grad()

        device = next(self.backbone.parameters()).device
        one_hot = collated_batch["one_hot_dna"].to(device)      # (B, 4, L_bucket)
        mask = collated_batch["mask"].to(device)                # (B, L_bucket)
        t_donor = collated_batch["target_donor"].to(device)     # (B, L_bucket)
        t_acceptor = collated_batch["target_acceptor"].to(device)
        t_start = collated_batch["target_start"].to(device)
        t_stop = collated_batch["target_stop"].to(device)

        if self.use_phase_aware_span and "target_span_7class" in collated_batch:
            t_span = collated_batch["target_span_7class"].to(device)  # (B, L_bucket) with -100 on pad
        else:
            t_span = collated_batch["target_span"].to(device)

        # 1. Batched Forward Pass on Bucketed Tensor Shape (B, 4, L_bucket)
        preds = self.backbone(one_hot)

        # 2. Masked Junction Boundary Losses (no tensor slicing)
        if isinstance(self.junc_loss_fn, FocalLoss):
            loss_donor = self.junc_loss_fn(preds["donor_prob"], t_donor, mask=mask)
            loss_acceptor = self.junc_loss_fn(preds["acceptor_prob"], t_acceptor, mask=mask)
            loss_start = self.junc_loss_fn(preds["start_prob"], t_start, mask=mask)
            loss_stop = self.junc_loss_fn(preds["stop_prob"], t_stop, mask=mask)
        else:
            m_f = mask.to(one_hot.dtype)
            denom = torch.clamp(m_f.sum(), min=1.0)
            loss_donor = (F.binary_cross_entropy(preds["donor_prob"], t_donor, reduction="none") * m_f).sum() / denom
            loss_acceptor = (F.binary_cross_entropy(preds["acceptor_prob"], t_acceptor, reduction="none") * m_f).sum() / denom
            loss_start = (F.binary_cross_entropy(preds["start_prob"], t_start, reduction="none") * m_f).sum() / denom
            loss_stop = (F.binary_cross_entropy(preds["stop_prob"], t_stop, reduction="none") * m_f).sum() / denom
        loss_junctions = loss_donor + loss_acceptor + loss_start + loss_stop

        # 3. Masked Span Segmentation Loss (ignore_index=-100) + Static-Shape Phase Transition Loss
        loss_phase_trans = torch.tensor(0.0, device=device)
        if self.use_phase_aware_span and "span_probs_raw" in preds:
            if self.span_loss_fn.weight is not None and self.span_loss_fn.weight.device != device:
                self.span_loss_fn.weight = self.span_loss_fn.weight.to(device)
            loss_span = self.span_loss_fn(preds["span_logits"], t_span)
            loss_phase_trans = self.compute_phase_transition_loss(preds["span_probs_raw"], t_span)
        else:
            loss_span = self.span_loss_fn(preds["span_probs"], t_span)

        # 4. Masked Auxiliary Neural Foldability Distillation Loss
        loss_foldability = torch.tensor(0.0, device=device)
        if "foldability_prob" in preds and "target_foldability" in collated_batch:
            t_fold = collated_batch["target_foldability"].to(device)
            m_f = mask.to(one_hot.dtype)
            raw_fold = F.smooth_l1_loss(preds["foldability_prob"], t_fold, reduction="none")
            loss_foldability = (raw_fold * m_f).sum() / torch.clamp(m_f.sum(), min=1.0)

        # 5. Composite Objective
        total_loss = (
            self.lambda_junc * loss_junctions
            + self.lambda_span * loss_span
            + self.lambda_phase_trans * loss_phase_trans
            + self.lambda_foldability * loss_foldability
        )

        loss_to_backprop = total_loss / max(1, accum_steps)
        loss_to_backprop.backward()

        if step_optimizer:
            torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), max_norm=1.0)
            self.optimizer.step()
            if not zero_grad:
                self.optimizer.zero_grad()

        return {
            "total_loss": total_loss.item(),
            "loss_junctions": loss_junctions.item(),
            "loss_span": loss_span.item(),
            "loss_phase_trans": loss_phase_trans.item(),
            "loss_foldability": loss_foldability.item(),
            "contrastive_loss": 0.0,
            "e_pos": 0.0,
            "is_true_valid": True,
        }

    def train_step(
        self,
        batch: Dict,
        compute_energy: bool = False,
        accum_steps: int = 1,
        step_optimizer: bool = True,
        zero_grad: bool = True,
    ) -> Dict[str, float]:
        """
        Executes a single optimization step on either a collated mini-batch (`one_hot_dna` is 3D `(B, 4, L)`)
        or a single sample (`one_hot_dna` is 2D `(4, L)`). When `compute_energy=False`, routes through
        `train_batch_step` so targets are bucket-padded to `one_hot_dna.shape[-1]` without dynamic slicing.
        """
        if batch["one_hot_dna"].ndim == 3 and not compute_energy:
            return self.train_batch_step(
                batch,
                accum_steps=accum_steps,
                step_optimizer=step_optimizer,
                zero_grad=zero_grad,
            )

        if not compute_energy:
            # Pad single sample targets to match bucketed one_hot_dna length so backward pass uses static bucket shape
            L_bucket = batch["one_hot_dna"].shape[-1]
            collated = collate_genomic_batch([batch], bucket_step=L_bucket)
            return self.train_batch_step(
                collated,
                accum_steps=accum_steps,
                step_optimizer=step_optimizer,
                zero_grad=zero_grad,
            )

        self.backbone.train()
        if zero_grad:
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

        preds = self.backbone(one_hot)
        L = one_hot.shape[-1]
        valid_len = batch.get("valid_len", L)

        if valid_len < L:
            preds = {
                k: v[..., :valid_len] if isinstance(v, torch.Tensor) and v.ndim >= 2 and v.shape[-1] == L else v
                for k, v in preds.items()
            }
            L = valid_len

        loss_donor = self.junc_loss_fn(preds["donor_prob"].squeeze(), t_donor)
        loss_acceptor = self.junc_loss_fn(preds["acceptor_prob"].squeeze(), t_acceptor)
        loss_start = self.junc_loss_fn(preds["start_prob"].squeeze(), t_start)
        loss_stop = self.junc_loss_fn(preds["stop_prob"].squeeze(), t_stop)
        loss_junctions = loss_donor + loss_acceptor + loss_start + loss_stop

        loss_phase_trans = torch.tensor(0.0, device=device)
        if self.use_phase_aware_span and "span_probs_raw" in preds:
            if self.span_loss_fn.weight is not None and self.span_loss_fn.weight.device != device:
                self.span_loss_fn.weight = self.span_loss_fn.weight.to(device)
            loss_span = self.span_loss_fn(preds["span_logits"].squeeze(0).permute(1, 0), t_span)
            loss_phase_trans = self.compute_phase_transition_loss(preds["span_probs_raw"], t_span)
        else:
            loss_span = self.span_loss_fn(preds["span_probs"].squeeze(0).permute(1, 0), t_span)

        loss_foldability = torch.tensor(0.0, device=device)
        if "foldability_prob" in preds and "target_foldability" in batch:
            t_fold = batch["target_foldability"].to(device)
            loss_foldability = F.smooth_l1_loss(preds["foldability_prob"].squeeze(), t_fold)

        contrastive_loss = torch.tensor(0.0, device=device)
        contrastive_cost = 0.0
        e_pos = 0.0
        is_valid = True

        if compute_energy and self.oracle is not None:
            true_tx = batch["true_transcript"]
            e_pos_dict = self.oracle.score_sequence(true_tx.protein_sequence)
            e_pos = e_pos_dict["total_energy"]
            is_valid = e_pos_dict["is_valid_orf"]

            score_pos = self.compute_transcript_log_prob(preds, true_tx.exon_spans, L)

            negatives = batch.get("negatives")
            if not negatives:
                negatives = self.operator.generate_counterfactual_negatives(batch["dna_str"], true_tx)

            ranking_losses = []
            for neg_name, neg_tx in negatives:
                e_neg_dict = self.oracle.score_sequence(neg_tx.protein_sequence)
                e_neg = e_neg_dict["total_energy"]
                energy_margin = max(self.margin, (e_pos - e_neg) + self.margin)
                score_neg = self.compute_transcript_log_prob(preds, neg_tx.exon_spans, L)
                violation = F.relu(score_neg - score_pos + energy_margin)
                ranking_losses.append(violation)

            if ranking_losses:
                contrastive_loss = torch.stack(ranking_losses).mean()
                contrastive_cost = contrastive_loss.item()

        total_loss = (
            self.lambda_junc * loss_junctions
            + self.lambda_span * loss_span
            + self.lambda_phase_trans * loss_phase_trans
            + self.lambda_foldability * loss_foldability
            + self.lambda_contrastive * contrastive_loss
        )

        loss_to_backprop = total_loss / max(1, accum_steps)
        loss_to_backprop.backward()

        if step_optimizer:
            torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), max_norm=1.0)
            self.optimizer.step()
            if not zero_grad:
                self.optimizer.zero_grad()

        if device.type == "mps":
            torch.mps.empty_cache()

        return {
            "total_loss": total_loss.item(),
            "loss_junctions": loss_junctions.item(),
            "loss_span": loss_span.item(),
            "loss_phase_trans": loss_phase_trans.item(),
            "loss_foldability": loss_foldability.item(),
            "contrastive_loss": contrastive_cost,
            "e_pos": e_pos,
            "is_true_valid": is_valid,
        }
