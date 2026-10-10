"""
Counterfactual Contrastive Training Engine
==========================================
Trains the composition-invariant DNA backbone using margin-based contrastive
protein energy objectives paired with sparse junction boundary losses.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
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


class CounterfactualDataset(Dataset):
    """
    Dataset that pairs genomic sequences with true exon structures and
    synthesizes on-the-fly hard counterfactual frame perturbations.
    """
    def __init__(self, samples: List[GenomicGeneSample]):
        self.samples = samples
        self.operator = SplicedTranslationOperator()

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

        # 2. Synthesize Hard Counterfactual Negatives
        negatives = self.operator.generate_counterfactual_negatives(dna_str, true_transcript)

        # 3. Create Ground Truth Target Masks for DNA Backbone
        L = len(dna_str)
        target_donor = torch.zeros(L, dtype=torch.float32)
        target_acceptor = torch.zeros(L, dtype=torch.float32)
        target_start = torch.zeros(L, dtype=torch.float32)
        target_stop = torch.zeros(L, dtype=torch.float32)
        target_span = torch.zeros(L, dtype=torch.long)  # 0: Intergenic, 1: Intron, 2: Exon

        for s, e in spans:
            target_span[s:e] = 2

        for i in range(len(spans) - 1):
            intron_start = spans[i][1]
            intron_end = spans[i+1][0]
            target_span[intron_start:intron_end] = 1
            target_donor[intron_start] = 1.0
            target_acceptor[intron_end - 1] = 1.0

        target_start[spans[0][0]] = 1.0
        target_stop[spans[-1][1] - 3] = 1.0

        one_hot_dna = CompositionNormalizedBackbone.sequence_to_one_hot(dna_str).squeeze(0)

        return {
            "dna_str": dna_str,
            "one_hot_dna": one_hot_dna,
            "target_donor": target_donor,
            "target_acceptor": target_acceptor,
            "target_start": target_start,
            "target_stop": target_stop,
            "target_span": target_span,
            "true_transcript": true_transcript,
            "negatives": negatives,
        }


class ContrastiveEnergyTrainer:
    """
    Orchestrates the training of the DNA backbone using the Dual-Space
    Translation-Energy objective.
    """
    def __init__(
        self,
        backbone: CompositionNormalizedBackbone,
        oracle: ProteinEnergyOracle,
        learning_rate: float = 5e-4,
        margin: float = 10.0,
        lambda_junc: float = 3.0,
        lambda_span: float = 1.0,
        lambda_contrastive: float = 0.15,
    ):
        self.backbone = backbone
        self.oracle = oracle
        self.margin = margin
        self.lambda_junc = lambda_junc
        self.lambda_span = lambda_span
        self.lambda_contrastive = lambda_contrastive

        self.optimizer = torch.optim.AdamW(self.backbone.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()

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
        t_span = batch["target_span"].to(device)

        preds = self.backbone(one_hot)

        loss_donor = self.bce_loss(preds["donor_prob"].squeeze(), t_donor)
        loss_acceptor = self.bce_loss(preds["acceptor_prob"].squeeze(), t_acceptor)
        loss_start = self.bce_loss(preds["start_prob"].squeeze(), t_start)
        loss_stop = self.bce_loss(preds["stop_prob"].squeeze(), t_stop)
        loss_junctions = loss_donor + loss_acceptor + loss_start + loss_stop

        loss_span = self.ce_loss(preds["span_probs"].squeeze(0).permute(1, 0), t_span)

        contrastive_cost = 0.0
        e_pos = 0.0
        is_valid = True

        if compute_energy and self.oracle is not None:
            true_tx = batch["true_transcript"]
            e_pos_dict = self.oracle.score_sequence(true_tx.protein_sequence)
            e_pos = e_pos_dict["total_energy"]
            is_valid = e_pos_dict["is_valid_orf"]

            negatives = batch["negatives"]
            for neg_name, neg_tx in negatives:
                e_neg_dict = self.oracle.score_sequence(neg_tx.protein_sequence)
                e_neg = e_neg_dict["total_energy"]
                margin_violation = max(0.0, e_pos - e_neg + self.margin)
                contrastive_cost += margin_violation

            if len(negatives) > 0:
                contrastive_cost /= len(negatives)

        total_loss = (
            self.lambda_junc * loss_junctions
            + self.lambda_span * loss_span
            + self.lambda_contrastive * float(contrastive_cost)
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "loss_junctions": loss_junctions.item(),
            "loss_span": loss_span.item(),
            "contrastive_cost": contrastive_cost,
            "e_pos": e_pos,
            "is_true_valid": is_valid,
        }
