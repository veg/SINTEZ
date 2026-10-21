"""
Tests for Upgraded SINTEZ Training Pipeline
===========================================
Verifies:
- FocalLoss gradient and reduction behavior
- MultiScaleCompositionNormalization and foundation_v2 backbone
- Phase-aware 7-class exon annotation
- Differentiable dual-space path-ranking loss
- Batched collation for GPU/MPS
"""

import pytest
import torch
import torch.nn as nn

from sintez.dna_backbone import (
    CompositionNormalizedBackbone,
    DNAEncoderConfig,
    MultiScaleCompositionNormalization,
)
from sintez.trainer import (
    FocalLoss,
    GenomicGeneSample,
    CounterfactualDataset,
    ContrastiveEnergyTrainer,
    collate_genomic_batch,
)
from sintez.protein_oracle import ProteinEnergyOracle


def test_focal_loss():
    loss_fn = FocalLoss(alpha=0.75, gamma=2.0)
    probs = torch.tensor([0.9, 0.1, 0.5], requires_grad=True)
    targets = torch.tensor([1.0, 0.0, 1.0])

    loss = loss_fn(probs, targets)
    assert loss.item() > 0.0
    loss.backward()
    assert probs.grad is not None
    assert torch.all(probs.grad != 0.0)


def test_multiscale_composition_normalization():
    norm = MultiScaleCompositionNormalization(channels=4, window_sizes=(11, 21, 51))
    x = torch.randn(2, 4, 100)
    out = norm(x)
    # 4 channels * 3 scales = 12 channels
    assert out.shape == (2, 12, 100)


def test_foundation_v2_architecture():
    cfg = DNAEncoderConfig.foundation_v2(
        hidden_dim=32,  # small for fast test
        dilations=(1, 2, 4, 8),
        norm_window_sizes=(21, 51),
        num_span_classes=7,
    )
    model = CompositionNormalizedBackbone(cfg)
    model.eval()

    L = 200
    dummy_dna = "ATGC" * 50
    one_hot = CompositionNormalizedBackbone.sequence_to_one_hot(dummy_dna)

    with torch.no_grad():
        out = model(one_hot)

    # 3-class backward compatible projection
    assert out["span_probs"].shape == (1, 3, L)
    # 7-class raw phase-aware probabilities
    assert out["span_probs_raw"].shape == (1, 7, L)
    assert out["donor_prob"].shape == (1, L)
    assert out["acceptor_prob"].shape == (1, L)
    assert out["start_prob"].shape == (1, L)
    assert out["stop_prob"].shape == (1, L)


def test_counterfactual_dataset_phase_aware():
    # Construct a simple spliced locus:
    # Exon 1: 0..12 (12 bp, 4 codons)
    # Intron 1: 12..25 (13 bp)
    # Exon 2: 25..37 (12 bp, 4 codons) -> Total CDS: 24 bp, 8 codons
    seq = "ATGGCCAAGTTT" + "GTACGTACGTACG" + "AAAGCCGATTGA"
    sample = GenomicGeneSample(
        dna_sequence=seq,
        exon_spans=[(0, 12), (25, 37)],
        strand="+",
        gene_id="test_gene",
    )
    dataset = CounterfactualDataset([sample])
    item = dataset[0]

    t_span = item["target_span"]
    t_span_7 = item["target_span_7class"]

    # Exons marked as 2 in 3-class target
    assert (t_span[0:12] == 2).all()
    assert (t_span[12:25] == 1).all()
    assert (t_span[25:37] == 2).all()

    # Intron marked as 1 in 7-class target
    assert (t_span_7[12:25] == 1).all()

    # Check phase continuity: in Exon 1, phases cycle 0, 1, 2
    # positions 3..12 should be phases 0, 1, 2
    for pos in range(3, 12):
        expected_phase = (pos) % 3
        assert t_span_7[pos].item() == 2 + expected_phase

    # Start codon marked as 5
    assert (t_span_7[0:3] == 5).all()
    # Stop codon marked as 6
    assert (t_span_7[34:37] == 6).all()


def test_collate_genomic_batch():
    s1 = GenomicGeneSample("ATGC" * 10, [(0, 12)], "+", "g1")
    s2 = GenomicGeneSample("ATGC" * 20, [(0, 24)], "+", "g2")
    dataset = CounterfactualDataset([s1, s2])

    batch = collate_genomic_batch([dataset[0], dataset[1]], bucket_step=None)
    assert batch["one_hot_dna"].shape == (2, 4, 80)
    assert batch["target_donor"].shape == (2, 80)
    assert batch["mask"].shape == (2, 80)
    # Sample 1 mask is True for 40, False for 40
    assert batch["mask"][0, :40].all()
    assert not batch["mask"][0, 40:].any()
    assert (batch["target_span"][0, 40:] == -100).all()
    # Sample 2 mask is True for all 80
    assert batch["mask"][1, :80].all()

    # Bucketed collation (bucket_step=128) -> pads 80 up to 128
    batch_b = collate_genomic_batch([dataset[0], dataset[1]], bucket_step=128)
    assert batch_b["one_hot_dna"].shape == (2, 4, 128)
    assert batch_b["mask"].shape == (2, 128)
    assert not batch_b["mask"][:, 80:].any()
    assert (batch_b["target_span_7class"][:, 80:] == -100).all()



def test_differentiable_contrastive_trainer_gradients():
    cfg = DNAEncoderConfig.classic_v1(hidden_dim=32, norm_window_size=21)
    backbone = CompositionNormalizedBackbone(cfg)
    trainer = ContrastiveEnergyTrainer(
        backbone=backbone,
        oracle=None,  # test without ESM-2 load for speed
        learning_rate=1e-3,
        loss_type="focal",
        lambda_contrastive=1.0,
    )

    seq = "ATGGCCAAGTTT" + "GTACGTACGTACG" + "AAAGCCGATTGA"
    sample = GenomicGeneSample(dna_sequence=seq, exon_spans=[(0, 12), (25, 37)], strand="+")
    dataset = CounterfactualDataset([sample])
    item = dataset[0]

    # Run train step
    metrics = trainer.train_step(item, compute_energy=False)
    assert "total_loss" in metrics
    assert metrics["total_loss"] > 0.0

    # Test differentiable path score function directly
    one_hot = item["one_hot_dna"].unsqueeze(0)
    preds = backbone(one_hot)
    score_pos = trainer.compute_transcript_log_prob(preds, [(0, 12), (25, 37)], len(seq))
    score_neg = trainer.compute_transcript_log_prob(preds, [(0, 13), (25, 37)], len(seq))

    # Margin ranking loss between pos and neg
    rank_loss = torch.relu(score_neg - score_pos + 1.0)
    rank_loss.backward()

    # Verify gradients flow into backbone parameters!
    has_grad = any(p.grad is not None and torch.any(p.grad != 0.0) for p in backbone.parameters())
    assert has_grad, "Expected non-zero gradients from path ranking loss into backbone!"


def test_length_bucketing_and_gradient_flow():
    cfg = DNAEncoderConfig.classic_v1(hidden_dim=32, norm_window_size=21)
    backbone = CompositionNormalizedBackbone(cfg)
    trainer = ContrastiveEnergyTrainer(
        backbone=backbone,
        oracle=None,
        learning_rate=1e-3,
        loss_type="focal",
    )

    # 37 bp sequence with bucket_step=64 -> should pad to 64 bp
    seq = "ATGGCCAAGTTT" + "GTACGTACGTACG" + "AAAGCCGATTGA"
    assert len(seq) == 37
    sample = GenomicGeneSample(dna_sequence=seq, exon_spans=[(0, 12), (25, 37)], strand="+")
    dataset = CounterfactualDataset([sample], bucket_step=64)
    item = dataset[0]

    assert item["valid_len"] == 37
    assert item["one_hot_dna"].shape == (4, 64)
    assert item["target_donor"].shape == (37,)

    # Train step with bucketing
    metrics = trainer.train_step(item, compute_energy=False)
    assert metrics["total_loss"] > 0.0

    # Ensure gradients flowed to backbone parameters
    has_grad = any(p.grad is not None and torch.any(p.grad != 0.0) for p in backbone.parameters())
    assert has_grad, "Expected gradients to flow to backbone through bucket-padded inputs!"


def test_foundation_v3_architecture_and_losses():
    cfg = DNAEncoderConfig.foundation_v3(
        hidden_dim=32,
        dilations=(1, 2, 4, 8),
        norm_window_sizes=(21, 51),
        num_span_classes=7,
    )
    backbone = CompositionNormalizedBackbone(cfg)

    # Test state_dict config reconstruction round-trip
    sd = backbone.state_dict()
    reconstructed_cfg = DNAEncoderConfig.from_state_dict(sd)
    assert reconstructed_cfg.hidden_dim == 32
    assert reconstructed_cfg.num_span_classes == 7
    assert reconstructed_cfg.use_codon_conv is True
    assert reconstructed_cfg.use_asymmetric_heads is True
    assert reconstructed_cfg.use_foldability_head is True

    trainer = ContrastiveEnergyTrainer(
        backbone=backbone,
        oracle=None,
        learning_rate=1e-3,
        loss_type="focal",
        use_phase_aware_span=True,
    )

    # Multi-gene window with two co-directional genes
    g1 = "ATGGCCAAGTTT" + "GTACGTACGTACG" + "AAAGCCGATTGA"  # 0..37
    spacer = "A" * 23                                      # 37..60
    g2 = "ATGGCCGATTAA"                                    # 60..72
    seq = g1 + spacer + g2
    sample = GenomicGeneSample(
        dna_sequence=seq,
        exon_spans=[(0, 12), (25, 37)],
        strand="+",
        gene_id="multi_win",
        multi_gene_spans=[[(0, 12), (25, 37)], [(60, 72)]],
    )
    dataset = CounterfactualDataset([sample], bucket_step=32)
    item = dataset[0]

    # Verify both genes are labeled in target_span and target_foldability
    assert (item["target_span"][0:12] == 2).all()
    assert (item["target_span"][60:72] == 2).all()
    assert (item["target_foldability"][0:12] > 0.5).all()
    assert (item["target_foldability"][60:72] > 0.5).all()
    assert (item["target_foldability"][37:60] == 0.0).all()

    metrics = trainer.train_step(item, compute_energy=False)
    assert metrics["total_loss"] > 0.0
    assert metrics["loss_phase_trans"] >= 0.0
    assert metrics["loss_foldability"] > 0.0
    assert backbone.motif_gate_scale.grad is not None
