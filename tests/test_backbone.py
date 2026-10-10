"""Tests for CompositionNormalizedBackbone."""

import pytest
import torch
from sintez.dna_backbone import CompositionNormalizedBackbone, DNAEncoderConfig


def test_one_hot_encoding():
    seq = "ACGTN"
    one_hot = CompositionNormalizedBackbone.sequence_to_one_hot(seq)
    assert one_hot.shape == (1, 4, 5)
    # A
    assert torch.all(one_hot[0, :, 0] == torch.tensor([1.0, 0.0, 0.0, 0.0]))
    # C
    assert torch.all(one_hot[0, :, 1] == torch.tensor([0.0, 1.0, 0.0, 0.0]))
    # G
    assert torch.all(one_hot[0, :, 2] == torch.tensor([0.0, 0.0, 1.0, 0.0]))
    # T
    assert torch.all(one_hot[0, :, 3] == torch.tensor([0.0, 0.0, 0.0, 1.0]))
    # N (uniform distribution)
    assert torch.allclose(one_hot[0, :, 4], torch.tensor([0.25, 0.25, 0.25, 0.25]))


def test_backbone_parameter_count():
    cfg = DNAEncoderConfig(
        in_channels=4,
        hidden_dim=64,
        kernel_size=7,
        norm_window_size=201,
        dropout=0.1,
    )
    model = CompositionNormalizedBackbone(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    # Exactly 238,407 parameters
    assert total_params == 238407


def test_backbone_forward_shapes():
    cfg = DNAEncoderConfig(in_channels=4, hidden_dim=64, kernel_size=7, norm_window_size=201)
    model = CompositionNormalizedBackbone(cfg)
    model.eval()

    L = 500
    dummy_dna = "ATGC" * 125
    one_hot = CompositionNormalizedBackbone.sequence_to_one_hot(dummy_dna)

    with torch.no_grad():
        out = model(one_hot)

    assert out["span_probs"].shape == (1, 3, L)
    assert out["donor_prob"].shape == (1, L)
    assert out["acceptor_prob"].shape == (1, L)
    assert out["start_prob"].shape == (1, L)
    assert out["stop_prob"].shape == (1, L)

    # Probabilities in [0, 1]
    assert (out["span_probs"] >= 0.0).all() and (out["span_probs"] <= 1.0).all()
    assert (out["donor_prob"] >= 0.0).all() and (out["donor_prob"] <= 1.0).all()
