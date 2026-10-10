"""Tests for DualStrandResolver."""

import pytest
from sintez.dual_strand import DualStrandResolver


def test_dual_strand_no_overlap():
    preds = [
        {"spans": [(100, 500)], "strand": "+", "score": 20.0},
        {"spans": [(600, 1000)], "strand": "-", "score": 18.0},
    ]
    resolved = DualStrandResolver.resolve_conflicts(preds, max_reciprocal_overlap=0.35)
    assert len(resolved) == 2


def test_dual_strand_collision_suppression():
    # Direct antisense overlap: 100 to 500 on both strands
    preds = [
        {"spans": [(100, 500)], "strand": "+", "score": 25.0},
        {"spans": [(120, 480)], "strand": "-", "score": 15.0},
    ]
    resolved = DualStrandResolver.resolve_conflicts(preds, max_reciprocal_overlap=0.35)
    # Lower-scoring (-) strand candidate should be suppressed
    assert len(resolved) == 1
    assert resolved[0]["strand"] == "+"
    assert resolved[0]["score"] == 25.0


def test_dual_strand_same_strand_overlap_untouched():
    # Same strand overlaps are handled upstream by intra-strand NMS; resolver only arbitrates cross-strand
    preds = [
        {"spans": [(100, 300)], "strand": "+", "score": 20.0},
        {"spans": [(200, 400)], "strand": "+", "score": 19.0},
    ]
    resolved = DualStrandResolver.resolve_conflicts(preds, max_reciprocal_overlap=0.35)
    assert len(resolved) == 2
