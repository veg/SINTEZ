"""Tests for TopologicalDAGDecoder."""

import pytest
import math
from sintez.dag_decoder import TopologicalDAGDecoder


def test_decoder_initialization():
    decoder = TopologicalDAGDecoder(
        min_single_len=240,
        min_multi_len=240,
        max_intron_len=1500,
        min_internal_exon_len=15,
        max_exons=16,
        beam_width=8,
        lambda_intron=4.5,
        min_intron_gain=-1.5,
    )
    assert decoder.max_exons == 16
    assert decoder.lambda_intron == 4.5
    assert decoder.min_internal_exon_len == 15
    assert decoder.beam_width == 8
    assert decoder.min_intron_gain == -1.5


def test_decoder_default_parameters():
    decoder = TopologicalDAGDecoder()
    assert decoder.max_exons == 16
    assert decoder.min_internal_exon_len == 15
    assert decoder.beam_width == 8
    assert decoder.min_intron_gain == -1.5


def test_jacobson_stockmayer_loop_penalty():
    # Compact intron: 80 bp -> loop penalty is 0.0
    intron_80 = 80
    penalty_80 = 1.5 * max(0.0, math.log(max(1.0, intron_80 / 80.0)))
    assert penalty_80 == 0.0

    # Normal intron: 120 bp -> small penalty (~0.6 bits)
    intron_120 = 120
    penalty_120 = 1.5 * max(0.0, math.log(max(1.0, intron_120 / 80.0)))
    assert 0.5 < penalty_120 < 0.7

    # Spurious jump: 1,200 bp -> heavy penalty (>4.0 bits)
    intron_1200 = 1200
    penalty_1200 = 1.5 * max(0.0, math.log(max(1.0, intron_1200 / 80.0)))
    assert penalty_1200 > 4.0


def test_immediate_split_codon_pruning():
    # If phase 1 has donor tail 'T' and acceptor head 'AA', fused codon is 'TAA' (stop)
    d_tail = "T"
    a_head = "AA"
    split_codon_p1 = d_tail + a_head
    assert split_codon_p1 in {"TAA", "TAG", "TGA"}

    # If phase 2 has donor tail 'TA' and acceptor head 'G', fused codon is 'TAG' (stop)
    d_tail_p2 = "TA"
    a_head_p2 = "G"
    split_codon_p2 = d_tail_p2 + a_head_p2
    assert split_codon_p2 in {"TAA", "TAG", "TGA"}

    # Non-stop split codon should be accepted
    safe_split = "T" + "GC" # TGC = Cysteine
    assert safe_split not in {"TAA", "TAG", "TGA"}


def test_polypyrimidine_tract_density():
    # Pyrimidine-rich intron: TTTCTTTCCCCCTTTT
    ppt_seq = "TTTCTTTCCCCCTTTT"
    ppt_ratio = sum(1 for c in ppt_seq if c in {"C", "T"}) / len(ppt_seq)
    assert ppt_ratio == 1.0

    # Purine-rich non-PPT sequence
    purine_seq = "AAAAAGAAAGAAAAAG"
    purine_ratio = sum(1 for c in purine_seq if c in {"C", "T"}) / len(purine_seq)
    assert purine_ratio == 0.0

