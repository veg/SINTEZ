"""Tests for SplicedTranslationOperator."""

import pytest
from sintez.translation_operator import SplicedTranslationOperator, SplicedTranscript


def test_single_exon_translation():
    op = SplicedTranslationOperator()
    # ATG GCT TAA -> M A *
    dna = "ATGGCTTAA"
    tx = op.process_transcript(dna, [(0, 9)], strand="+")
    assert tx.is_in_frame
    assert tx.has_valid_start
    assert tx.has_valid_stop
    assert not tx.has_internal_stop
    assert tx.protein_sequence == "MA*"


def test_split_codon_splicing():
    op = SplicedTranslationOperator()
    # Exon 1: ATG G (4 bp)
    # Intron: GT...AG (8 bp)
    # Exon 2: CT TAA (5 bp)
    # Spliced: ATG GCT TAA -> M A *
    dna = "ATGG" + "GTAAAAG" + "CTTAA"
    spans = [(0, 4), (11, 16)]
    tx = op.process_transcript(dna, spans, strand="+")
    assert tx.is_in_frame
    assert tx.protein_sequence == "MA*"
    assert tx.has_valid_start
    assert tx.has_valid_stop
    assert not tx.has_internal_stop


def test_reverse_strand_translation():
    op = SplicedTranslationOperator()
    # On reverse strand: original forward DNA is reverse-complement of ATG GCT TAA -> TTA AGC CAT
    dna = "TTAAGCCAT"
    tx = op.process_transcript(dna, [(0, 9)], strand="-")
    assert tx.is_in_frame
    assert tx.protein_sequence == "MA*"


def test_counterfactual_negative_generation():
    op = SplicedTranslationOperator()
    dna = "ATGG" + "GTAAAAG" + "CTTAA" + "AA"
    spans = [(0, 4), (11, 16)]
    tx = op.process_transcript(dna, spans, strand="+")
    negatives = op.generate_counterfactual_negatives(dna, tx)
    assert len(negatives) >= 2
    # Negatives should introduce frameshifts or broken junctions
    for neg_name, neg_tx in negatives:
        assert isinstance(neg_tx, SplicedTranscript)
