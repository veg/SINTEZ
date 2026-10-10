"""Tests for TransposonClassifier."""

import pytest
from sintez.te_classifier import TransposonClassifier


def test_classify_host_protein():
    classifier = TransposonClassifier()
    # Typical host enzyme (e.g. glycolytic kinase fragment)
    host_pep = "MVKVGVNGFGRIGRLVTRAAFNSGKVDIVAINDPFIDLNYMVYMFQYDSTHGKFHGTVKAENGKLVINGNPITIFQERDPSKIKWGDAGAEYVVESTGVFTTMEKAGAHLQGGAKRVIISAPSADAPMFVMGVNHEKYDNSLKIISNASCTTNCLAPLAKVIHDNFGIVEGLMTTVHAITATQKTVDGPSGKLWRDGRGALQNIIPASTGAAKAVGKVIPELNGKLTGMAFRVPTANVSVVDLTCRLEKPAKYDDIKKVV"
    res = classifier.classify_protein(host_pep, [(100, 800)], "A" * 1000)
    assert not res["is_transposon"]
    assert res["biotype"] == "protein_coding"
    assert res["gene_type"] == "host"


def test_classify_rvt_signature():
    classifier = TransposonClassifier()
    # Reverse transcriptase signature containing catalytic triad YMDD
    te_pep = "MSLKQKLIKDKYMDDVVIGSNKDE"
    res = classifier.classify_protein(te_pep, [(100, 200)], "A" * 1000)
    assert res["is_transposon"]
    assert res["biotype"] == "transposon_derived"
    assert res["gene_type"] == "mobile_element"
    assert "rvt_catalytic_triad" in res["te_signatures"]


def test_classify_1360_transposase():
    classifier = TransposonClassifier()
    te_pep = "MRNAIRFFGTRFFGRGSRGGSRLSRIFVRERESEERYSKQLFSTHTVIADNFMCAHE"
    res = classifier.classify_protein(te_pep, [(100, 300)], "A" * 1000)
    assert res["is_transposon"]
    assert res["biotype"] == "transposon_derived"
    assert "transposase_1360_hoppel" in res["te_signatures"]


def test_classify_kmer_repeat_depth():
    # Build a small sequence with high copy repeats
    repeat_unit = "ATGCGATCGATCGATC"
    seq = (repeat_unit * 10) + ("G" * 500)
    classifier = TransposonClassifier(genome_seq=seq, k=10, min_repeat_depth=3.0)

    # Candidate inside the high-copy repeat
    res = classifier.classify_protein("MQRI", [(0, 100)], seq)
    assert res["is_transposon"]
    assert res["repeat_depth"] >= 3.0


def test_classify_soft_masked():
    classifier = TransposonClassifier(min_masked_fraction=0.50)
    seq = "A" * 100 + "atcg" * 50 + "A" * 100
    # Candidate in soft-masked region
    res = classifier.classify_protein("MTEST", [(100, 300)], seq)
    assert res["is_transposon"]
    assert res["masked_fraction"] == 1.0


def test_stratify_predictions():
    classifier = TransposonClassifier()
    preds = [
        {"spans": [(0, 60)], "strand": "+", "peptide": "MVKVGVNGFGRIGRLVTRAA"},
        {"spans": [(100, 160)], "strand": "+", "peptide": "MSLKQKLIKDKYMDDVVIGS"},
    ]
    host_preds, te_preds = classifier.stratify_predictions(preds, "A" * 500)
    assert len(host_preds) == 1
    assert len(te_preds) == 1
    assert host_preds[0]["biotype"] == "protein_coding"
    assert te_preds[0]["biotype"] == "transposon_derived"
