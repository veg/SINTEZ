"""Tests for ProteinEnergyOracle and Shannon complexity normalization."""

import pytest
import math
from sintez.protein_oracle import ProteinEnergyOracle


def test_shannon_entropy_calculation():
    # Single repeated amino acid -> 0.0 bits
    assert ProteinEnergyOracle.calculate_shannon_entropy("AAAAAA") == 0.0

    # 4 distinct amino acids in equal proportions -> log2(4) = 2.0 bits
    assert math.isclose(ProteinEnergyOracle.calculate_shannon_entropy("ABCD" * 10), 2.0, rel_tol=1e-5)

    # 20 distinct amino acids in equal proportions -> log2(20) ~ 4.3219 bits
    all_20 = "ACDEFGHIKLMNPQRSTVWY"
    assert math.isclose(ProteinEnergyOracle.calculate_shannon_entropy(all_20 * 5), math.log2(20), rel_tol=1e-5)


def test_complexity_penalty_on_repeats():
    oracle = ProteinEnergyOracle(device="cpu")
    # Low-complexity periodic tandem repeat (TARE-like)
    repeat_seq = "MLTDMMTDITTNITTNLTTNNITHMTTNIITNIITNMVTNMITNMTTYMTTNMTTNIIINMTTNIIITNMITNISTN"
    # Diverse natural-like sequence
    diverse_seq = "MKMYYLKMLLFTFLINTLVARHYENFVNNHYNVSLIQNKTKRVTIKSRLLNKKKKYKSLK"

    res_repeat = oracle.score_sequence(repeat_seq)
    res_diverse = oracle.score_sequence(diverse_seq)

    # Repeat has low Shannon entropy (<3.0 bits)
    assert res_repeat["shannon_entropy"] < 3.0
    # Diverse sequence has high Shannon entropy (>3.4 bits)
    assert res_diverse["shannon_entropy"] > 3.4

    # Repeat receives substantial penalty
    assert res_repeat["entropy_penalty"] > 3.0
    # Diverse sequence receives zero penalty
    assert res_diverse["entropy_penalty"] == 0.0


def test_ptc_internal_stop_penalty():
    oracle = ProteinEnergyOracle(device="cpu")
    normal_seq = "MKMYYLKMLLFTFLINTLVARHYENFVNNHYNVSLIQNKTKRVTIKSRLL"
    stop_seq = "MKMYYLKMLLFTFLINTLVARHY*NFVNNHYNVSLIQNKTKRVTIKSRLL"

    res_normal = oracle.score_sequence(normal_seq)
    res_stop = oracle.score_sequence(stop_seq)

    assert res_normal["is_valid_orf"] is True
    assert res_stop["is_valid_orf"] is False
    assert res_stop["total_energy"] >= res_normal["total_energy"] + 90.0
