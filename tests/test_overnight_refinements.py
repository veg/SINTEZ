"""Unit tests for overnight SINTEZ refinements (batch oracle, holdout, v3.1 gate, 17-mer depths)."""

import torch
import numpy as np

from sintez.cohort_builder import BENCHMARK_HOLDOUT_CHROMS, BENCHMARK_HOLDOUT_STEMS
from sintez.dag_decoder import TopologicalDAGDecoder, _compute_canon_17mer_depths
from sintez.dna_backbone import CompositionNormalizedBackbone, DNAEncoderConfig
from sintez.protein_oracle import ProteinEnergyOracle
from sintez.trainer import CounterfactualDataset, GenomicGeneSample


def test_sharp_motif_gate_roundtrip_and_param_parity():
    cfg_v3 = DNAEncoderConfig.foundation_v3()
    m_v3 = CompositionNormalizedBackbone(cfg_v3)

    cfg_v31 = DNAEncoderConfig.foundation_v3()
    cfg_v31.sharp_motif_gate = True
    m_v31 = CompositionNormalizedBackbone(cfg_v31)

    p_v3 = sum(p.numel() for p in m_v3.parameters())
    p_v31 = sum(p.numel() for p in m_v31.parameters())
    assert p_v3 == p_v31

    sd_v3 = m_v3.state_dict()
    sd_v31 = m_v31.state_dict()
    assert "_sharp_motif_gate" not in sd_v3
    assert "_sharp_motif_gate" in sd_v31

    reconstructed_v3 = DNAEncoderConfig.from_state_dict(sd_v3)
    reconstructed_v31 = DNAEncoderConfig.from_state_dict(sd_v31)
    assert reconstructed_v3.sharp_motif_gate is False
    assert reconstructed_v31.sharp_motif_gate is True

    dna = CompositionNormalizedBackbone.sequence_to_one_hot("AAGGTAC" * 5)
    m_don_wide, _, _, _ = CompositionNormalizedBackbone._compute_canonical_motif_indicators(dna, sharp=False)
    m_don_sharp, _, _, _ = CompositionNormalizedBackbone._compute_canonical_motif_indicators(dna, sharp=True)
    # Position 3 is G before T ("GT"): exact motif is 1.0; +/-1 bp (pos 2) is 0.25 sharp; +2 bp (pos 5) is 0.0 sharp vs 1.0 wide
    assert torch.isclose(m_don_sharp[0, 3], torch.tensor(1.0))
    assert torch.isclose(m_don_sharp[0, 2], torch.tensor(0.25))
    assert float(m_don_sharp[0, 5]) == 0.0
    assert float(m_don_wide[0, 5]) == 1.0


def test_counterfactual_dataset_boundary_sigma():
    dna = "A" * 50 + "ATGGCTAGCTAGCTAGCTAGTA" + "C" * 50
    sample = GenomicGeneSample(dna_sequence=dna, exon_spans=[(50, 72)], strand="+")
    ds_wide = CounterfactualDataset([sample], boundary_sigma=1.0, boundary_radius=2)
    ds_sharp = CounterfactualDataset([sample], boundary_sigma=0.45, boundary_radius=1)
    item_w = ds_wide[0]
    item_s = ds_sharp[0]
    assert float(item_s["target_start"][50]) == 1.0
    assert float(item_s["target_start"][52]) == 0.0
    assert float(item_w["target_start"][52]) > 0.1


def test_canon_17mer_depths_strand_symmetry():
    repeat_unit = "ATGCTAGCTAGCTGATCGATCG"
    seq = ("AAACCCGGGTTT" * 2) + repeat_unit + "TTTTCCCC" + repeat_unit + ("GGGAAATTTCCC" * 2)
    d_fwd = _compute_canon_17mer_depths(seq)
    comp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    rc_seq = "".join(comp[b] for b in reversed(seq))
    d_rev = _compute_canon_17mer_depths(rc_seq)

    # First occurrence of repeat_unit has copy depth >= 2 on every interior 17-mer start
    idx = seq.index(repeat_unit)
    assert int(d_fwd[idx]) >= 2
    # Total count of duplicated 17-mer positions matches across forward and reverse complement
    assert int(np.sum(d_fwd >= 2)) == int(np.sum(d_rev >= 2))


def test_holdout_chroms_cover_all_benchmark_accessions():
    expected = {
        "NC_006980.1", "NC_004329.3", "NC_031232.1", "NC_004353.4", "NC_037280.1",
        "NC_001134.8", "NC_003421.2", "NC_003279.8", "NC_003070.9", "NC_057004.1",
    }
    assert expected.issubset(BENCHMARK_HOLDOUT_CHROMS)
    for acc in expected:
        assert acc.split(".")[0] in BENCHMARK_HOLDOUT_STEMS


def test_oracle_masked_profiles_batch_matches_single():
    oracle = ProteinEnergyOracle(device="cpu")
    p1 = "MKMYYLKMLLFTFLINTLVARHYENFVNNHYNVSLIQNKTKRVTIKSRLL"
    p2 = "MKTLLLTLVVVTIVCLDLGYTQLQKRSLLKSLIQNKTKRVTIKSRLLNKK"
    s1 = oracle.score_masked_profile(p1, junction_aa_indices=[10, 25])
    s2 = oracle.score_masked_profile(p2, junction_aa_indices=[15])

    # Clear cache so batch path exercises true forward pass
    oracle._collar_cache.clear()
    oracle.score_masked_profiles_batch([(p1, [10, 25], []), (p2, [15], [])], batch_size=2)
    b1 = oracle.score_masked_profile(p1, junction_aa_indices=[10, 25])
    b2 = oracle.score_masked_profile(p2, junction_aa_indices=[15])
    assert abs(s1["masked_junc_nll"] - b1["masked_junc_nll"]) < 1e-4
    assert abs(s2["masked_junc_nll"] - b2["masked_junc_nll"]) < 1e-4
