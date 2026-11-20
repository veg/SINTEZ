"""Tests for SINTEZ v3.2 KOSHKA-inspired refinements and SINTEZ-SIGNALS/1.0 exporter."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sintez.dag_decoder import TopologicalDAGDecoder
from sintez.protein_oracle import ProteinEnergyOracle
from sintez.signal_exporter import _revcomp, export_signals_for_fasta


def _build_synthetic_gene_contig():
    # Two-exon '+' gene: ATG (120..123) ... GT (240..242) ... AG (358..360) ... TAA (477..480)
    rng = np.random.default_rng(7)
    L = 700
    dna = list("".join(rng.choice(["A", "C", "G", "T"], size=L)))
    codons = ["ATG", "GCT", "GAC", "TTC", "GGC", "AAA", "CTG", "GTT", "ACC", "GAG"]
    # Exon 1: [120, 240) = 120 bp (40 codons)
    ex1 = "".join(codons[i % len(codons)] for i in range(40))
    ex1 = "ATG" + ex1[3:]
    # Exon 2: [360, 480) = 120 bp (39 codons + TAA)
    ex2 = "".join(codons[(i + 3) % len(codons)] for i in range(39)) + "TAA"
    for i, ch in enumerate(ex1):
        dna[120 + i] = ch
    dna[240], dna[241] = "G", "T"
    dna[358], dna[359] = "A", "G"
    for i, ch in enumerate(ex2):
        dna[360 + i] = ch
    seq = "".join(dna)
    return seq, L


def _build_preds_for_spans(L: int, exons):
    preds = {
        "start": np.full(L, 0.0005, dtype=np.float32),
        "stop": np.full(L, 0.0005, dtype=np.float32),
        "donor": np.full(L, 0.0005, dtype=np.float32),
        "acceptor": np.full(L, 0.0005, dtype=np.float32),
        "coding": np.full(L, 0.03, dtype=np.float32),
        "intron": np.full(L, 0.05, dtype=np.float32),
        "intergenic": np.full(L, 0.92, dtype=np.float32),
        "foldability": np.full(L, 0.05, dtype=np.float32),
    }
    if exons:
        preds["start"][exons[0][0]] = 0.96
        preds["stop"][exons[-1][1] - 3] = 0.96
    for i, (s, e) in enumerate(exons):
        preds["coding"][s:e] = 0.95
        preds["intergenic"][s:e] = 0.03
        preds["intron"][s:e] = 0.02
        preds["foldability"][s:e] = 0.90
        if i + 1 < len(exons):
            ns = exons[i + 1][0]
            preds["donor"][e] = 0.95
            preds["acceptor"][ns - 1] = 0.95
            preds["intron"][e:ns] = 0.90
            preds["intergenic"][e:ns] = 0.08
    return preds


def test_globular_core_low_complexity_exemption():
    oracle = ProteinEnergyOracle(device="cpu")
    # Diverse globular core (55 aa) + long poly-N/S low-complexity tail (45 aa) -> total 100 aa
    core = "MKMYYLKMLLFTFLINTLVARHYENFVNNHYNVSLIQNKTKRVTIKSRLLHGDWP"
    tail = "NNNNNNNNNNSSSSSNNNNNNNNNNSSSSSNNNNNNNNNNSSSSS"
    chimera = core + tail
    pure_repeat = "MLTDMMTDITTNITTNLTTNNITHMTTNIITNIITNMVTNMITNMTTYMTTNMTTNIIINMTTNIIITNMITNISTN"

    lc_mask = oracle.low_complexity_mask(chimera)
    assert sum(lc_mask[-35:]) >= 30
    assert oracle.globular_core_entropy(chimera, window=45) >= 3.25
    assert oracle.globular_core_entropy(pure_repeat, window=45) < 3.25

    res_chimera = oracle.score_sequence(chimera)
    res_repeat = oracle.score_sequence(pure_repeat)
    assert res_chimera["oracle_schema"] == 2
    assert res_chimera["has_globular_core"] is True
    assert res_chimera["entropy_penalty"] == 0.0
    assert res_repeat["has_globular_core"] is False
    assert res_repeat["entropy_penalty"] > 3.0


def test_decoder_calibrated_posteriors_and_collector():
    seq, L = _build_synthetic_gene_contig()
    preds = _build_preds_for_spans(L, [(120, 240), (360, 480)])
    oracle = ProteinEnergyOracle(device="cpu")
    decoder = TopologicalDAGDecoder()
    collector = {}
    genes = decoder.decode_strand(seq, preds, oracle, strand_label="+", export_collector=collector)
    assert len(genes) == 1
    g = genes[0]
    assert g["spans"] == [(120, 240), (360, 480)]
    assert 0.0 < g["posterior_pass1"] <= 1.0
    assert 0.0 < g["posterior"] <= 1.0
    assert g["tier"] in ("high", "medium", "low")
    assert len(g["exon_posteriors"]) == 2
    assert all(0.0 < p <= 1.0 for p in g["exon_posteriors"])
    assert (120, 240, "initial") in collector["exon_scores"]
    assert (360, 480, "terminal") in collector["exon_scores"]
    assert (240, 360) in collector["intron_scores"]
    assert collector["intron_scores"][(240, 360)] > 0.0


def test_export_signals_dual_strand_roundtrip(tmp_path: Path):
    seq_fwd, L = _build_synthetic_gene_contig()
    # Embed the gene on '-' strand on contig_2 by taking revcomp of seq_fwd
    seq_rev_contig = _revcomp(seq_fwd)
    fasta = tmp_path / "test_dual.fa"
    fasta.write_text(f">ctg_plus desc\n{seq_fwd}\n>ctg_minus desc\n{seq_rev_contig}\n")

    fwd_preds_1 = _build_preds_for_spans(L, [(120, 240), (360, 480)])
    rev_preds_1 = _build_preds_for_spans(L, [])
    cache_1 = tmp_path / "c1.npz"
    np.savez(
        cache_1,
        **{f"fwd_{k}": v for k, v in fwd_preds_1.items()},
        **{f"rev_{k}": v for k, v in rev_preds_1.items()},
    )

    fwd_preds_2 = _build_preds_for_spans(L, [])
    rev_preds_2 = _build_preds_for_spans(L, [(120, 240), (360, 480)])
    cache_2 = tmp_path / "c2.npz"
    np.savez(
        cache_2,
        **{f"fwd_{k}": v for k, v in fwd_preds_2.items()},
        **{f"rev_{k}": v for k, v in rev_preds_2.items()},
    )

    out_dir = tmp_path / "signals_out"
    res = export_signals_for_fasta(
        fasta,
        out_dir,
        run_decoder=True,
        prob_cache_map={"ctg_plus": cache_1, "ctg_minus": cache_2},
    )
    assert res["n_genes"] == 2
    assert (out_dir / "docs" / "SIGNALS.md").exists()
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["schema"] == "SINTEZ-SIGNALS/1.0"
    assert set(manifest["contigs"].keys()) == {"ctg_plus", "ctg_minus"}
    assert (out_dir / manifest["contigs"]["ctg_plus"]["track_file"]).exists()

    ex_lines = [
        l.split("\t")
        for l in (out_dir / "exon_scores.tsv").read_text().splitlines()
        if l and not l.startswith("#")
    ]
    in_lines = [
        l.split("\t")
        for l in (out_dir / "intron_scores.tsv").read_text().splitlines()
        if l and not l.startswith("#")
    ]
    ex_keys = {(r[0], r[1], int(r[2]), int(r[3])) for r in ex_lines}
    in_keys = {(r[0], r[1], int(r[2]), int(r[3])) for r in in_lines}
    # Forward contig: [120, 240) and [360, 480), intron [240, 360)
    assert ("ctg_plus", "+", 120, 240) in ex_keys
    assert ("ctg_plus", "+", 360, 480) in ex_keys
    assert ("ctg_plus", "+", 240, 360) in in_keys
    # Reverse contig (700 - b_rc, 700 - a_rc): [460, 580) and [220, 340), intron [340, 460)
    assert ("ctg_minus", "-", 220, 340) in ex_keys
    assert ("ctg_minus", "-", 460, 580) in ex_keys
    assert ("ctg_minus", "-", 340, 460) in in_keys

    gff_text = (out_dir / "genes_sintez.gff3").read_text()
    assert "posterior=" in gff_text and "tier=" in gff_text


def test_export_signals_cli_smoke(tmp_path: Path):
    ckpt = Path(__file__).resolve().parent.parent / "models" / "sintez_backbone_v3_multiphyla_swa.pt"
    if not ckpt.exists():
        pytest.skip("Backbone checkpoint not present")
    seq, _ = _build_synthetic_gene_contig()
    fasta = tmp_path / "cli_smoke.fa"
    fasta.write_text(f">chr_smoke\n{seq}\n")
    out_dir = tmp_path / "cli_sig"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "sintez.cli",
            "export-signals",
            "--fasta",
            str(fasta),
            "--out_dir",
            str(out_dir),
            "--ckpt",
            str(ckpt),
            "--device",
            "cpu",
            "--fast_tracks_only",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Exported SINTEZ signals" in proc.stdout
    assert (out_dir / "docs" / "SIGNALS.md").exists()
    assert (out_dir / "manifest.json").exists()
    npz = np.load(out_dir / "tracks" / "chr_smoke.npz")
    assert "fwd_intron" in npz.files and "rev_intergenic" in npz.files
    assert npz["fwd_coding"].dtype == np.float16
