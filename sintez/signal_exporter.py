"""
SINTEZ Signal Exporter (`SINTEZ-SIGNALS/1.0`)
=============================================
Exports dense strand-oriented neural structural tracks (`float16` `.npz`),
sparse candidate exon/intron log-likelihood tables (`exon_scores.tsv`,
`intron_scores.tsv`), and decoded SINTEZ v3.2 gene models (`genes_sintez.gff3`)
for downstream integrative tools such as KOSHKA (`S5`).
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .cli import _resolve_checkpoint_path, read_fasta_records, run_windowed_inference, write_gff3
from .dag_decoder import TopologicalDAGDecoder
from .dna_backbone import CompositionNormalizedBackbone, DNAEncoderConfig
from .dual_strand import DualStrandResolver
from .protein_oracle import ProteinEnergyOracle

_REVCOMP_TRANS = str.maketrans("ACGTNacgtn", "TGCANtgcan")
_EXPORT_TRACK_KEYS = (
    "start",
    "stop",
    "donor",
    "acceptor",
    "coding",
    "intron",
    "intergenic",
    "foldability",
)


def _revcomp(seq: str) -> str:
    return seq.translate(_REVCOMP_TRANS)[::-1]


def _safe_seqid(seqid: str, used: set) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", seqid).strip("._") or "contig"
    cand = base
    idx = 2
    while cand in used:
        cand = f"{base}_{idx}"
        idx += 1
    used.add(cand)
    return cand


def _resolve_device(device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_prob_cache(npz_path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    data = np.load(npz_path)
    fwd: Dict[str, np.ndarray] = {}
    rev: Dict[str, np.ndarray] = {}
    for k in data.files:
        if k.startswith("fwd_"):
            fwd[k[4:]] = data[k].astype(np.float32)
        elif k.startswith("rev_"):
            rev[k[4:]] = data[k].astype(np.float32)
    return fwd, rev


def export_signals_for_fasta(
    fasta_path: str | Path,
    out_dir: str | Path,
    ckpt_path: Optional[str | Path] = None,
    device: Optional[str] = None,
    run_decoder: bool = True,
    prob_cache_map: Optional[Dict[str, str | Path]] = None,
    oracle: Optional[ProteinEnergyOracle] = None,
) -> Dict[str, object]:
    """Export SINTEZ-SIGNALS/1.0 bundle for every contig in `fasta_path`."""
    fasta_p = Path(fasta_path)
    out_p = Path(out_dir)
    tracks_dir = out_p / "tracks"
    docs_dir = out_p / "docs"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    spec_src = Path(__file__).resolve().parent.parent / "docs" / "SIGNALS.md"
    spec_dst = docs_dir / "SIGNALS.md"
    if spec_src.exists():
        shutil.copy2(spec_src, spec_dst)
    else:
        spec_dst.write_text("# SINTEZ-SIGNALS/1.0\n", encoding="utf-8")

    records = read_fasta_records(str(fasta_p))
    if not records:
        raise ValueError(f"No FASTA sequences found in {fasta_p}")

    prob_cache_map = prob_cache_map or {}
    need_backbone = any(
        seqid not in prob_cache_map or not Path(prob_cache_map[seqid]).exists()
        for seqid in records
    )

    dev = _resolve_device(device)
    model = None
    if need_backbone:
        ckpt_resolved = _resolve_checkpoint_path(str(ckpt_path) if ckpt_path else None)
        state = torch.load(ckpt_resolved, map_location=dev)
        state_dict = (
            state["model_state_dict"]
            if isinstance(state, dict) and "model_state_dict" in state
            else state
        )
        cfg = DNAEncoderConfig.from_state_dict(state_dict)
        model = CompositionNormalizedBackbone(cfg).to(dev)
        model.load_state_dict(state_dict)
        model.eval()

    decoder = TopologicalDAGDecoder() if run_decoder else None
    if run_decoder and oracle is None:
        oracle_dev = "cpu" if dev.type == "mps" else str(dev)
        oracle = ProteinEnergyOracle(device=oracle_dev)

    manifest_contigs: Dict[str, Dict[str, object]] = {}
    exon_rows: List[Tuple[str, str, int, int, float, str]] = []
    intron_rows: List[Tuple[str, str, int, int, float]] = []
    all_preds: List[Dict] = []
    seq_lengths: Dict[str, int] = {}
    used_safe_ids = set()

    for seqid, seq in records.items():
        L = len(seq)
        seq_lengths[seqid] = L
        safe_id = _safe_seqid(seqid, used_safe_ids)

        cache_file = Path(prob_cache_map[seqid]) if seqid in prob_cache_map else None
        if cache_file is not None and cache_file.exists():
            fwd_preds, rev_preds = _load_prob_cache(cache_file)
        else:
            rev_seq = _revcomp(seq)
            fwd_preds = run_windowed_inference(model, seq, dev)
            rev_preds = run_windowed_inference(model, rev_seq, dev)

        track_arrays = {}
        for k in _EXPORT_TRACK_KEYS:
            if k in fwd_preds:
                track_arrays[f"fwd_{k}"] = np.asarray(fwd_preds[k], dtype=np.float16)
            if k in rev_preds:
                track_arrays[f"rev_{k}"] = np.asarray(rev_preds[k], dtype=np.float16)

        rel_track = f"tracks/{safe_id}.npz"
        np.savez_compressed(out_p / rel_track, **track_arrays)
        manifest_contigs[seqid] = {"length": L, "track_file": rel_track}

        if run_decoder and decoder is not None and oracle is not None:
            rev_seq = _revcomp(seq)
            fwd_col: Dict = {}
            rev_col: Dict = {}
            fwd_genes = decoder.decode_strand(
                seq,
                fwd_preds,
                oracle,
                strand_label="+",
                antisense_coding=rev_preds["coding"][::-1],
                export_collector=fwd_col,
            )
            for g in fwd_genes:
                g["strand"] = "+"
                g["seqid"] = seqid

            rev_genes_raw = decoder.decode_strand(
                rev_seq,
                rev_preds,
                oracle,
                strand_label="-",
                antisense_coding=fwd_preds["coding"][::-1],
                export_collector=rev_col,
            )
            rev_genes: List[Dict] = []
            for g in rev_genes_raw:
                g["spans"] = [(L - e, L - s) for s, e in reversed(g["spans"])]
                if "exon_posteriors" in g:
                    g["exon_posteriors"] = list(reversed(g["exon_posteriors"]))
                g["strand"] = "-"
                g["seqid"] = seqid
                rev_genes.append(g)

            resolved = DualStrandResolver.resolve_conflicts(
                fwd_genes + rev_genes, max_reciprocal_overlap=0.15
            )
            all_preds.extend(resolved)

            best_exons: Dict[Tuple[str, int, int, str], float] = {}
            for (es, ee, etype), val in fwd_col.get("exon_scores", {}).items():
                k = ("+", int(es), int(ee), etype)
                if k not in best_exons or val > best_exons[k]:
                    best_exons[k] = float(val)
            for (es_rc, ee_rc, etype), val in rev_col.get("exon_scores", {}).items():
                gs, ge = int(L - ee_rc), int(L - es_rc)
                k = ("-", gs, ge, etype)
                if k not in best_exons or val > best_exons[k]:
                    best_exons[k] = float(val)
            for (st, gs, ge, etype), val in sorted(best_exons.items()):
                exon_rows.append((seqid, st, gs, ge, val, etype))

            best_introns: Dict[Tuple[str, int, int], float] = {}
            for (d_pos, a_pos), val in fwd_col.get("intron_scores", {}).items():
                k = ("+", int(d_pos), int(a_pos))
                if k not in best_introns or val > best_introns[k]:
                    best_introns[k] = float(val)
            for (d_rc, a_rc), val in rev_col.get("intron_scores", {}).items():
                gs, ge = int(L - a_rc), int(L - d_rc)
                k = ("-", gs, ge)
                if k not in best_introns or val > best_introns[k]:
                    best_introns[k] = float(val)
            for (st, gs, ge), val in sorted(best_introns.items()):
                intron_rows.append((seqid, st, gs, ge, val))

    manifest = {
        "schema": "SINTEZ-SIGNALS/1.0",
        "contigs": manifest_contigs,
    }
    (out_p / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with open(out_p / "exon_scores.tsv", "w", encoding="utf-8") as fh:
        fh.write("#seqid\tstrand\tstart0\tend\tllr\ttype\n")
        for seqid, st, gs, ge, val, etype in exon_rows:
            fh.write(f"{seqid}\t{st}\t{gs}\t{ge}\t{val:.4f}\t{etype}\n")

    with open(out_p / "intron_scores.tsv", "w", encoding="utf-8") as fh:
        fh.write("#seqid\tstrand\tstart0\tend\tllr\n")
        for seqid, st, gs, ge, val in intron_rows:
            fh.write(f"{seqid}\t{st}\t{gs}\t{ge}\t{val:.4f}\n")

    n_genes = 0
    if run_decoder:
        n_genes = len(all_preds)
        first_seqid = next(iter(seq_lengths))
        write_gff3(
            all_preds,
            chrom_id=first_seqid,
            out_path=out_p / "genes_sintez.gff3",
            seq_regions=seq_lengths,
        )

    return {
        "out_dir": str(out_p),
        "manifest": manifest,
        "n_genes": n_genes,
        "n_exons": len(exon_rows),
        "n_introns": len(intron_rows),
    }
