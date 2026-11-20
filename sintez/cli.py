"""
SINTEZ Command Line Interface (CLI)
===================================
Provides command line entry points for:
- sintez predict: Predict genes on a FASTA file.
- sintez benchmark: Evaluate predictions against NCBI ground-truth GFF3/FT.
"""

from __future__ import annotations
import argparse
import gzip
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .dna_backbone import CompositionNormalizedBackbone, DNAEncoderConfig
from .protein_oracle import ProteinEnergyOracle
from .translation_operator import SplicedTranslationOperator
from .dag_decoder import TopologicalDAGDecoder
from .dual_strand import DualStrandResolver
from .te_classifier import TransposonClassifier


def read_fasta_records(fa_path: Path) -> Dict[str, str]:
    """Reads a (possibly gzipped) FASTA file into an ordered dict {seqid: sequence}."""
    records: Dict[str, List[str]] = {}
    cur_id: Optional[str] = None
    opener = gzip.open if str(fa_path).endswith(".gz") else open
    with opener(fa_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                cur_id = line[1:].split()[0]
                if cur_id not in records:
                    records[cur_id] = []
            elif cur_id is not None:
                records[cur_id].append(line)
    return {k: "".join(v) for k, v in records.items()}


def run_windowed_inference(
    backbone: CompositionNormalizedBackbone,
    seq: str,
    device: torch.device,
    window_size: int = 25000,
    stride: int = 20000,
) -> Dict[str, np.ndarray]:
    """Runs sliding-window inference across long genomic scaffolds."""
    backbone.eval()
    L = len(seq)
    pad = (window_size - stride) // 2
    preds = {
        "start": np.zeros(L, dtype=np.float32),
        "stop": np.zeros(L, dtype=np.float32),
        "donor": np.zeros(L, dtype=np.float32),
        "acceptor": np.zeros(L, dtype=np.float32),
        "intergenic": np.zeros(L, dtype=np.float32),
        "intron": np.zeros(L, dtype=np.float32),
        "coding": np.zeros(L, dtype=np.float32),
        "phase": np.zeros((3, L), dtype=np.float32),
    }
    counts = np.zeros(L, dtype=np.float32)

    windows = []
    for start in range(0, L, stride):
        end = min(start + window_size, L)
        windows.append((start, end))
        if end == L:
            break

    for w_idx, (w_start, w_end) in enumerate(windows, start=1):
        w_seq = seq[w_start:w_end]
        w_len = len(w_seq)
        one_hot = CompositionNormalizedBackbone.sequence_to_one_hot(w_seq).to(device)
        with torch.no_grad():
            out = backbone(one_hot)

        span_intergenic = out["span_probs"][:, 0, :].reshape(-1).cpu().numpy()
        span_intron = out["span_probs"][:, 1, :].reshape(-1).cpu().numpy()
        span_coding = out["span_probs"][:, 2, :].reshape(-1).cpu().numpy()
        p_start = out["start_prob"].reshape(-1).cpu().numpy()
        p_stop = out["stop_prob"].reshape(-1).cpu().numpy()
        p_donor = out["donor_prob"].reshape(-1).cpu().numpy()
        p_accept = out["acceptor_prob"].reshape(-1).cpu().numpy()

        raw_sp = out.get("span_probs_raw", out["span_probs"])
        if raw_sp.shape[1] >= 7:
            p_3 = (
                raw_sp[:, 2:5, :]
                + (raw_sp[:, 5:6, :] + raw_sp[:, 6:7, :]) / 3.0
            ).squeeze(0).cpu().numpy()
        else:
            p_3 = np.repeat(span_coding[np.newaxis, :] / 3.0, 3, axis=0)

        weight = np.ones(w_len, dtype=np.float32)
        edge = min(pad, w_len // 4)
        if edge > 0:
            ramp = np.linspace(0.1, 1.0, edge, dtype=np.float32)
            if w_start > 0:
                weight[:edge] *= ramp
            if w_end < L:
                weight[-edge:] *= ramp[::-1]

        preds["start"][w_start:w_end] += p_start * weight
        preds["stop"][w_start:w_end] += p_stop * weight
        preds["donor"][w_start:w_end] += p_donor * weight
        preds["acceptor"][w_start:w_end] += p_accept * weight
        preds["intergenic"][w_start:w_end] += span_intergenic * weight
        preds["intron"][w_start:w_end] += span_intron * weight
        preds["coding"][w_start:w_end] += span_coding * weight
        preds["phase"][:, w_start:w_end] += p_3 * weight[np.newaxis, :]
        if "foldability_prob" in out:
            if "foldability" not in preds:
                preds["foldability"] = np.zeros(L, dtype=np.float32)
            preds["foldability"][w_start:w_end] += out["foldability_prob"].reshape(-1).cpu().numpy() * weight
        counts[w_start:w_end] += weight

    counts = np.maximum(counts, 1e-6)
    for k in ("start", "stop", "donor", "acceptor", "intergenic", "intron", "coding"):
        preds[k] /= counts
    preds["phase"] /= counts[np.newaxis, :]
    if "foldability" in preds:
        preds["foldability"] /= counts

    return preds


def write_gff3(
    predictions: List[Dict],
    chrom_id: str,
    out_path: Path,
    seq_regions: Optional[Dict[str, int]] = None,
):
    """Writes predicted gene models into standard GFF3 format with stratified biotype attributes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("##gff-version 3\n")
        if seq_regions:
            for sid, slen in seq_regions.items():
                f.write(f"##sequence-region {sid} 1 {slen}\n")
        else:
            f.write(f"##sequence-region {chrom_id} 1 100000000\n")
        f.write("# Generated by SINTEZ (https://github.com/veg/SINTEZ)\n")

        for g_idx, pred in enumerate(predictions, start=1):
            seqid = pred.get("seqid", chrom_id)
            gene_id = f"SINTEZ_{g_idx:05d}"
            tx_id = f"{gene_id}.1"
            spans = pred["spans"]
            strand = pred["strand"]
            g_start = spans[0][0] + 1
            g_end = spans[-1][1]
            score = pred.get("score", 0.0)

            biotype = pred.get("biotype", "protein_coding")
            gene_type = pred.get("gene_type", "host")
            te_family = pred.get("te_family", "none")

            attrs = f"ID={gene_id};Name={gene_id};biotype={biotype};gene_type={gene_type}"
            if "posterior" in pred:
                attrs += f";posterior={float(pred['posterior']):.4f}"
            if "tier" in pred:
                attrs += f";tier={pred['tier']}"
            if biotype == "transposon_derived" and te_family != "none":
                attrs += f";te_family={te_family}"

            f.write(f"{seqid}\tSINTEZ\tgene\t{g_start}\t{g_end}\t{score:.2f}\t{strand}\t.\t{attrs}\n")
            f.write(f"{seqid}\tSINTEZ\tmRNA\t{g_start}\t{g_end}\t{score:.2f}\t{strand}\t.\tID={tx_id};Parent={gene_id}\n")

            for e_idx, (e_s, e_e) in enumerate(spans, start=1):
                f.write(f"{seqid}\tSINTEZ\tCDS\t{e_s + 1}\t{e_e}\t.\t{strand}\t0\tID={tx_id}.cds{e_idx};Parent={tx_id}\n")


def _resolve_checkpoint_path(checkpoint_arg: Optional[str]) -> Path:
    models_dir = Path(__file__).parent.parent / "models"
    if checkpoint_arg:
        ckpt_path = Path(checkpoint_arg)
    elif (models_dir / "sintez_backbone_v3_multiphyla_swa.pt").exists():
        ckpt_path = models_dir / "sintez_backbone_v3_multiphyla_swa.pt"
    elif (models_dir / "sintez_backbone_v3_multiphyla.pt").exists():
        ckpt_path = models_dir / "sintez_backbone_v3_multiphyla.pt"
    else:
        ckpt_path = models_dir / "sintez_backbone_multiphyla.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
    return ckpt_path


def cmd_predict(args):
    """Executes full chromosome or multi-contig gene discovery."""
    device = torch.device(args.device if args.device else ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")))
    print(f"[*] Target Compute Device: {device.type.upper()}")

    fa_path = Path(args.fasta)
    records = read_fasta_records(fa_path)
    if not records:
        raise ValueError(f"No FASTA records found in {fa_path}")
    total_bp = sum(len(s) for s in records.values())
    print(f"[*] Loaded FASTA: {len(records)} contig(s), {total_bp:,} bp.")

    ckpt_path = _resolve_checkpoint_path(args.checkpoint)
    print(f"[+] Loading SINTEZ Backbone from {ckpt_path.resolve()}...")
    state_dict = torch.load(ckpt_path, map_location=device)
    cfg = DNAEncoderConfig.from_state_dict(state_dict)
    backbone = CompositionNormalizedBackbone(cfg).to(device)
    backbone.load_state_dict(state_dict)
    backbone.eval()

    oracle_dev = "cpu" if device.type == "mps" else str(device)
    oracle = ProteinEnergyOracle(device=oracle_dev)
    decoder = TopologicalDAGDecoder(
        lambda_intron=args.lambda_intron,
        max_exons=args.max_exons,
        energy_cutoff=args.energy_cutoff,
    )

    t0 = time.time()
    op = SplicedTranslationOperator()
    all_preds: List[Dict] = []
    all_te_preds: List[Dict] = []
    seq_regions: Dict[str, int] = {}

    for rec_idx, (seqid, chrom_seq) in enumerate(records.items()):
        out_seqid = args.chrom_id if (args.chrom_id and len(records) == 1) else seqid
        L = len(chrom_seq)
        seq_regions[out_seqid] = L
        rc_seq = op.code.reverse_complement(chrom_seq)

        print(f"[*] Contig {rec_idx + 1}/{len(records)} {out_seqid} ({L:,} bp): Neural Inference (+) & (-)...")
        fwd_preds = run_windowed_inference(backbone, chrom_seq, device)
        rev_preds = run_windowed_inference(backbone, rc_seq, device)

        fwd_genes_raw = decoder.decode_strand(
            chrom_seq, fwd_preds, oracle, strand_label="+",
            antisense_coding=rev_preds["coding"][::-1],
        )
        for g in fwd_genes_raw:
            g["strand"] = "+"
            g["seqid"] = out_seqid

        rev_genes_raw = decoder.decode_strand(
            rc_seq, rev_preds, oracle, strand_label="-",
            antisense_coding=fwd_preds["coding"][::-1],
        )
        rev_genes = []
        for g in rev_genes_raw:
            g["spans"] = [(L - e, L - s) for s, e in reversed(g["spans"])]
            g["strand"] = "-"
            g["seqid"] = out_seqid
            rev_genes.append(g)

        contig_preds = DualStrandResolver.resolve_conflicts(fwd_genes_raw + rev_genes, max_reciprocal_overlap=0.15)
        if args.stratify_te and contig_preds:
            te_classifier = TransposonClassifier(genome_seq=chrom_seq)
            host_preds, te_preds = te_classifier.stratify_predictions(contig_preds, chrom_seq)
            all_te_preds.extend(te_preds)
            contig_preds = host_preds if args.exclude_transposons else contig_preds
        all_preds.extend(contig_preds)

    print(f"[+] Final Discovered Gene Models: {len(all_preds)} in {time.time()-t0:.1f}s")
    if args.transposon_out and args.stratify_te:
        te_out = Path(args.transposon_out)
        write_gff3(all_te_preds, chrom_id=args.chrom_id or next(iter(seq_regions)), out_path=te_out, seq_regions=seq_regions)
        print(f"[+] Transposon models ({len(all_te_preds)}) written to: {te_out.resolve()}")

    out_gff = Path(args.out)
    write_gff3(all_preds, chrom_id=args.chrom_id or next(iter(seq_regions)), out_path=out_gff, seq_regions=seq_regions)
    print(f"[+] Output written to: {out_gff.resolve()}")


def cmd_export_signals(args):
    from .signal_exporter import export_signals_for_fasta

    summary = export_signals_for_fasta(
        fasta_path=args.fasta,
        out_dir=args.out_dir,
        ckpt_path=args.ckpt,
        device=args.device,
        run_decoder=not args.fast_tracks_only,
    )
    print(
        f"[+] Exported SINTEZ signals to {summary['out_dir']} "
        f"({len(summary['manifest']['contigs'])} contigs, "
        f"{summary['n_genes']} genes, {summary['n_exons']} exons, {summary['n_introns']} introns)"
    )


def main():
    parser = argparse.ArgumentParser(description="SINTEZ: Structural Inference by Neural Translation & Energy Zero-shot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # predict
    p_pred = subparsers.add_parser("predict", help="Predict gene models on genomic FASTA")
    p_pred.add_argument("--fasta", required=True, help="Input genomic FASTA file")
    p_pred.add_argument("--out", default="predictions.gff3", help="Output GFF3 file path")
    p_pred.add_argument("--chrom_id", default=None, help="Override chromosome ID for single-contig GFF3 output (default: FASTA header ID)")
    p_pred.add_argument("--checkpoint", default=None, help="Path to backbone .pt weights")
    p_pred.add_argument("--device", default=None, help="Compute device ('cuda', 'mps', 'cpu')")
    p_pred.add_argument("--lambda_intron", type=float, default=4.5, help="Splicing log-odds penalty (default: 4.5)")
    p_pred.add_argument("--max_exons", type=int, default=16, help="Maximum exon depth in DAG (default: 16)")
    p_pred.add_argument("--energy_cutoff", type=float, default=13.0, help="ESM-2 free energy cutoff (default: 13.0)")
    p_pred.add_argument("--stratify_te", action="store_true", default=True, help="Stratify output into host vs transposon-derived genes (default: True)")
    p_pred.add_argument("--no_stratify_te", action="store_false", dest="stratify_te", help="Disable TE classification")
    p_pred.add_argument("--exclude_transposons", action="store_true", default=False, help="Exclude transposon-derived genes from primary GFF3 output")
    p_pred.add_argument("--transposon_out", default=None, help="Optional separate output GFF3 file for transposon-derived genes")
    p_pred.set_defaults(func=cmd_predict)

    # export-signals
    p_exp = subparsers.add_parser("export-signals", help="Export SINTEZ-SIGNALS/1.0 dense tracks and TSVs for KOSHKA S5")
    p_exp.add_argument("--fasta", required=True, help="Input genomic FASTA file")
    p_exp.add_argument("--out_dir", required=True, help="Output signal bundle directory")
    p_exp.add_argument("--ckpt", default=None, help="Path to backbone .pt weights")
    p_exp.add_argument("--device", default=None, help="Compute device ('cuda', 'mps', 'cpu')")
    p_exp.add_argument("--fast_tracks_only", action="store_true", default=False, help="Export dense tracks without running DAG+ESM2 decoder")
    p_exp.set_defaults(func=cmd_export_signals)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
