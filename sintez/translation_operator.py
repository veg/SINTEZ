"""
Translation Operator & Genetic Grammar Engine
=============================================
Handles discrete and continuous mapping from genomic nucleotide space
to proteomic amino acid space across reading frames, splicing junctions,
and counterfactual frame perturbations.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union
import torch
import torch.nn as nn
import numpy as np


STANDARD_CODON_TABLE: Dict[str, str] = {
    # Phenylalanine
    "TTT": "F", "TTC": "F",
    # Leucine
    "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    # Isoleucine
    "ATT": "I", "ATC": "I", "ATA": "I",
    # Methionine (Start)
    "ATG": "M",
    # Valine
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    # Serine
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "AGT": "S", "AGC": "S",
    # Proline
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    # Threonine
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    # Alanine
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    # Tyrosine
    "TAT": "Y", "TAC": "Y",
    # Histidine
    "CAT": "H", "CAC": "H",
    # Glutamine
    "CAA": "Q", "CAG": "Q",
    # Asparagine
    "AAT": "N", "AAC": "N",
    # Lysine
    "AAA": "K", "AAG": "K",
    # Aspartate
    "GAT": "D", "GAC": "D",
    # Glutamate
    "GAA": "E", "GAG": "E",
    # Cysteine
    "TGT": "C", "TGC": "C",
    # Tryptophan
    "TGG": "W",
    # Arginine
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R", "AGG": "R",
    # Glycine
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
    # Stop Codons
    "TAA": "*", "TAG": "*", "TGA": "*"
}

STOP_CODONS = {"TAA", "TAG", "TGA"}
START_CODONS = {"ATG"}


@dataclass
class SplicedTranscript:
    """Represents a candidate or true spliced transcript structure."""
    exon_spans: List[Tuple[int, int]]  # 0-indexed [start, end) intervals
    strand: str = "+"                  # '+' or '-'
    mrna_sequence: str = ""
    protein_sequence: str = ""
    has_valid_start: bool = False
    has_valid_stop: bool = False
    has_internal_stop: bool = False
    is_in_frame: bool = False
    coding_length: int = 0


class GeneticCode:
    """Utilities for codon translation and reading frame operations."""
    
    @staticmethod
    def translate_dna(seq: str) -> str:
        """Translates a DNA sequence into an amino acid sequence."""
        seq = seq.upper().replace("U", "T")
        pep = []
        for i in range(0, len(seq) - 2, 3):
            codon = seq[i:i+3]
            aa = STANDARD_CODON_TABLE.get(codon, "X")
            pep.append(aa)
        return "".join(pep)

    @staticmethod
    def reverse_complement(seq: str) -> str:
        """Computes reverse complement of a DNA sequence."""
        comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N",
                "a": "t", "t": "a", "c": "g", "g": "c", "n": "n"}
        return "".join(comp.get(base, "N") for base in reversed(seq))


class SplicedTranslationOperator:
    """
    Extracts, stitches, and translates candidate genomic intervals into protein space.
    Evaluates coding frame continuity and creates counterfactual negative paths.
    """
    def __init__(self):
        self.code = GeneticCode()

    def process_transcript(self, genomic_seq: str, exon_spans: List[Tuple[int, int]], strand: str = "+") -> SplicedTranscript:
        """
        Slices exon spans, stitches them, and translates into protein space.
        
        Args:
            genomic_seq: Full genomic DNA string.
            exon_spans: Sorted list of [start, end) intervals.
            strand: '+' for forward, '-' for reverse complement.
        """
        # Slice and concatenate exons
        exon_seqs = [genomic_seq[start:end] for start, end in exon_spans]
        mrna = "".join(exon_seqs).upper().replace("U", "T")
        
        if strand == "-":
            mrna = self.code.reverse_complement(mrna)
            
        coding_len = len(mrna)
        is_in_frame = (coding_len % 3 == 0) and (coding_len >= 9)
        
        if not is_in_frame:
            # Truncate to multiple of 3 for evaluation purposes
            eval_mrna = mrna[: (coding_len // 3) * 3]
        else:
            eval_mrna = mrna
            
        protein = self.code.translate_dna(eval_mrna) if len(eval_mrna) >= 3 else ""
        
        # Check start & stop codon semantics
        has_valid_start = len(eval_mrna) >= 3 and eval_mrna[:3] in START_CODONS
        has_valid_stop = len(eval_mrna) >= 3 and eval_mrna[-3:] in STOP_CODONS
        
        # Internal stop: contains '*' anywhere before the final amino acid
        has_internal_stop = False
        if len(protein) > 1 and "*" in protein[:-1]:
            has_internal_stop = True
        elif len(protein) == 1 and protein[0] == "*":
            has_internal_stop = True

        return SplicedTranscript(
            exon_spans=exon_spans,
            strand=strand,
            mrna_sequence=mrna,
            protein_sequence=protein,
            has_valid_start=has_valid_start,
            has_valid_stop=has_valid_stop,
            has_internal_stop=has_internal_stop,
            is_in_frame=is_in_frame,
            coding_length=coding_len,
        )

    def generate_counterfactual_negatives(
        self,
        genomic_seq: str,
        true_transcript: SplicedTranscript,
        max_negatives: int = 4
    ) -> List[Tuple[str, SplicedTranscript]]:
        """
        Synthesizes hard counterfactual negative variants of a true gene structure:
        1. Frameshift +1: Shifts splice donor by +1 bp (disrupts reading frame).
        2. Frameshift +2: Shifts splice donor by +2 bp (disrupts reading frame).
        3. Exon Skipping: Drops an internal exon.
        4. Decoy Intron Expansion: Shifts junction into adjacent cryptic motif.
        
        Returns:
            List of (negative_type, SplicedTranscript) tuples.
        """
        negatives = []
        spans = list(true_transcript.exon_spans)
        
        if len(spans) >= 2:
            # 1. Frameshift +1 on first intron boundary
            s0, e0 = spans[0]
            if e0 + 1 < len(genomic_seq):
                perturbed_spans_1 = [(s0, e0 + 1)] + spans[1:]
                neg1 = self.process_transcript(genomic_seq, perturbed_spans_1, true_transcript.strand)
                negatives.append(("frameshift_plus_1", neg1))
                
            # 2. Frameshift +2
            if e0 + 2 < len(genomic_seq):
                perturbed_spans_2 = [(s0, e0 + 2)] + spans[1:]
                neg2 = self.process_transcript(genomic_seq, perturbed_spans_2, true_transcript.strand)
                negatives.append(("frameshift_plus_2", neg2))
                
        if len(spans) >= 3:
            # 3. Exon Skip (drop middle exon)
            skip_spans = [spans[0]] + spans[2:]
            neg_skip = self.process_transcript(genomic_seq, skip_spans, true_transcript.strand)
            negatives.append(("exon_skip", neg_skip))
            
        # 4. Cryptic Decoy Shift (shift by 3 bp, in-frame but alters amino acids)
        if len(spans) >= 2:
            s0, e0 = spans[0]
            if e0 + 3 < len(genomic_seq):
                inframe_shift_spans = [(s0, e0 + 3)] + spans[1:]
                neg_shift = self.process_transcript(genomic_seq, inframe_shift_spans, true_transcript.strand)
                negatives.append(("inframe_cryptic_shift", neg_shift))
                
        return negatives[:max_negatives]
