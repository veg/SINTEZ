"""
СИНТЕЗ (SINTEZ)
===============
Structural Inference by Neural Translation and Energy Zero-shot.

A non-HMM dual-space gene discovery architecture combining:
1. A composition-normalized CNN/BiGRU DNA backbone (238k parameters).
2. A virtual spliced translation operator with canonical reading-frame tracking.
3. A frozen Meta ESM-2 biophysical protein energy oracle with Shannon complexity normalization.
4. A Pareto-frontier topological splicing DAG decoder with Jacobson-Stockmayer polymer looping prior.
5. A global dual-strand conflict resolver.
"""

__version__ = "1.0.0"
__author__ = "VEG Lab"
__license__ = "MIT"

from .dna_backbone import CompositionNormalizedBackbone, DNAEncoderConfig
from .protein_oracle import ProteinEnergyOracle
from .translation_operator import SplicedTranslationOperator, SplicedTranscript
from .dag_decoder import TopologicalDAGDecoder
from .dual_strand import DualStrandResolver

__all__ = [
    "CompositionNormalizedBackbone",
    "DNAEncoderConfig",
    "ProteinEnergyOracle",
    "SplicedTranslationOperator",
    "SplicedTranscript",
    "TopologicalDAGDecoder",
    "DualStrandResolver",
]
