"""
Composition-Invariant DNA Neural Backbone
=========================================
A PyTorch convolutional-recurrent architecture designed for zero-shot
cross-phyla generalization. Incorporates running compositional normalization
to insulate feature learning from extreme AT/GC skews.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DNAEncoderConfig:
    in_channels: int = 4            # A, C, G, T
    hidden_dim: int = 128
    kernel_size: int = 7
    dilations: Tuple[int, ...] = (1, 3, 9, 27)
    norm_window_size: int = 501     # Single running background GC/AT normalization window
    norm_window_sizes: Optional[Tuple[int, ...]] = None  # Multi-scale windows (e.g. 101, 501, 2001)
    num_span_classes: int = 3       # 3 (Intergenic, Intron, Exon) or 7 (Phase-aware)
    dropout: float = 0.1
    use_gru: bool = True            # Classic v1 uses BiGRU; Foundation v2 uses pure dilated conv

    @classmethod
    def classic_v1(cls, hidden_dim: int = 64, norm_window_size: int = 201) -> DNAEncoderConfig:
        """Configuration matching the original 238,407-parameter v1 backbone."""
        return cls(
            in_channels=4,
            hidden_dim=hidden_dim,
            kernel_size=7,
            dilations=(1, 3, 9, 27),
            norm_window_size=norm_window_size,
            norm_window_sizes=None,
            num_span_classes=3,
            dropout=0.1,
            use_gru=True,
        )

    @classmethod
    def foundation_v2(
        cls,
        hidden_dim: int = 128,
        dilations: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128),
        norm_window_sizes: Tuple[int, ...] = (101, 501, 2001),
        num_span_classes: int = 7,
        dropout: float = 0.1,
        use_gru: bool = False,
    ) -> DNAEncoderConfig:
        """Foundation-grade v2 architecture with deep receptive field and multi-scale normalization."""
        return cls(
            in_channels=4,
            hidden_dim=hidden_dim,
            kernel_size=7,
            dilations=dilations,
            norm_window_size=501,
            norm_window_sizes=norm_window_sizes,
            num_span_classes=num_span_classes,
            dropout=dropout,
            use_gru=use_gru,
        )


class LocalCompositionNormalization(nn.Module):
    """
    Subtracts the local background nucleotide distribution using an average pooling window.
    This prevents the neural network from overfitting to clade-specific GC/AT content
    (e.g., human ~41% GC vs Plasmodium ~19% GC / 81% AT).
    """
    def __init__(self, channels: int = 4, window_size: int = 501):
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.pad = window_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Channels, Length)
        Returns:
            Normalized tensor where local compositional mean is subtracted.
        """
        padded = F.pad(x, (self.pad, self.pad), mode="replicate")
        local_mean = F.avg_pool1d(padded, kernel_size=self.window_size, stride=1)
        return x - local_mean


class MultiScaleCompositionNormalization(nn.Module):
    """
    Computes local composition normalization across multiple window scales (e.g. 101, 501, 2001 bp).
    Simultaneously isolates local codon bias, regional gene composition, and whole-isochore shifts.
    """
    def __init__(self, channels: int = 4, window_sizes: Tuple[int, ...] = (101, 501, 2001)):
        super().__init__()
        self.channels = channels
        self.window_sizes = window_sizes
        self.norms = nn.ModuleList([
            LocalCompositionNormalization(channels, w) for w in window_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns concatenated normalized features across all scales.
        Shape: (Batch, channels * len(window_sizes), Length)
        """
        features = [norm(x) for norm in self.norms]
        return torch.cat(features, dim=1)


class DilatedConvBlock(nn.Module):
    """Multi-scale residual dilated convolution block."""
    def __init__(self, channels: int, dilation: int, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.act2 = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.drop(x)
        x = self.act2(self.norm2(self.conv2(x)))
        return x + res


class CompositionNormalizedBackbone(nn.Module):
    """
    Deep neural DNA backbone with composition normalization, multi-scale
    dilated convolutions, bidirectional sequence context, and multi-task junction heads.
    """
    def __init__(self, cfg: Optional[DNAEncoderConfig] = None):
        super().__init__()
        self.cfg = cfg or DNAEncoderConfig()
        
        # 1. Composition Normalizer (Single or Multi-scale)
        if self.cfg.norm_window_sizes is not None and len(self.cfg.norm_window_sizes) > 0:
            self.comp_norm = MultiScaleCompositionNormalization(
                channels=self.cfg.in_channels,
                window_sizes=self.cfg.norm_window_sizes,
            )
            in_channels_total = self.cfg.in_channels * (1 + len(self.cfg.norm_window_sizes))
        else:
            self.comp_norm = LocalCompositionNormalization(
                channels=self.cfg.in_channels,
                window_size=self.cfg.norm_window_size,
            )
            in_channels_total = self.cfg.in_channels * 2

        # 2. Input Embedding
        self.in_proj = nn.Conv1d(in_channels_total, self.cfg.hidden_dim, kernel_size=5, padding=2)

        # 3. Stack of Multi-scale Dilated Convolutions
        self.dilated_blocks = nn.ModuleList([
            DilatedConvBlock(self.cfg.hidden_dim, dilation=d, kernel_size=self.cfg.kernel_size, dropout=self.cfg.dropout)
            for d in self.cfg.dilations
        ])

        # 4. Bidirectional Sequence Context (optional, disabled in foundation_v2 for 150x GPU acceleration)
        if self.cfg.use_gru:
            self.bi_gru = nn.GRU(
                input_size=self.cfg.hidden_dim,
                hidden_size=self.cfg.hidden_dim // 2,
                batch_first=True,
                bidirectional=True,
            )
        else:
            self.bi_gru = None

        # 5. Multi-Task Heads (Single-Nucleotide Resolution)
        self.head_donor = nn.Sequential(
            nn.Conv1d(self.cfg.hidden_dim, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=1)
        )

        self.head_acceptor = nn.Sequential(
            nn.Conv1d(self.cfg.hidden_dim, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=1)
        )

        self.head_start = nn.Sequential(
            nn.Conv1d(self.cfg.hidden_dim, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=1)
        )

        self.head_stop = nn.Sequential(
            nn.Conv1d(self.cfg.hidden_dim, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=1)
        )

        self.head_span = nn.Sequential(
            nn.Conv1d(self.cfg.hidden_dim, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, self.cfg.num_span_classes, kernel_size=1)
        )

    def forward(self, one_hot_dna: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            one_hot_dna: (Batch, 4, Length)
        Returns:
            Dictionary of predicted probabilities and latent features:
                - 'donor_prob': (Batch, Length)
                - 'acceptor_prob': (Batch, Length)
                - 'start_prob': (Batch, Length)
                - 'stop_prob': (Batch, Length)
                - 'span_probs': (Batch, 3, Length) [always 3-class for downstream DAG decoders]
                - 'span_probs_raw': (Batch, num_span_classes, Length)
                - 'span_logits': (Batch, num_span_classes, Length)
                - 'embeddings': (Batch, Hidden_Dim, Length)
        """
        # Concatenate raw DNA with composition-normalized DNA
        normed = self.comp_norm(one_hot_dna)
        dual_input = torch.cat([one_hot_dna, normed], dim=1)

        x = F.gelu(self.in_proj(dual_input))

        for block in self.dilated_blocks:
            x = block(x)

        # BiGRU context (if enabled)
        if self.bi_gru is not None:
            x_perm = x.permute(0, 2, 1)
            gru_out, _ = self.bi_gru(x_perm)
            x_ctx = gru_out.permute(0, 2, 1) + x  # Residual connection
        else:
            x_ctx = x

        donor_logits = self.head_donor(x_ctx).squeeze(1)
        acceptor_logits = self.head_acceptor(x_ctx).squeeze(1)
        start_logits = self.head_start(x_ctx).squeeze(1)
        stop_logits = self.head_stop(x_ctx).squeeze(1)
        span_logits = self.head_span(x_ctx)

        span_probs_raw = F.softmax(span_logits, dim=1)

        # Downstream compatibility: if 7-class phase-aware is used, project to 3 classes
        if self.cfg.num_span_classes == 7:
            # Classes: 0: Intergenic, 1: Intron, 2: Phase0, 3: Phase1, 4: Phase2, 5: Start, 6: Stop
            # Exon probability is sum of Phase 0, 1, 2 (+ start/stop codons)
            exon_prob = span_probs_raw[:, 2:7, :].sum(dim=1, keepdim=True)
            span_probs_3class = torch.cat([span_probs_raw[:, 0:2, :], exon_prob], dim=1)
        else:
            span_probs_3class = span_probs_raw

        return {
            "donor_prob": torch.sigmoid(donor_logits),
            "acceptor_prob": torch.sigmoid(acceptor_logits),
            "start_prob": torch.sigmoid(start_logits),
            "stop_prob": torch.sigmoid(stop_logits),
            "span_probs": span_probs_3class,
            "span_probs_raw": span_probs_raw,
            "span_logits": span_logits,
            "embeddings": x_ctx,
        }

    @staticmethod
    def sequence_to_one_hot(seq: str, device: Optional[torch.device] = None) -> torch.Tensor:
        """Helper to convert DNA string to (1, 4, L) one-hot tensor."""
        table = {"A": 0, "C": 1, "G": 2, "T": 3,
                 "a": 0, "c": 1, "g": 2, "t": 3}
        L = len(seq)
        arr = np.zeros((4, L), dtype=np.float32)
        for i, char in enumerate(seq):
            idx = table.get(char, None)
            if idx is not None:
                arr[idx, i] = 1.0
            else:
                arr[:, i] = 0.25  # Uniform background for unknown N
        t = torch.from_numpy(arr).unsqueeze(0)
        return t.to(device) if device is not None else t
