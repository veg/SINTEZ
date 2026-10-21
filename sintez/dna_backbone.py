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
    use_gru: bool = True            # Classic v1 uses BiGRU; Foundation v2/v3 uses pure dilated conv
    use_codon_conv: bool = False    # Foundation v3: parallel dilation=3 codon-periodic residual branch
    use_asymmetric_heads: bool = False  # Foundation v3: asymmetric boundary-difference splice/codon heads + dinuc gates
    use_foldability_head: bool = False  # Foundation v3: auxiliary neural foldability / coding quality distillation head

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
            use_codon_conv=False,
            use_asymmetric_heads=False,
            use_foldability_head=False,
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
            use_codon_conv=False,
            use_asymmetric_heads=False,
            use_foldability_head=False,
        )

    @classmethod
    def foundation_v3(
        cls,
        hidden_dim: int = 128,
        dilations: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128),
        norm_window_sizes: Tuple[int, ...] = (101, 501, 2001),
        num_span_classes: int = 7,
        dropout: float = 0.1,
        use_gru: bool = False,
    ) -> DNAEncoderConfig:
        """
        Foundation v3 architecture featuring:
        - Multi-scale composition normalization (101, 501, 2001 bp)
        - Parallel codon-periodic (dilation=3) residual branches in every DilatedConvBlock
        - Asymmetric boundary-difference junction heads with canonical dinucleotide residual gating
        - Auxiliary foldability/coding-potential distillation head
        """
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
            use_codon_conv=True,
            use_asymmetric_heads=True,
            use_foldability_head=True,
        )

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, torch.Tensor]) -> DNAEncoderConfig:
        """Automatically reconstructs the exact DNAEncoderConfig from a saved state_dict."""
        in_w = state_dict.get("in_proj.weight")
        hidden_dim = int(in_w.shape[0]) if in_w is not None else 128
        in_ch_total = int(in_w.shape[1]) if in_w is not None else 8
        norm_window_sizes = (101, 501, 2001) if in_ch_total == 16 else None

        span_w = state_dict.get("head_span.2.weight")
        num_span_classes = int(span_w.shape[0]) if span_w is not None else 3

        block_indices = sorted({
            int(k.split(".")[1])
            for k in state_dict.keys()
            if k.startswith("dilated_blocks.") and k.split(".")[1].isdigit()
        })
        if len(block_indices) == 4:
            dilations = (1, 3, 9, 27)
        else:
            dilations = tuple(2 ** i for i in range(len(block_indices))) if block_indices else (1, 2, 4, 8, 16, 32, 64, 128)

        use_gru = any(k.startswith("bi_gru.") for k in state_dict.keys())
        use_codon_conv = any("conv_codon" in k for k in state_dict.keys())
        use_asymmetric_heads = any("head_donor.proj" in k for k in state_dict.keys())
        use_foldability_head = any(k.startswith("head_foldability.") for k in state_dict.keys())

        return cls(
            in_channels=4,
            hidden_dim=hidden_dim,
            kernel_size=7,
            dilations=dilations,
            norm_window_size=201 if len(block_indices) == 4 else 501,
            norm_window_sizes=norm_window_sizes,
            num_span_classes=num_span_classes,
            dropout=0.1,
            use_gru=use_gru,
            use_codon_conv=use_codon_conv,
            use_asymmetric_heads=use_asymmetric_heads,
            use_foldability_head=use_foldability_head,
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
    """
    Multi-scale residual dilated convolution block.
    When `use_codon_conv=True` (Foundation v3), adds a parallel `dilation=3`
    branch that directly links corresponding codon positions (i - 3, i, i + 3)
    across adjacent triplets to provide an inductive bias for 3-base periodicity.
    """
    def __init__(
        self,
        channels: int,
        dilation: int,
        kernel_size: int = 7,
        dropout: float = 0.1,
        use_codon_conv: bool = False,
    ):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.use_codon_conv = use_codon_conv
        if use_codon_conv:
            self.conv_codon = nn.Conv1d(channels, channels, kernel_size=3, padding=3, dilation=3)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.act2 = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.drop(x)
        out = self.conv2(x)
        if self.use_codon_conv:
            out = out + self.conv_codon(x)
        x = self.act2(self.norm2(out))
        return x + res


class AsymmetricJunctionHead(nn.Module):
    """
    Asymmetric boundary-transition detection head for splice junctions and start/stop codons.
    Combines:
      1. Local context embeddings x_ctx(t)
      2. Multi-scale asymmetric step-differences: h(t + k) - h(t - k) for k in (4, 16, 48)
         capturing the transition between 3-periodic coding exons and non-coding introns/UTRs
      3. High-resolution raw DNA motif features around the boundary (kernel_size=15)
    """
    def __init__(self, hidden_dim: int, dna_channels: int = 4):
        super().__init__()
        self.dna_conv = nn.Sequential(
            nn.Conv1d(dna_channels, 16, kernel_size=15, padding=7),
            nn.GELU(),
        )
        # Input channels: hidden_dim (base) + 3 * hidden_dim (scales k=4, 16, 48) + 16 (local DNA motif)
        in_total = hidden_dim * 4 + 16
        self.proj = nn.Sequential(
            nn.Conv1d(in_total, 48, kernel_size=5, padding=2),
            nn.GroupNorm(num_groups=8, num_channels=48),
            nn.GELU(),
            nn.Conv1d(48, 1, kernel_size=1),
        )

    @staticmethod
    def _step_diff(x: torch.Tensor, k: int) -> torch.Tensor:
        """Computes h(t + k) - h(t - k) along the sequence dimension with replicate padding."""
        padded = F.pad(x, (k, k), mode="replicate")
        L = x.shape[-1]
        downstream = padded[:, :, 2 * k : 2 * k + L]
        upstream = padded[:, :, :L]
        return downstream - upstream

    def forward(self, x_ctx: torch.Tensor, one_hot_dna: torch.Tensor) -> torch.Tensor:
        d4 = self._step_diff(x_ctx, 4)
        d16 = self._step_diff(x_ctx, 16)
        d48 = self._step_diff(x_ctx, 48)
        motif = self.dna_conv(one_hot_dna)
        feat = torch.cat([x_ctx, d4, d16, d48, motif], dim=1)
        return self.proj(feat).squeeze(1)


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
            DilatedConvBlock(
                self.cfg.hidden_dim,
                dilation=d,
                kernel_size=self.cfg.kernel_size,
                dropout=self.cfg.dropout,
                use_codon_conv=self.cfg.use_codon_conv,
            )
            for d in self.cfg.dilations
        ])

        # 4. Bidirectional Sequence Context (optional, disabled in foundation_v2/v3 for 150x GPU acceleration)
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
        if self.cfg.use_asymmetric_heads:
            self.head_donor = AsymmetricJunctionHead(self.cfg.hidden_dim, self.cfg.in_channels)
            self.head_acceptor = AsymmetricJunctionHead(self.cfg.hidden_dim, self.cfg.in_channels)
            self.head_start = AsymmetricJunctionHead(self.cfg.hidden_dim, self.cfg.in_channels)
            self.head_stop = AsymmetricJunctionHead(self.cfg.hidden_dim, self.cfg.in_channels)
            # Learnable soft dinucleotide motif gates (initialized so canonical sites get +1.5 logit boost
            # and non-canonical sites get -1.5 suppression while still permitting Gaussian soft windows)
            self.motif_gate_scale = nn.Parameter(torch.tensor([2.5, 2.5, 2.0, 2.0], dtype=torch.float32))
        else:
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

        # 6. Auxiliary Foldability / Coding Quality Distillation Head (Foundation v3)
        if self.cfg.use_foldability_head:
            self.head_foldability = nn.Sequential(
                nn.Conv1d(self.cfg.hidden_dim, 32, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(32, 1, kernel_size=1),
            )
        else:
            self.head_foldability = None

    @staticmethod
    def _compute_canonical_motif_indicators(one_hot_dna: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes smooth local indicators (pooled over +/-2 bp radius to match Gaussian target smoothing)
        for canonical splice and translation boundary motifs:
          - Donor: GT or GC at (pos, pos+1)
          - Acceptor: AG at (pos-1, pos)
          - Start: ATG/GTG/TTG at (pos, pos+1, pos+2)
          - Stop: TAA/TAG/TGA at (pos, pos+1, pos+2)
        """
        # Channels: 0=A, 1=C, 2=G, 3=T
        A = one_hot_dna[:, 0, :]
        C = one_hot_dna[:, 1, :]
        G = one_hot_dna[:, 2, :]
        T = one_hot_dna[:, 3, :]

        G_next = F.pad(G[:, 1:], (0, 1), value=0.0)
        T_next = F.pad(T[:, 1:], (0, 1), value=0.0)
        C_next = F.pad(C[:, 1:], (0, 1), value=0.0)
        A_next = F.pad(A[:, 1:], (0, 1), value=0.0)

        G_next2 = F.pad(G[:, 2:], (0, 2), value=0.0)
        A_next2 = F.pad(A[:, 2:], (0, 2), value=0.0)
        A_prev = F.pad(A[:, :-1], (1, 0), value=0.0)

        # Exact position indicators in [0, 1]
        is_donor = G * torch.clamp(T_next + 0.5 * C_next, 0.0, 1.0)
        is_accept = A_prev * G
        is_start = torch.clamp(A + 0.3 * G + 0.3 * T, 0.0, 1.0) * T_next * G_next2
        is_stop = T * torch.clamp(A_next * A_next2 + A_next * G_next2 + G_next * A_next2, 0.0, 1.0)

        # Max-pool over a 5-bp window (radius=2) so Gaussian target neighbors around canonical sites remain unpenalized
        def pool5(t: torch.Tensor) -> torch.Tensor:
            return F.max_pool1d(F.pad(t.unsqueeze(1), (2, 2), mode="replicate"), kernel_size=5, stride=1).squeeze(1)

        return pool5(is_donor), pool5(is_accept), pool5(is_start), pool5(is_stop)

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
                - 'foldability_prob': (Batch, Length) [if use_foldability_head=True]
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

        if self.cfg.use_asymmetric_heads:
            donor_logits = self.head_donor(x_ctx, one_hot_dna)
            acceptor_logits = self.head_acceptor(x_ctx, one_hot_dna)
            start_logits = self.head_start(x_ctx, one_hot_dna)
            stop_logits = self.head_stop(x_ctx, one_hot_dna)

            m_don, m_acc, m_sta, m_sto = self._compute_canonical_motif_indicators(one_hot_dna)
            scales = torch.clamp(self.motif_gate_scale, 0.0, 6.0)
            # Centered residual gate: (m - 0.5) is +0.5 near canonical motifs and -0.5 elsewhere
            donor_logits = donor_logits + scales[0] * (m_don - 0.5)
            acceptor_logits = acceptor_logits + scales[1] * (m_acc - 0.5)
            start_logits = start_logits + scales[2] * (m_sta - 0.5)
            stop_logits = stop_logits + scales[3] * (m_sto - 0.5)
        else:
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

        out_dict = {
            "donor_prob": torch.sigmoid(donor_logits),
            "acceptor_prob": torch.sigmoid(acceptor_logits),
            "start_prob": torch.sigmoid(start_logits),
            "stop_prob": torch.sigmoid(stop_logits),
            "span_probs": span_probs_3class,
            "span_probs_raw": span_probs_raw,
            "span_logits": span_logits,
            "embeddings": x_ctx,
        }
        if self.head_foldability is not None:
            out_dict["foldability_prob"] = torch.sigmoid(self.head_foldability(x_ctx).squeeze(1))
        return out_dict

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
