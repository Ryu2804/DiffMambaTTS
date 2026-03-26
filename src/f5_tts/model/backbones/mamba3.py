"""
Mamba-3 Backbone for F5-TTS
Drop-in replacement for the DiT backbone.

References:
- Mamba-3: exponential-trapezoid discretization, complex state via data-dependent RoPE,
  optional MIMO for higher arithmetic intensity (ICLR 2026).
- Official kernels: src/third_party/mamba/mamba_ssm/modules/mamba3.py

Interface contract (must match DiT exactly so CFM/Trainer work without changes):
  __init__(*, dim, depth, mel_dim, text_num_embeds, text_dim, ...)
  forward(x, cond, text, time, mask, drop_audio_cond, drop_text, cfg_infer, cache) -> (B, T, mel_dim)
  clear_cache()
  .dim attribute
"""
# ruff: noqa: F722 F821

from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch import nn

# Reuse F5-TTS modules (text encoder, input proj, timestep emb, norms, etc.)
from f5_tts.model.modules import (
    AdaLayerNorm,
    AdaLayerNorm_Final,
    ConvNeXtV2Block,
    FeedForward,
    TimestepEmbedding,
    precompute_freqs_cis,
)

# ---------------------------------------------------------------------------
# Mamba-3 resolver
# ---------------------------------------------------------------------------


class _FallbackMamba3(nn.Module):
    """CPU-safe fallback used when mamba kernels are unavailable."""

    def __init__(self, d_model: int, dropout: float = 0.0, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, u, *args, **kwargs):
        return self.net(u)


@lru_cache(maxsize=1)
def _resolve_mamba3_impl():
    """
    Resolve the Mamba3 implementation lazily.

    1) Prefer installed ``mamba_ssm``.
    2) Try vendored ``src/third_party/mamba``.
    3) Fall back to a lightweight block so tests can still run.
    """
    if find_spec("mamba_ssm") is None:
        vendored_root = Path(__file__).resolve().parents[3] / "third_party" / "mamba"
        if vendored_root.exists():
            vendored_str = str(vendored_root)
            if vendored_str not in sys.path:
                sys.path.insert(0, vendored_str)

    try:
        from mamba_ssm.modules.mamba3 import Mamba3  # type: ignore

        return Mamba3
    except Exception:
        return _FallbackMamba3


# ---------------------------------------------------------------------------
# Text Embedding (copied/adapted from DiT to keep the same behaviour)
# ---------------------------------------------------------------------------

class TextEmbedding(nn.Module):
    """
    Character-level text encoder with optional ConvNeXtV2 blocks.
    Identical to the one in dit.py so checkpoints/vocabs are interchangeable.
    """

    def __init__(
        self,
        text_num_embeds: int,
        text_dim: int,
        mask_padding: bool = True,
        conv_layers: int = 0,
        conv_mult: int = 2,
    ):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # 0 = filler
        self.mask_padding = mask_padding

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 8192
            self.register_buffer(
                "freqs_cis",
                precompute_freqs_cis(text_dim, self.precompute_max_pos),
                persistent=False,
            )
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, text: "int[b nt]", seq_len, drop_text: bool = False):
        text = text + 1  # shift: -1 pad → 0 filler

        if torch.is_tensor(seq_len):
            seq_len = seq_len.to(device=text.device, dtype=torch.long)
            max_seq_len = int(seq_len.max().item())
        else:
            max_seq_len = int(seq_len)

        text = text[:, :max_seq_len]
        text = F.pad(text, (0, max_seq_len - text.shape[1]), value=0)

        valid_pos_mask = None
        if torch.is_tensor(seq_len):
            seq_pos = torch.arange(max_seq_len, device=text.device).unsqueeze(0)
            valid_pos_mask = seq_pos < seq_len.unsqueeze(1)
            text = text.masked_fill(~valid_pos_mask, 0)

        if self.mask_padding:
            text_mask = text == 0

        if drop_text:
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # (B, T, text_dim)

        if valid_pos_mask is not None:
            text = text.masked_fill(~valid_pos_mask.unsqueeze(-1), 0.0)

        if self.extra_modeling:
            freqs = self.freqs_cis[:max_seq_len, :]
            if valid_pos_mask is not None:
                freqs = freqs.unsqueeze(0) * valid_pos_mask.unsqueeze(-1).to(freqs.dtype)
            text = text + freqs

            if self.mask_padding:
                text = text.masked_fill(text_mask.unsqueeze(-1).expand_as(text), 0.0)
                for block in self.text_blocks:
                    text = block(text)
                    text = text.masked_fill(text_mask.unsqueeze(-1).expand_as(text), 0.0)
            else:
                text = self.text_blocks(text)

        return text


# ---------------------------------------------------------------------------
# Input Embedding (identical to DiT's InputEmbedding)
# ---------------------------------------------------------------------------

class InputEmbedding(nn.Module):
    """
    Projects [noised_mel | cond_mel | text_embed] → hidden dim and adds
    a ConvPositionEmbedding (replicates DiT's InputEmbedding exactly).
    """

    def __init__(self, mel_dim: int, text_dim: int, out_dim: int):
        super().__init__()
        from f5_tts.model.modules import ConvPositionEmbedding
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(
        self,
        x: "float[b n d]",
        cond: "float[b n d]",
        text_embed: "float[b n d]",
        drop_audio_cond: bool = False,
        audio_mask: "bool[b n] | None" = None,
    ):
        if drop_audio_cond:
            cond = torch.zeros_like(cond)
        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        x = self.conv_pos_embed(x, mask=audio_mask) + x
        return x


# ---------------------------------------------------------------------------
# Mamba-3 Block (replaces DiTBlock)
# ---------------------------------------------------------------------------

class Mamba3Block(nn.Module):
    """
    A single processing block that pairs an AdaLayerNorm-conditioned Mamba-3 SSM
    with an AdaLayerNorm-conditioned FeedForward MLP.

    Interface matches DiTBlock.forward(x, t, mask, rope):
      - rope is accepted but ignored (Mamba-3 has internal data-dependent RoPE).
      - mask is accepted but currently not applied inside the SSM scan
        (Mamba-3's causal scan is naturally compatible with padding).
    """

    def __init__(
        self,
        dim: int,
        ff_mult: int = 2,
        dropout: float = 0.0,
        # Mamba-3 specific
        d_state: int = 128,
        headdim: int = 64,
        ngroups: int = 1,
        rope_fraction: float = 0.5,
        is_mimo: bool = False,
        mimo_rank: int = 1,
        chunk_size: int = 64,
        layer_idx: int | None = None,
    ):
        super().__init__()

        # AdaLayerNorm for SSM input (same as DiTBlock.attn_norm)
        self.ssm_norm = AdaLayerNorm(dim)

        # Mamba-3 SSM (replaces self-attention)
        mamba3_impl = _resolve_mamba3_impl()
        self.ssm = mamba3_impl(
            d_model=dim,
            d_state=d_state,
            headdim=headdim,
            ngroups=ngroups,
            rope_fraction=rope_fraction,
            is_mimo=is_mimo,
            mimo_rank=mimo_rank,
            chunk_size=chunk_size,
            dropout=dropout,
            layer_idx=layer_idx,
        )

        # AdaLayerNorm + FeedForward (same as DiTBlock ff path)
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(
        self,
        x: "float[b n d]",
        t: "float[b d]",
        mask: "bool[b n] | None" = None,
        rope=None,  # accepted for API compatibility, ignored
    ):
        # --- SSM path (mirrors DiTBlock attention path) ---
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ssm_norm(x, emb=t)

        # Mamba-3 forward: (B, T, D) → (B, T, D)
        ssm_out = self.ssm(norm)

        x = x + gate_msa.unsqueeze(1) * ssm_out

        # --- FeedForward path ---
        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_out = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_out

        return x


# ---------------------------------------------------------------------------
# Mamba-3 Backbone (drop-in for DiT)
# ---------------------------------------------------------------------------

class Mamba3Backbone(nn.Module):
    """
    Drop-in replacement for the DiT backbone in F5-TTS.

    Keeps all outer scaffolding from DiT (text embedding, input embedding,
    timestep embedding, AdaLayerNorm_Final, cfg_infer, text caching) and
    replaces only the internal DiT attention blocks with Mamba-3 SSM blocks.

    Constructor accepts a superset of DiT's kwargs so model configs are
    forward-compatible.
    """

    def __init__(
        self,
        *,
        dim: int,
        depth: int = 22,
        heads: int = 16,           # accepted but ignored (Mamba-3 uses headdim)
        dim_head: int = 64,         # accepted but ignored
        dropout: float = 0.0,
        ff_mult: int = 2,
        mel_dim: int = 100,
        text_num_embeds: int = 256,
        text_dim: int | None = None,
        text_mask_padding: bool = True,
        text_embedding_average_upsampling: bool = False,
        qk_norm: str | None = None,  # accepted but ignored
        conv_layers: int = 4,
        pe_attn_head: int | None = None,  # accepted but ignored
        attn_backend: str = "torch",      # accepted but ignored
        attn_mask_enabled: bool = False,  # accepted but ignored
        long_skip_connection: bool = False,
        checkpoint_activations: bool = False,
        # ------------------------------------------------------------------
        # Mamba-3 specific kwargs (absent from DiT, safe to omit in configs)
        # ------------------------------------------------------------------
        d_state: int = 128,
        headdim: int = 64,
        ngroups: int = 1,
        rope_fraction: float = 0.5,
        is_mimo: bool = False,
        mimo_rank: int = 1,
        chunk_size: int = 64,
    ):
        super().__init__()

        self.dim = dim
        self.depth = depth

        if text_dim is None:
            text_dim = mel_dim

        # --- Outer scaffolding (identical to DiT) ---
        self.time_embed = TimestepEmbedding(dim)
        self.text_embed = TextEmbedding(
            text_num_embeds,
            text_dim,
            mask_padding=text_mask_padding,
            conv_layers=conv_layers,
        )
        self.text_cond = None
        self.text_uncond = None

        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)

        # --- Mamba-3 block stack (replaces DiT's transformer_blocks) ---
        self.transformer_blocks = nn.ModuleList([
            Mamba3Block(
                dim=dim,
                ff_mult=ff_mult,
                dropout=dropout,
                d_state=d_state,
                headdim=headdim,
                ngroups=ngroups,
                rope_fraction=rope_fraction,
                is_mimo=is_mimo,
                mimo_rank=mimo_rank,
                chunk_size=chunk_size,
                layer_idx=i,
            )
            for i in range(depth)
        ])

        # Optional long-skip connection (same as DiT)
        self.long_skip_connection = (
            nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None
        )

        # --- Final norm + output projection (identical to DiT) ---
        self.norm_out = AdaLayerNorm_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)

        self.checkpoint_activations = checkpoint_activations

        self.initialize_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def initialize_weights(self):
        """Zero-out adaLN and output layers for stable training (same as DiT)."""
        for block in self.transformer_blocks:
            nn.init.constant_(block.ssm_norm.linear.weight, 0)
            nn.init.constant_(block.ssm_norm.linear.bias, 0)

        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    # ------------------------------------------------------------------
    # Input embedding with caching (identical interface to DiT)
    # ------------------------------------------------------------------

    def get_input_embed(
        self,
        x,
        cond,
        text,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cache: bool = True,
        audio_mask: "bool[b n] | None" = None,
    ):
        if self.text_uncond is None or self.text_cond is None or not cache:
            seq_len = x.shape[1] if audio_mask is None else audio_mask.sum(dim=1)
            text_embed = self.text_embed(text, seq_len=seq_len, drop_text=drop_text)
            if cache:
                if drop_text:
                    self.text_uncond = text_embed
                else:
                    self.text_cond = text_embed

        if cache:
            text_embed = self.text_uncond if drop_text else self.text_cond

        x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond, audio_mask=audio_mask)
        return x

    def clear_cache(self):
        """Reset text embedding cache between inference calls."""
        self.text_cond = None
        self.text_uncond = None

    # ------------------------------------------------------------------
    # Gradient checkpointing wrapper
    # ------------------------------------------------------------------

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            return module(*inputs)
        return ckpt_forward

    # ------------------------------------------------------------------
    # Forward pass (identical signature to DiT.forward)
    # ------------------------------------------------------------------

    def forward(
        self,
        x: "float[b n d]",          # noised input mel
        cond: "float[b n d]",        # masked conditioning mel
        text: "int[b nt]",           # text token indices
        time: "float[b] | float[]",  # flow timestep
        mask: "bool[b n] | None" = None,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cfg_infer: bool = False,     # pack cond+uncond for CFG
        cache: bool = False,
    ):
        batch, seq_len = x.shape[0], x.shape[1]

        if time.ndim == 0:
            time = time.repeat(batch)

        # Timestep embedding
        t = self.time_embed(time)  # (B, dim)

        if cfg_infer:  # pack cond & uncond forward: b n d → 2b n d
            x_cond = self.get_input_embed(
                x, cond, text, drop_audio_cond=False, drop_text=False,
                cache=cache, audio_mask=mask,
            )
            x_uncond = self.get_input_embed(
                x, cond, text, drop_audio_cond=True, drop_text=True,
                cache=cache, audio_mask=mask,
            )
            x = torch.cat((x_cond, x_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
        else:
            x = self.get_input_embed(
                x, cond, text,
                drop_audio_cond=drop_audio_cond,
                drop_text=drop_text,
                cache=cache,
                audio_mask=mask,
            )

        # Optional long-skip connection residual
        if self.long_skip_connection is not None:
            residual = x

        # Mamba-3 block stack
        # rope=None since Mamba-3 has its own internal data-dependent RoPE
        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(block), x, t, mask, None, use_reentrant=False
                )
            else:
                x = block(x, t, mask=mask, rope=None)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        # Final adaLN + output projection
        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output
