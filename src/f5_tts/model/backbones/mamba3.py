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

import inspect
from importlib.util import find_spec
from pathlib import Path
import sys
import warnings

import torch
import torch.nn.functional as F
from torch import nn

# Reuse F5-TTS modules (text encoder, input proj, timestep emb, norms, etc.)
from f5_tts.model.modules import (
    AdaLayerNorm_Final,
    ConvNeXtV2Block,
    FeedForward,
    TimestepEmbedding,
    precompute_freqs_cis,
)

# ---------------------------------------------------------------------------
# Mamba-3 resolver
# ---------------------------------------------------------------------------

_MAMBA3_IMPL_CACHE = None


class _FallbackMamba3(nn.Module):
    """CPU-safe fallback used when mamba kernels are unavailable.

    WARNING: This is a non-SSM placeholder (Linear+GELU). It does NOT have
    the sequence-mixing or state-space properties of a real Mamba kernel.
    Install `mamba_ssm` + `causal_conv1d` for correct behaviour.
    """

    def __init__(self, d_model: int, dropout: float = 0.0, **kwargs):
        super().__init__()
        warnings.warn(
            "mamba_ssm not found — using _FallbackMamba3 (two Linear layers). "
            "Training will proceed but WITHOUT a real SSM scan. "
            "Install `mamba_ssm` and `causal_conv1d` to fix this.",
            RuntimeWarning,
            stacklevel=2,
        )
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, u, *args, **kwargs):
        return self.net(u)


def _resolve_mamba3_impl():
    """
    Resolve the Mamba3 implementation lazily.

    1) Prefer installed ``mamba_ssm``.
    2) Try vendored ``src/third_party/mamba``.
    3) Fall back to a lightweight block so tests can still run.
    """
    global _MAMBA3_IMPL_CACHE
    if _MAMBA3_IMPL_CACHE is not None:
        return _MAMBA3_IMPL_CACHE

    if find_spec("mamba_ssm") is None:
        vendored_root = Path(__file__).resolve().parents[3] / "third_party" / "mamba"
        if vendored_root.exists():
            vendored_str = str(vendored_root)
            if vendored_str not in sys.path:
                sys.path.insert(0, vendored_str)

    try:
        from mamba_ssm.modules.mamba3 import Mamba3  # type: ignore

        _MAMBA3_IMPL_CACHE = Mamba3
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
        average_upsampling: bool = False,
        conv_layers: int = 0,
        conv_mult: int = 2,
    ):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # 0 = filler
        self.mask_padding = mask_padding
        self.average_upsampling = average_upsampling

        if average_upsampling:
            assert mask_padding, "text_embedding_average_upsampling requires text_mask_padding to be True"

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

    def average_upsample_text_by_mask(self, text, text_mask, target_lens):
        batch, _, _ = text.shape
        text_lens = text_mask.sum(dim=1)

        upsampled_text = torch.zeros_like(text)

        for i in range(batch):
            text_len = int(text_lens[i].item())
            audio_len = int(target_lens[i].item())

            if text_len == 0 or audio_len <= 0:
                continue

            valid_ind = torch.where(text_mask[i])[0]
            valid_data = text[i, valid_ind, :]

            base_repeat = audio_len // text_len
            remainder = audio_len % text_len

            indices = []
            for j in range(text_len):
                repeat_count = base_repeat + (1 if j >= text_len - remainder else 0)
                indices.extend([j] * repeat_count)

            indices = torch.tensor(indices[:audio_len], device=text.device, dtype=torch.long)
            upsampled = valid_data[indices]

            upsampled_text[i, :audio_len, :] = upsampled

        return upsampled_text

    def forward(self, text: torch.Tensor, seq_len, drop_text: bool = False):
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

        if self.average_upsampling:
            if torch.is_tensor(seq_len):
                target_lens = seq_len.to(device=text.device, dtype=torch.long)
            else:
                target_lens = torch.full((text.shape[0],), int(seq_len), device=text.device, dtype=torch.long)

            text = self.average_upsample_text_by_mask(text, ~text_mask, target_lens)

        return text


class AdaLayerNormSimple(nn.Module):
    """AdaLayerNorm variant that returns only SSM modulation terms."""

    def __init__(self, dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 3)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        shift, scale, gate = torch.chunk(emb, 3, dim=1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x, gate


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
        x: torch.Tensor,
        cond: torch.Tensor,
        text_embed: torch.Tensor,
        drop_audio_cond: bool = False,
        audio_mask: torch.Tensor | None = None,
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
    A single processing block that pairs an AdaLayerNorm-conditioned **bidirectional**
    Mamba-3 SSM with a separately AdaLayerNorm-conditioned FeedForward MLP.

    Bidirectionality: two independent SSM instances scan the sequence forward and
    backward; their outputs are summed before the residual add.  This is required
    for non-causal tasks such as TTS diffusion where every frame needs full context.

    Interface matches DiTBlock.forward(x, t, mask, rope):
      - rope is accepted but ignored (Mamba-3 has internal data-dependent RoPE).
      - mask is zeroed in/out of the SSM scan to prevent gradient leakage from
        padding positions into valid frames.
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

        # AdaLayerNorm for SSM path (shift, scale, gate only).
        self.ssm_norm = AdaLayerNormSimple(dim)

        # Build SSM kwargs once so both fwd/bwd share the same hyper-params.
        mamba3_impl = _resolve_mamba3_impl()
        _ssm_kwargs = dict(
            d_model=dim,
            d_state=d_state,
            headdim=headdim,
            ngroups=ngroups,
            rope_fraction=rope_fraction,
            is_mimo=is_mimo,
            mimo_rank=mimo_rank,
            chunk_size=chunk_size,
            layer_idx=layer_idx,
        )
        supports_dropout = False
        try:
            sig = inspect.signature(mamba3_impl.__init__)
            supports_dropout = "dropout" in sig.parameters
        except (TypeError, ValueError):
            supports_dropout = False

        if supports_dropout:
            _ssm_kwargs["dropout"] = dropout

        # Forward scan
        self.ssm = mamba3_impl(**_ssm_kwargs)
        # Backward scan — separate parameters, same architecture
        self.ssm_bwd = mamba3_impl(**_ssm_kwargs)
        self.ssm_dropout = nn.Identity() if (dropout == 0.0 or supports_dropout) else nn.Dropout(dropout)

        # FF path uses plain LN, then explicit MLP modulation.
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_ada_proj = nn.Linear(dim, dim * 3)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor | None = None,
        rope=None,  # accepted for API compatibility, ignored
    ):
        # ------------------------------------------------------------------ #
        # SSM path  (bidirectional = forward scan + backward scan)
        # ------------------------------------------------------------------ #
        norm, gate_msa = self.ssm_norm(x, emb=t)

        # Zero-out padding positions *before* the scan so padding tokens do
        # not inject gradients into valid positions via the state update.
        if mask is not None:
            norm = norm.masked_fill(~mask.unsqueeze(-1), 0.0)

        # Forward scan: left-to-right
        fwd = self.ssm(norm)
        # Backward scan: flip sequence, scan left-to-right, flip back.
        # Each token's backward output therefore summarises right-context.
        bwd = self.ssm_bwd(norm.flip(1)).flip(1)
        ssm_out = self.ssm_dropout(fwd + bwd)

        # Zero-out padding positions *after* the scan as well.
        if mask is not None:
            ssm_out = ssm_out.masked_fill(~mask.unsqueeze(-1), 0.0)

        x = x + gate_msa.unsqueeze(1) * ssm_out

        # ------------------------------------------------------------------ #
        # FeedForward path  (independent AdaLayerNorm)
        # ------------------------------------------------------------------ #
        ff_ada = self.ff_ada_proj(F.silu(t))
        shift_mlp, scale_mlp, gate_mlp = torch.chunk(ff_ada, 3, dim=1)
        ff_norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_out = self.ff(ff_norm)
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
            average_upsampling=text_embedding_average_upsampling,
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
        """Zero-out adaLN modulation layers for stable training start (same as DiT).

        proj_out is intentionally NOT zero-initialised: combined with a real SSM
        scan, zero proj_out would make the model extremely slow to produce
        meaningful predictions early in training.
        """
        for block in self.transformer_blocks:
            # SSM-path AdaLN
            nn.init.constant_(block.ssm_norm.linear.weight, 0)
            nn.init.constant_(block.ssm_norm.linear.bias, 0)
            # FF-path explicit modulation projection
            nn.init.constant_(block.ff_ada_proj.weight, 0)
            nn.init.constant_(block.ff_ada_proj.bias, 0)

        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)

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
        audio_mask: torch.Tensor | None = None,
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

    @staticmethod
    def ckpt_wrapper(module):
        """Wrap a module for use with torch.utils.checkpoint.

        Using @staticmethod avoids implicitly capturing ``self`` in the closure,
        which would be fragile if ``self`` ever holds non-tensor state that
        changes between forward calls.  The module itself is still captured but
        that is unavoidable and safe (its parameter tensors are what matter).
        """
        def ckpt_forward(*inputs):
            return module(*inputs)
        return ckpt_forward

    # ------------------------------------------------------------------
    # Forward pass (identical signature to DiT.forward)
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,          # noised input mel
        cond: torch.Tensor,        # masked conditioning mel
        text: torch.Tensor,           # text token indices
        time: torch.Tensor,  # flow timestep
        mask: torch.Tensor | None = None,
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
