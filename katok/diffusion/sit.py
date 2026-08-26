"""SiT backbone for generation over KATok's sparse latent tokens.

Adapted from SiT (https://github.com/willisma/SiT) for variable-length 1D token
sequences. The differences from vanilla SiT that matter here:

* The input is a packed sequence of latent tokens, not a 2D image grid. Samples in
  a batch have different token counts, so attention is masked per sample.
* Conditioning is timestep + class label + **token count**, all combined additively
  and injected through adaLN-Zero. The token-count signal is what makes generation
  length controllable at inference.
* Positions are handled differently per variant: the *joint* model carries raw
  ``(t, h, w)`` coordinates as extra channels, while the *cascaded* model receives
  them as an additive embedding produced by a small MLP.

The blocks reimplement the subset of ``timm`` layers the checkpoints were trained
with, keeping identical parameter names so state dicts load unchanged.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_flash_attn_varlen():
    try:
        from flash_attn import flash_attn_varlen_func

        return flash_attn_varlen_func
    except Exception:
        return None


_FLASH_ATTN_VARLEN = _load_flash_attn_varlen()


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    """Multi-head self-attention with optional QK-normalization.

    Mirrors ``timm.models.vision_transformer.Attention`` parameter-for-parameter.
    The core attention runs in bf16 (which is how the released checkpoints were
    trained and evaluated) while the qkv/proj projections stay in fp32.
    """

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, qk_norm: bool = False):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        input_dtype = q.dtype
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x.to(input_dtype)

        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Mlp(nn.Module):
    """Two-layer MLP, matching ``timm.layers.Mlp`` parameter names."""

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def varlen_attn(x, attn_module, cu_seqlens, max_seqlen, qk_norm=False):
    """Attention over a packed batch of variable-length sequences.

    ``x`` is ``(total_tokens, D)`` with per-sample boundaries in ``cu_seqlens``,
    so no padding is materialized. FlashAttention's varlen kernel is used when
    available; otherwise the sequences are unpacked into a padded batch and run
    through SDPA with a key-padding mask, which computes the same thing.
    """
    total, D = x.shape
    num_heads = attn_module.num_heads
    head_dim = D // num_heads

    qkv = attn_module.qkv(x).bfloat16()
    q, k, v = qkv.reshape(total, 3, num_heads, head_dim).unbind(1)
    if qk_norm:
        q, k = attn_module.q_norm(q), attn_module.k_norm(k)

    if _FLASH_ATTN_VARLEN is not None:
        out = _FLASH_ATTN_VARLEN(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen)
    else:
        out = _varlen_attn_sdpa(q, k, v, cu_seqlens, max_seqlen)

    return attn_module.proj(out.reshape(total, D).to(x.dtype))


def _varlen_attn_sdpa(q, k, v, cu_seqlens, max_seqlen):
    """Pure-PyTorch stand-in for ``flash_attn_varlen_func``.

    Scatters the packed sequences into a ``(B, max_seqlen, H, d)`` batch, runs SDPA
    with a key-padding mask so tokens never attend across sample boundaries, then
    gathers the valid positions back into packed layout.
    """
    total, num_heads, head_dim = q.shape
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
    B = lengths.shape[0]
    device = q.device

    slot = torch.arange(max_seqlen, device=device).unsqueeze(0)  # (B, S)
    valid = slot < lengths.unsqueeze(1)  # (B, S)
    flat_idx = (cu_seqlens[:-1].long().unsqueeze(1) + slot)[valid]  # (total,)

    def scatter(t):
        padded = t.new_zeros(B, max_seqlen, num_heads, head_dim)
        padded[valid] = t[flat_idx]
        return padded.permute(0, 2, 1, 3)  # (B, H, S, d)

    # Keys at padded slots are masked out; every query row keeps its own sample.
    attn_mask = valid[:, None, None, :].expand(B, 1, max_seqlen, max_seqlen)
    out = F.scaled_dot_product_attention(scatter(q), scatter(k), scatter(v), attn_mask=attn_mask)

    return out.permute(0, 2, 1, 3)[valid]  # (total, H, d)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding followed by an MLP."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class LabelEmbedder(nn.Module):
    """Class-label embedding with a dedicated slot for the unconditional label."""

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def forward(self, labels):
        return self.embedding_table(labels)


class TokenLenEmbedder(nn.Module):
    """Embeds the target number of active tokens.

    This is the handle for the paper's emergent controllability: raising or lowering
    the requested token count at sampling time modulates motion and visual detail.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, dropout_prob=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.dropout_prob = dropout_prob

    def forward(self, token_len):
        return self.mlp(TimestepEmbedder.timestep_embedding(token_len, self.frequency_embedding_size))


class SiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, qk_norm=False):
        super().__init__()
        self.qk_norm = qk_norm
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c, cu_seqlens=None, max_seqlen=None):
        mod = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)

        if cu_seqlens is not None:
            # Packed: x is (total, D) and the modulations are already per-token.
            h = self.norm1(x) * (1 + scale_msa) + shift_msa
            x = x + gate_msa * varlen_attn(h, self.attn, cu_seqlens, max_seqlen, qk_norm=self.qk_norm)
            h = self.norm2(x) * (1 + scale_mlp) + shift_mlp
            x = x + gate_mlp * self.mlp(h)
        else:
            h = self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            x = x + gate_msa.unsqueeze(1) * h.to(x.dtype)
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """adaLN + linear projection back to the token channel dimension."""

    def __init__(self, hidden_size, out_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c, packed=False):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale) + shift if packed else modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


def get_1d_sincos_pos_embed(embed_dim, length):
    """Fixed 1D sinusoidal embedding over token index, ``(length, embed_dim)``."""
    pos = np.arange(length, dtype=np.float64)
    omega = 1.0 / 10000 ** (np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0))
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class SiT(nn.Module):
    """SiT over a variable-length sequence of latent tokens.

    Args:
        input_dim: channels per token. 64 for content-only models; 67 for the joint
            variant, whose last 3 channels carry the ``(t, h, w)`` coordinates.
        hidden_size / depth / num_heads / mlp_ratio: backbone size (SiT-XL: 1152/28/16).
        num_classes: number of class labels; one extra slot is reserved for the
            unconditional embedding used by classifier-free guidance.
        max_tokens: length of the learnable/sinusoidal token-index embedding table.
        embed_token_len: add the token-count embedding to the conditioning vector.
        sinusoidal_1d_pos_embed: use a fixed sinusoidal token-index embedding rather
            than a learned one. Encodes packing order, not spatial position.
        predict_pos: joint variant -- the model also denoises the position channels.
        use_position_prior: cascaded variant -- positions arrive as ``gt_positions``
            and are injected through ``pos_to_embed`` instead of a token-index embedding.
        qk_norm: RMSNorm on queries and keys.
    """

    def __init__(
        self,
        input_dim=64,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=101,
        max_tokens=8192,
        embed_token_len=True,
        token_len_dropout_prob=0.0,
        sinusoidal_1d_pos_embed=False,
        predict_pos=False,
        use_position_prior=False,
        qk_norm=False,
        # cascaded mask prior
        prior_hidden_size=192,
        prior_depth=12,
        prior_num_heads=3,
        prior_n_reg=2,
        prior_grid=(2, 16, 16),
        prior_qk_norm=True,
    ):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = input_dim
        self.embed_token_len = embed_token_len
        self.sinusoidal_1d_pos_embed = sinusoidal_1d_pos_embed
        self.predict_pos = predict_pos
        self.use_position_prior = use_position_prior
        self.num_heads = num_heads
        self._prior_n_reg = prior_n_reg if use_position_prior else 0

        self.x_embedder = nn.Linear(input_dim, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, hidden_size))
        self.final_layer = FinalLayer(hidden_size, out_dim=input_dim)

        self.t_embedder = TimestepEmbedder(hidden_size)
        if predict_pos and not use_position_prior:
            # Joint variant: content and position are noised on separate schedules,
            # so the position timestep gets its own embedder.
            self.t_pos_embedder = TimestepEmbedder(hidden_size)
        if embed_token_len:
            self.token_len_embedder = TokenLenEmbedder(hidden_size, dropout_prob=token_len_dropout_prob)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        if use_position_prior:
            from .mask_prior import grid_center_coords

            # The mask prior is itself a small SiT over the fixed occupancy grid:
            # one token per grid cell, one channel per token.
            grid_t, grid_h, grid_w = prior_grid
            self.prior_grid = (grid_t, grid_h, grid_w)
            self.position_prior = SiT(
                input_dim=1,
                hidden_size=prior_hidden_size,
                depth=prior_depth,
                num_heads=prior_num_heads,
                mlp_ratio=4.0,
                class_dropout_prob=class_dropout_prob,
                num_classes=num_classes,
                max_tokens=grid_t * grid_h * grid_w,
                embed_token_len=True,
                token_len_dropout_prob=token_len_dropout_prob,
                qk_norm=prior_qk_norm,
            )
            self.register_buffer("prior_grid_coords", grid_center_coords(prior_grid))

            # Continuous (t, h, w) -> hidden positional conditioning.
            self.pos_to_embed = nn.Sequential(
                nn.Linear(3, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )

        self.blocks = nn.ModuleList(
            [SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, qk_norm=qk_norm) for _ in range(depth)]
        )

        if sinusoidal_1d_pos_embed:
            emb = get_1d_sincos_pos_embed(hidden_size, max_tokens)
            self.pos_embed.data.copy_(torch.from_numpy(emb).float().unsqueeze(0))
            self.pos_embed.requires_grad_(False)

    def forward(self, x, t, y, valid_counts=None, t_pos=None, token_len=None, gt_positions=None):
        """Predict the flow-matching velocity.

        Args:
            x: ``(B, L, input_dim)`` noisy latent tokens.
            t: ``(B,)`` content timestep.
            y: ``(B,)`` class labels.
            valid_counts: ``(B,)`` active tokens per sample. Slots beyond the count
                are padding and are excluded from attention.
            t_pos: ``(B,)`` position timestep (joint variant).
            token_len: ``(B,)`` target token count used as conditioning.
            gt_positions: ``(B, L, 3)`` token positions (cascaded variant).
        """
        B, L, _ = x.shape

        x = self.x_embedder(x)
        if not self.use_position_prior:
            x = x + self.pos_embed[:, :L, :]

        c = self.t_embedder(t) + self.y_embedder(y)
        if t_pos is not None and hasattr(self, "t_pos_embedder"):
            c = c + self.t_pos_embedder(t_pos)
        if token_len is not None and self.embed_token_len:
            c = c + self.token_len_embedder(token_len.float())

        if self.use_position_prior:
            if gt_positions is not None:
                pos_embed = self.pos_to_embed(gt_positions)
                if self._prior_n_reg > 0:
                    pos_embed = pos_embed.clone()
                    pos_embed[:, : self._prior_n_reg] = 0  # registers carry no position
            else:
                pos_embed = x.new_zeros(B, L, x.shape[-1])
            x = x + pos_embed

        if valid_counts is not None:
            x = self._forward_packed(x, c, valid_counts, L)
        else:
            for block in self.blocks:
                x = block(x, c)
            x = self.final_layer(x, c)
        return x

    def _forward_packed(self, x, c, valid_counts, L):
        """Run the blocks over only the active tokens of each sample."""
        B = x.shape[0]
        mask = torch.arange(L, device=x.device).unsqueeze(0) < valid_counts.unsqueeze(1)

        x_packed = x[mask]
        c_packed = torch.repeat_interleave(c, valid_counts, dim=0)
        cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device=x.device)
        cu_seqlens[1:] = valid_counts.cumsum(0).to(torch.int32)
        max_seqlen = int(valid_counts.max().item())

        for block in self.blocks:
            x_packed = block(x_packed, c_packed, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        x_packed = self.final_layer(x_packed, c_packed, packed=True)

        out = x_packed.new_zeros(B, L, x_packed.shape[-1])
        out[mask] = x_packed
        return out

    def forward_prior(self, noisy_mask, prior_t, y, token_len=None):
        """Velocity for the cascaded mask prior; matches the ``model(x, t, **kw)`` interface.

        ``noisy_mask`` is ``(B, G, 1)`` over the fixed occupancy grid, so the prior
        always runs on a full-length sequence and needs no packing.
        """
        return self.position_prior(noisy_mask, prior_t, y, token_len=token_len)

    def forward_with_cfg(self, x, t, y, cfg_scale, valid_counts=None, t_pos=None, token_len=None,
                         gt_positions=None):
        """Classifier-free guidance over a doubled batch ``[conditional | unconditional]``."""
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)

        def _dup(v):
            return torch.cat([v, v], dim=0) if v is not None else None

        out = self.forward(
            combined, t, y,
            valid_counts=_dup(valid_counts),
            t_pos=t_pos,
            token_len=_dup(token_len),
            gt_positions=_dup(gt_positions),
        )
        cond, uncond = torch.split(out, len(out) // 2, dim=0)
        guided = uncond + cfg_scale * (cond - uncond)
        return torch.cat([guided, guided], dim=0)


def SiT_XL_2(**kwargs):
    return SiT(hidden_size=1152, depth=28, num_heads=16, **kwargs)


SiT_models = {"SiT-XL/2": SiT_XL_2}
