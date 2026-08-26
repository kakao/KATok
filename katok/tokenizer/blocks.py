"""Transformer blocks and attention backends.

The block design follows FLUX (https://github.com/black-forest-labs/flux) with the
modulation layers removed, as described in the paper: a single-stream encoder and a
double-stream decoder, both using 3D rotary positional embeddings.

Attention runs on plain PyTorch SDPA by default. If ``flash_attn`` (or
``flash_attn_interface`` for FlashAttention-3) is installed it is used automatically
for the paths where it is faster or more memory efficient; results are equivalent
either way. Set ``ATTN_BACKEND=sdpa`` to force the pure-PyTorch path.
"""

import logging
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

os.environ.setdefault("TORCH_CUDNN_SDPA_ENABLED", "1")

_CAUSAL_DEFAULT = False
_FORCE_BACKEND = os.getenv("ATTN_BACKEND", "").lower()  # "fa3" | "sdpa" | ""

logger = logging.getLogger(__name__)


# -------------------------
# Optional FlashAttention
# -------------------------


def _load_flash_attn_func():
    """Return ``flash_attn_func`` if FlashAttention-2 is installed, else ``None``."""
    try:
        from flash_attn import flash_attn_func

        return flash_attn_func
    except Exception:
        return None


def _load_flash_attn3_func():
    """Return a FlashAttention-3 (or -2) entry point if installed, else ``None``."""
    try:
        import flash_attn_interface as fai

        return fai.flash_attn_func
    except Exception:
        pass
    try:
        from flash_attn.flash_attn_interface import flash_attn_func

        return flash_attn_func
    except Exception:
        return None


_FLASH_ATTN_FUNC = _load_flash_attn_func()
_FLASH_ATTN3_FUNC = _load_flash_attn3_func()


# -------------------------
# Backend helpers
# -------------------------


def _is_hopper() -> bool:
    """True on Hopper (H100/H200, compute capability 9.x) or newer."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


def _choose_backend(attn_mask: torch.Tensor | None) -> str:
    """env override > FA3 (no mask, Hopper, installed) > SDPA."""
    if _FORCE_BACKEND in {"fa3", "sdpa"}:
        return _FORCE_BACKEND
    if attn_mask is None and _is_hopper() and _FLASH_ATTN3_FUNC is not None:
        return "fa3"
    return "sdpa"


# -------------------------
# Implementations
# -------------------------


def _flashattn3(q, k, v, dropout, is_causal):
    q3 = rearrange(q, "b h l d -> b l h d")
    k3 = rearrange(k, "b h l d -> b l h d")
    v3 = rearrange(v, "b h l d -> b l h d")
    out = _FLASH_ATTN3_FUNC(
        q3,
        k3,
        v3,
        dropout_p=dropout,
        causal=is_causal,
        softmax_scale=None,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
    )
    return rearrange(out, "b l h d -> b h l d")


@torch._dynamo.disable()
def _sdpa_flash(q, k, v, dropout, attn_mask, is_causal, **attn_kwargs):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout, attn_mask=attn_mask, is_causal=is_causal, **attn_kwargs
        )


@torch._dynamo.disable()
def _sdpa_efficient(q, k, v, dropout, attn_mask, is_causal, **attn_kwargs):
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        return F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout, attn_mask=attn_mask, is_causal=is_causal, **attn_kwargs
        )


def _sdpa_math(q, k, v, dropout, attn_mask, is_causal, **attn_kwargs):
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout, attn_mask=attn_mask, is_causal=is_causal, **attn_kwargs
        )


def _sdpa(q, k, v, dropout, attn_mask, is_causal, **attn_kwargs):
    """Run SDPA, falling back through the backends that accept the given mask."""
    if attn_mask is not None:
        attn_mask = attn_mask.contiguous()
        try:
            return _sdpa_efficient(q, k, v, dropout, attn_mask, is_causal, **attn_kwargs)
        except RuntimeError:
            return _sdpa_math(q, k, v, dropout, attn_mask, is_causal, **attn_kwargs)

    for fn in (_sdpa_flash, _sdpa_efficient, _sdpa_math):
        try:
            return fn(q, k, v, dropout, None, is_causal, **attn_kwargs)
        except RuntimeError:
            continue
    raise RuntimeError("No SDPA backend accepted the given inputs")


@torch._dynamo.disable()
def flashattn_kv_bias(
    q: torch.Tensor,  # (B, H, Sq, d)
    k: torch.Tensor,  # (B, H, Sk, d)
    v: torch.Tensor,  # (B, H, Sk, d)
    kv_bias: torch.Tensor,  # (B, Sk)   key-only logit bias
    dropout_p: float = 0.0,
    causal: bool = False,
):
    """Additive key-only attention bias without materializing an N x N mask.

    The soft token-drop mask enters decoder attention as a per-key logit shift
    ``b_j = log(m_j + eps)``. Rather than building a full ``(B, 1, Sq, Sk)`` mask,
    the q/k embeddings are augmented as ``q_i <- [q_i, 1]`` and
    ``k_j <- [k_j, sqrt(d) * b_j]``, so the extra channel contributes exactly ``b_j``
    to every attention logit. This keeps the FlashAttention kernel usable.
    """
    q = rearrange(q, "b h l d -> b l h d")
    k = rearrange(k, "b h l d -> b l h d")
    v = rearrange(v, "b h l d -> b l h d")

    def _ensure_half(x):
        return x if x.dtype in (torch.bfloat16, torch.float16) else x.bfloat16()

    B, L, H, d = k.shape
    scale = d ** (-0.5)

    if tuple(kv_bias.shape) != (B, L):
        raise ValueError("`kv_bias` must have shape (B, L)")

    # Expects kv_bias: (B, Sk)
    b_k = _ensure_half(kv_bias / scale).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, -1)

    pad = (-(d + 1)) % 8
    q = _ensure_half(q)
    q_aug = torch.cat([q, torch.ones_like(q[..., :1]), torch.zeros_like(q[..., :1]).expand(-1, -1, -1, pad)], dim=-1)
    k_aug = torch.cat([_ensure_half(k), b_k, torch.zeros_like(b_k).expand(-1, -1, -1, pad)], dim=-1)
    v_pad = torch.nn.functional.pad(_ensure_half(v), (0, pad + 1))

    # Run FlashAttention-2 once with softmax_scale based on original d
    out_aug = _FLASH_ATTN_FUNC(q_aug, k_aug, v_pad, dropout_p=dropout_p, softmax_scale=scale, causal=causal)
    out_aug = out_aug[..., :d].to(dtype=v.dtype)
    return rearrange(out_aug, "b l h d -> b h l d")


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pe: torch.Tensor,
    pe_k: torch.Tensor | None = None,
    dropout: float = 0.0,
    attn_mask: torch.Tensor | None = None,
    use_kv_bias_trick: bool = False,
    **attn_kwargs: Any,
) -> torch.Tensor:
    q, k = apply_rope(q, k, pe, pe_k)

    is_causal = _CAUSAL_DEFAULT

    if use_kv_bias_trick:
        if _FLASH_ATTN_FUNC is not None:
            x = flashattn_kv_bias(q, k, v, attn_mask, dropout, is_causal, **attn_kwargs)
            return rearrange(x, "B H L D -> B L (H D)")
        # Without FlashAttention, apply the same key-only bias as an additive SDPA
        # mask. It broadcasts over the query axis, so this is exactly equivalent.
        attn_mask = attn_mask[:, None, None, :]

    if attn_mask is None and _choose_backend(attn_mask) == "fa3":
        try:
            x = _flashattn3(q, k, v, dropout, is_causal)
        except Exception:
            x = _sdpa(q, k, v, dropout, None, is_causal, **attn_kwargs)
    else:
        x = _sdpa(q, k, v, dropout, attn_mask, is_causal, **attn_kwargs)

    return rearrange(x, "B H L D -> B L (H D)")


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor, freqs_cis_k: Tensor | None = None) -> tuple[Tensor, Tensor]:
    if freqs_cis_k is None:
        freqs_cis_k = freqs_cis
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis_k[..., 0] * xk_[..., 0] + freqs_cis_k[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


class EmbedND(nn.Module):
    """3D rotary positional embedding: one RoPE band per (t, h, w) axis."""

    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat([rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)], dim=-3)

        return emb.unsqueeze(1)


# flux layers


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe, attn_mask=attn_mask)
        x = self.proj(x)
        return x


class SingleStreamBlock(nn.Module):
    """FLUX single-stream block with parallel attention/MLP, modulation removed.

    Used by the encoder. See https://arxiv.org/abs/2302.05442.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.dropout = dropout

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")

    def forward(self, x: Tensor, pe: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        x_mod = self.pre_norm(x)

        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # compute attention
        dropout = self.dropout if self.training else 0.0
        attn = attention(q, k, v, pe=pe, dropout=dropout, attn_mask=attn_mask)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))

        return x + output


class DoubleStreamBlock(nn.Module):
    """FLUX double-stream block without modulation.

    Used by the decoder: ``img`` carries the learnable query tokens (with 3D RoPE)
    and ``txt`` carries the latent tokens (no positional encoding). Both streams
    keep their own projections but attend over the concatenated sequence.
    """

    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False, dropout: float = 0.0
    ):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        pe: Tensor,
        attn_mask: Tensor | None = None,
        use_kv_bias_trick: bool = False,
    ) -> tuple[Tensor, Tensor]:
        # prepare image (query token) stream for attention
        img_normed = self.img_norm1(img)
        img_qkv = self.img_attn.qkv(img_normed)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt (latent token) stream for attention
        txt_normed = self.txt_norm1(txt)
        txt_qkv = self.txt_attn.qkv(txt_normed)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention over the concatenated sequence
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        dropout = self.dropout if self.training else 0.0
        attn = attention(q, k, v, pe=pe, dropout=dropout, attn_mask=attn_mask, use_kv_bias_trick=use_kv_bias_trick)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img blocks
        img = img + self.img_attn.proj(img_attn)
        img = img + self.img_mlp(self.img_norm2(img))

        # calculate the txt blocks
        txt = txt + self.txt_attn.proj(txt_attn)
        txt = txt + self.txt_mlp(self.txt_norm2(txt))

        return img, txt
