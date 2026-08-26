import logging

import numpy as np
import torch
import torch.nn as nn
from einops import repeat

from .blocks import DoubleStreamBlock, EmbedND
from .common import get_patch_ids
from .interface import (
    AutoEncoderParams,
    DecoderAbstract,
    DecoderOutput,
    LatentProcessorOutput,
    Shape4D,
    TokenData,
)

logpy = logging.getLogger(__name__)


class RopeDecoder(DecoderAbstract):
    """Double-stream transformer decoder driven by learnable query tokens.

    One stream carries repeated learnable query tokens with 3D RoPE, the other carries
    the sparse latent tokens with no positional encoding (all-zero position ids). The
    soft token-drop mask enters as an additive bias on the latent keys, so dropped
    tokens stop contributing to decoding.

    Asymmetric coarse-to-fine decoding is controlled by the query grid: ``upsample``
    doubles the encoder patch grid along every axis (the ``16^2 x 8`` encoder patch
    becomes an ``8^2 x 4`` decoder patch), and ``mask_token_shape`` sets it explicitly.
    """

    def __init__(
        self,
        common_params: AutoEncoderParams,
        num_layers: int,
        use_latent_norm: bool = True,
        soft_attn_mask: bool = True,
        use_kv_bias_trick: bool = False,
        upsample: bool = False,
        mask_token_shape: list[int] | None = None,
    ):
        super().__init__()
        self.common_params = common_params
        self.soft_attn_mask = soft_attn_mask
        self.use_kv_bias_trick = use_kv_bias_trick
        self.upsample = upsample
        self.mask_token_shape = mask_token_shape

        self.layers = nn.ModuleList(
            [
                DoubleStreamBlock(
                    hidden_size=self.common_params.hidden_dim,
                    num_heads=self.common_params.num_heads,
                    mlp_ratio=self.common_params.mlp_ratio,
                    qkv_bias=False,
                    dropout=self.common_params.dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.token_decoder = nn.Linear(self.common_params.latent_token_dim, self.common_params.hidden_dim)
        if use_latent_norm:
            self.latent_norm = nn.LayerNorm(self.common_params.hidden_dim)
        else:
            self.latent_norm = nn.Identity()

        patch_pe_dim = self.common_params.hidden_dim // self.common_params.num_heads
        axes_dim = patch_pe_dim // 3
        assert axes_dim % 2 == 0, "axes_dim must be divisible by 2"
        self.patch_pos_embedder = EmbedND(dim=patch_pe_dim, theta=10_000, axes_dim=[axes_dim, axes_dim, axes_dim])

        self.mask_tokens = nn.Parameter(
            self.common_params.hidden_dim**-0.5 * torch.randn(self.common_params.hidden_dim)
        )

        # the latent stream output of the last block is never read
        self.layers[-1].txt_mlp = nn.Identity()
        self.layers[-1].txt_attn.proj = nn.Identity()

    def forward(self, latent_processor_output: LatentProcessorOutput) -> DecoderOutput:
        tokens = latent_processor_output.tokens
        latent_tokens = tokens.tensor  # (B, L, D)
        latent_mask = latent_processor_output.mask  # token drop mask, (B, L, 1)
        orig_shape = tokens.original_shape

        latent_tokens = self.token_decoder(latent_tokens)
        latent_tokens = self.latent_norm(latent_tokens)

        patch_shape = self.query_patch_shape(orig_shape)

        pe = self.prepare_pos_emb(tokens, latent_mask, patch_shape)
        attn_mask = self.prepare_attention_mask(
            latent_mask, patch_shape, dtype=latent_tokens.dtype, device=latent_tokens.device
        )

        mask_tokens = self.prepare_mask_tokens(patch_shape, tokens.bs)

        for layer in self.layers:
            mask_tokens, latent_tokens = layer(
                img=mask_tokens,
                txt=latent_tokens,
                pe=pe,
                attn_mask=attn_mask,
                use_kv_bias_trick=self.use_kv_bias_trick,
            )

        return DecoderOutput(
            tokens=TokenData(
                tensor=mask_tokens,
                original_shape=patch_shape,
                original_data_shape=tokens.original_data_shape,
            )
        )

    def query_patch_shape(self, orig_shape: Shape4D) -> Shape4D:
        """Grid the learnable query tokens are laid out on."""
        if self.mask_token_shape is not None:
            T, H, W = self.mask_token_shape
            return (orig_shape[0], T, H, W)
        if self.upsample:
            return (orig_shape[0], orig_shape[1] * 2, orig_shape[2] * 2, orig_shape[3] * 2)
        return orig_shape

    def prepare_pos_emb(self, tokens: TokenData, latent_mask: torch.Tensor, patch_shape: Shape4D) -> torch.Tensor:
        # returns pe: (B, 1, L + P, D, 2, 2)
        B, device, dtype = tokens.bs, tokens.device, tokens.dtype

        patch_id = get_patch_ids(patch_shape, device=device)
        if self.common_params.rope_base_spatial_resolution is not None:
            sh = (self.common_params.rope_base_spatial_resolution[0] - 1) / (patch_shape[2] - 1)
            sw = (self.common_params.rope_base_spatial_resolution[1] - 1) / (patch_shape[3] - 1)
            patch_id[..., 1] = patch_id[..., 1] * sh
            patch_id[..., 2] = patch_id[..., 2] * sw
        patch_id = repeat(patch_id, "p d -> b p d", b=B).contiguous()

        # latent tokens carry no positional information
        latent_id = torch.zeros(
            (latent_mask.shape[0], latent_mask.shape[1], patch_id.shape[2]), device=device, dtype=dtype
        )

        id_all = torch.cat([latent_id, patch_id], dim=1)
        return self.patch_pos_embedder(id_all)

    def prepare_attention_mask(
        self, latent_mask: torch.Tensor, patch_shape: Shape4D, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Build the additive attention bias from the token-drop mask.

        With ``soft_attn_mask`` the bias is ``log(m_j + eps)``, so the soft mask acts as
        a differentiable gate; otherwise dropped keys are hard-masked with ``-inf``.
        Query tokens are never masked. Returns ``(B, L + P)`` for the KV bias trick, or
        the full ``(B, 1, L + P, L + P)`` mask otherwise.
        """
        if self.soft_attn_mask:
            attn_mask = torch.log(latent_mask + 1e-10)
        else:
            attn_mask = torch.zeros_like(latent_mask).masked_fill(latent_mask == 0, float("-inf"))

        L, P = latent_mask.shape[1], int(np.prod(patch_shape[1:]))

        out = torch.cat([attn_mask, torch.zeros((latent_mask.shape[0], P, 1), device=device, dtype=dtype)], dim=1)

        if self.use_kv_bias_trick:
            return out.squeeze(-1)

        return repeat(out, "b l 1 -> b 1 p l", p=L + P)

    def prepare_mask_tokens(self, patch_shape: Shape4D, bs: int) -> torch.Tensor:
        return self.mask_tokens[None, None, :].repeat(bs, int(np.prod(patch_shape[1:])), 1)  # d -> b p d
