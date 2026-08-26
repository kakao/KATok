import logging
from typing import Any

import torch
from einops import repeat

from .blocks import EmbedND, SingleStreamBlock
from .common import get_patch_ids
from .interface import AutoEncoderParams, EncoderAbstract, EncoderOutput, PatchifierOutput, TokenData

logpy = logging.getLogger(__name__)


class RopeEncoder(EncoderAbstract):
    """Single-stream transformer encoder with 3D RoPE and register tokens.

    Consumes patch embeddings and produces the intermediate embeddings ``e_i`` that
    the adaptive token selector turns into latents and keep/drop logits. Register
    tokens are prepended and act as global anchors; they carry an identity rotation
    instead of a grid position.
    """

    def __init__(
        self,
        common_params: AutoEncoderParams,
        num_layers: int,
        n_registers: int = 0,
    ):
        super().__init__()
        self.common_params = common_params

        self.layers = torch.nn.ModuleList(
            [
                SingleStreamBlock(
                    hidden_size=self.common_params.hidden_dim,
                    num_heads=self.common_params.num_heads,
                    mlp_ratio=self.common_params.mlp_ratio,
                    qk_scale=self.common_params.qk_scale,
                    dropout=self.common_params.dropout,
                )
                for _ in range(num_layers)
            ]
        )

        patch_pe_dim = self.common_params.hidden_dim // self.common_params.num_heads
        axes_dim = patch_pe_dim // 3
        assert axes_dim % 2 == 0, "axes_dim must be divisible by 2"
        self.patch_pos_embedder = EmbedND(dim=patch_pe_dim, theta=10_000, axes_dim=[axes_dim, axes_dim, axes_dim])

        registers = self.common_params.hidden_dim**-0.5 * torch.randn(n_registers, self.common_params.hidden_dim)
        if n_registers > 0:
            self.register_tokens = torch.nn.Parameter(registers)
        else:
            self.register_buffer("register_tokens", registers)  # for compatibility

    def forward(self, patchifier_output: PatchifierOutput) -> EncoderOutput:
        patches = patchifier_output.tokens

        pe = self.prepare_pos_emb(patches)
        patches, pe = self.prepare_register_tokens(patches, pe)

        patch_tensor = patches.tensor
        for layer in self.layers:
            patch_tensor = layer(x=patch_tensor, pe=pe, attn_mask=None)

        return EncoderOutput(tokens=patches.replace_tensor(new_tensor=patch_tensor))

    def prepare_pos_emb(self, patches: TokenData, **kwargs: Any) -> torch.Tensor:
        # returns pe: (B, 1, L, D, 2, 2)
        patch_shape = patches.original_shape

        patch_id = get_patch_ids(patch_shape, device=patches.device)
        if self.common_params.rope_base_spatial_resolution is not None:
            sh = (self.common_params.rope_base_spatial_resolution[0] - 1) / (patch_shape[2] - 1)
            sw = (self.common_params.rope_base_spatial_resolution[1] - 1) / (patch_shape[3] - 1)
            patch_id[..., 1] = patch_id[..., 1] * sh
            patch_id[..., 2] = patch_id[..., 2] * sw
        patch_id = repeat(patch_id, "p d -> b p d", b=patches.bs).contiguous()

        return self.patch_pos_embedder(patch_id)

    def prepare_register_tokens(self, patches: TokenData, pe: torch.Tensor) -> tuple[TokenData, torch.Tensor]:
        n_registers = self.register_tokens.shape[0]
        if n_registers == 0:
            return patches, pe

        register_tokens = repeat(self.register_tokens, "r d -> b r d", b=patches.bs).contiguous()
        patch_tensor = torch.cat([register_tokens, patches.tensor], dim=1)

        # Registers get an identity rotation, i.e. no positional information.
        pe_eye = repeat(
            torch.eye(2, device=pe.device), "i j -> b 1 r c i j", b=pe.shape[0], r=n_registers, c=pe.shape[3]
        )
        pe_with_register = torch.cat([pe_eye, pe], dim=2)

        patches_with_register = patches.replace_tensor(new_tensor=patch_tensor, n_registers=n_registers)
        return patches_with_register, pe_with_register
