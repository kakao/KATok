import logging

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from .interface import (
    DecoderOutput,
    PatchifierAbstract,
    PatchifierOutput,
    Shape4D,
    TokenData,
    UnpatchifierAbstract,
    UnpatchifierOutput,
)
from .layers import PixelShuffleNd, PixelUnshuffleNd

logpy = logging.getLogger(__name__)


class LinearPatchifier(PatchifierAbstract):
    """Split a video into 3D patches and linearly embed them.

    Implemented as a pixel-unshuffle followed by a 1x1x1 convolution, which is a plain
    linear projection over each ``scale_factors``-sized patch and keeps the whole
    pipeline differentiable.
    """

    def __init__(self, input_dim: int, hidden_dim: int, scale_factors: tuple[int, int, int]):
        super().__init__()
        self.layers = nn.Sequential(
            PixelUnshuffleNd(scale_factors, op_type="3d"),
            nn.Conv3d(input_dim * int(np.prod(scale_factors)), hidden_dim, kernel_size=1),
        )

    def forward(self, batch: dict) -> PatchifierOutput:
        # batch["vid"] is assumed to be in [-1, 1]
        vid = batch["vid"]
        with torch.autocast(device_type=vid.device.type, enabled=False):
            patches = self.layers(vid.float())
            patch_shape = tuple(patches.shape[1:])

            patches = rearrange(patches, "b c t h w -> b (t h w) c").contiguous()
            patches_output = TokenData(
                tensor=patches, original_shape=patch_shape, original_data_shape=tuple(vid.shape[1:])
            )
            return PatchifierOutput(tokens=patches_output)


class LinearUnpatchifier(UnpatchifierAbstract):
    """Project decoder tokens back to pixels (inverse of :class:`LinearPatchifier`)."""

    def __init__(self, output_dim: int, hidden_dim: int, scale_factors: tuple[int, int, int]):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(hidden_dim, output_dim * int(np.prod(scale_factors)), kernel_size=1),
            PixelShuffleNd(scale_factors, op_type="3d"),
        )

    def forward_tensor(self, decoded_tensor: torch.Tensor, patch_shape: Shape4D) -> torch.Tensor:
        with torch.autocast(device_type=decoded_tensor.device.type, enabled=False):
            patches = rearrange(
                decoded_tensor.float(),
                "b (t h w) c -> b c t h w",
                t=patch_shape[1],
                h=patch_shape[2],
                w=patch_shape[3],
            ).contiguous()
            return self.layers(patches)

    def forward(self, decoded: DecoderOutput) -> UnpatchifierOutput:
        output = self.forward_tensor(decoded.tokens.tensor, decoded.tokens.original_shape)
        return UnpatchifierOutput(recon=output)
