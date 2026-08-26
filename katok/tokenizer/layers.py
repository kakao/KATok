from typing import Union

import torch
from einops import rearrange


class PixelShuffleNd(torch.nn.Module):
    def __init__(self, upscale_factor: Union[int, tuple], op_type: str = "2d"):
        super().__init__()
        assert op_type in ["1d", "2d", "3d"]

        if isinstance(upscale_factor, int):
            if op_type == "1d":
                upscale_factor = (upscale_factor,)
            elif op_type == "2d":
                upscale_factor = (upscale_factor, upscale_factor)
            elif op_type == "3d":
                upscale_factor = (upscale_factor, upscale_factor, upscale_factor)

        self.upscale_factor = upscale_factor
        self.op_type = op_type

    def forward(self, x):
        if self.op_type == "1d":
            return rearrange(x, "b (c r1) h -> b c (h r1)", r1=self.upscale_factor[0]).contiguous()
        elif self.op_type == "2d":
            return rearrange(
                x, "b (c r1 r2) h w -> b c (h r1) (w r2)", r1=self.upscale_factor[0], r2=self.upscale_factor[1]
            ).contiguous()
        elif self.op_type == "3d":
            return rearrange(
                x,
                "b (c r1 r2 r3) d h w -> b c (d r1) (h r2) (w r3)",
                r1=self.upscale_factor[0],
                r2=self.upscale_factor[1],
                r3=self.upscale_factor[2],
            ).contiguous()


class PixelUnshuffleNd(torch.nn.Module):
    def __init__(self, downscale_factor: Union[int, tuple], op_type: str = "2d"):
        super().__init__()
        assert op_type in ["1d", "2d", "3d"]

        if isinstance(downscale_factor, int):
            if op_type == "1d":
                downscale_factor = (downscale_factor,)
            elif op_type == "2d":
                downscale_factor = (downscale_factor, downscale_factor)
            elif op_type == "3d":
                downscale_factor = (downscale_factor, downscale_factor, downscale_factor)

        self.downscale_factor = downscale_factor
        self.op_type = op_type

    def forward(self, x):
        if self.op_type == "1d":
            return rearrange(x, "b c (h r1) -> b (c r1) h", r1=self.downscale_factor[0]).contiguous()
        elif self.op_type == "2d":
            return rearrange(
                x, "b c (h r1) (w r2) -> b (c r1 r2) h w", r1=self.downscale_factor[0], r2=self.downscale_factor[1]
            ).contiguous()
        elif self.op_type == "3d":
            return rearrange(
                x,
                "b c (d r1) (h r2) (w r3) -> b (c r1 r2 r3) d h w",
                r1=self.downscale_factor[0],
                r2=self.downscale_factor[1],
                r3=self.downscale_factor[2],
            ).contiguous()
