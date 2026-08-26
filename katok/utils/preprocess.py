"""Turning a decoded video into a model input.

The tokenizer expects ``(B, C, T, H, W)`` in ``[-1, 1]``, with ``T`` a multiple of the
encoder's temporal patch and ``H``/``W`` multiples of its spatial patch. The released
model uses ``16^2 x 8`` patches, so 256x256x16 and 512x512x32 both work as-is.
"""

import torch
import torch.nn.functional as F

# Encoder patch size (t, h, w) of the released model; see configs/tokenizer.yaml.
DEFAULT_PATCH = (8, 16, 16)


def parse_resolution(value: str | None) -> int | tuple[int, int] | None:
    """Parse a CLI resolution: ``'256'``, ``'368x640'``, or ``'none'``."""
    if value is None or value.lower() == "none":
        return None
    if "x" in value.lower():
        h, w = value.lower().split("x")
        return (int(h), int(w))
    return int(value)


def center_crop(video: torch.Tensor, aspect: float = 1.0) -> torch.Tensor:
    """Center-crop a ``(C, T, H, W)`` video to the given width/height ratio."""
    _, _, H, W = video.shape
    if W / H > aspect:
        new_w = int(round(H * aspect))
        left = (W - new_w) // 2
        return video[:, :, :, left : left + new_w]
    new_h = int(round(W / aspect))
    top = (H - new_h) // 2
    return video[:, :, top : top + new_h, :]


def resize(video: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Bicubic-resize a ``(C, T, H, W)`` video to ``(H, W) = size``."""
    resized = F.interpolate(video.permute(1, 0, 2, 3), size=size, mode="bicubic", align_corners=False)
    return resized.permute(1, 0, 2, 3)


def round_down_to_multiple(value: int, multiple: int) -> int:
    if value < multiple:
        raise ValueError(f"{value} is smaller than the patch size {multiple}")
    return (value // multiple) * multiple


def prepare(
    video: torch.Tensor,
    resolution: int | tuple[int, int] | None = 256,
    n_frames: int | None = 16,
    patch: tuple[int, int, int] = DEFAULT_PATCH,
) -> torch.Tensor:
    """Crop, resize and normalize a ``(C, T, H, W)`` video in ``[0, 1]``.

    Returns ``(1, C, T, H, W)`` in ``[-1, 1]``, ready for :meth:`KATok.encode`.

    Args:
        video: decoded frames in ``[0, 1]``.
        resolution: target ``(H, W)``, or an int for a square center crop. ``None``
            keeps the source size, trimmed down to the patch grid.
        n_frames: number of frames to keep. ``None`` keeps as many as fit the grid.
        patch: encoder patch size ``(t, h, w)``; sizes are snapped to multiples of it.
    """
    pt, ph, pw = patch
    _, T, H, W = video.shape

    if resolution is None:
        target = (round_down_to_multiple(H, ph), round_down_to_multiple(W, pw))
        video = center_crop(video, aspect=target[1] / target[0])
    else:
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        target = (round_down_to_multiple(resolution[0], ph), round_down_to_multiple(resolution[1], pw))
        video = center_crop(video, aspect=target[1] / target[0])

    if tuple(video.shape[-2:]) != target:
        video = resize(video, target)

    n = round_down_to_multiple(T if n_frames is None else min(n_frames, T), pt)
    if n_frames is not None and n < n_frames:
        raise ValueError(f"cannot use {n_frames} frames: only {T} decoded, and T must be a multiple of {pt}")
    video = video[:, :n]

    return (video * 2 - 1).clamp(-1, 1).unsqueeze(0)
