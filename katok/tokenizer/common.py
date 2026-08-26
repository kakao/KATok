import torch
from einops import rearrange

from .interface import Shape4D


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Sample from a diagonal Gaussian via the reparameterization trick."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu


def get_patch_ids(patch_shape: Shape4D, device: torch.device) -> torch.Tensor:
    """Build (t, h, w) grid coordinates for every patch, flattened in (t h w) order.

    Returns a ``(t*h*w, 3)`` tensor consumed by the 3D RoPE embedder.
    """
    _, t, h, w = patch_shape
    ids = torch.zeros(t, h, w, 3, device=device)
    ids[..., 0] = ids[..., 0] + torch.arange(t, device=device)[:, None, None]
    ids[..., 1] = ids[..., 1] + torch.arange(h, device=device)[None, :, None]
    ids[..., 2] = ids[..., 2] + torch.arange(w, device=device)[None, None, :]

    return rearrange(ids, "t h w d -> (t h w) d").contiguous()
