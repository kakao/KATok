"""Occupancy-grid utilities for the cascaded mask prior.

The cascaded variant decouples *where* tokens go from *what* they contain. A small
flow-matching model generates an occupancy mask over a fixed 3D grid -- for the
released model, 2 x 16 x 16 = 512 cells, matching the tokenizer's patch grid at
256x256x16 -- and the selected cells become the positional conditioning for the
content model.

Masks live in ``[-1, +1]``: ``+1`` means a token occupies that cell.
"""

import torch


def grid_center_coords(grid: tuple[int, int, int]) -> torch.Tensor:
    """Normalized ``(t, h, w)`` centers of every grid cell, ``(G, 3)`` in ``[-1, 1]``.

    Cells are laid out in ``(t, h, w)`` scan order, matching the tokenizer's
    flattening of its patch grid.
    """
    axes = [torch.linspace(-1, 1, n) if n > 1 else torch.zeros(1) for n in grid]
    mt, mh, mw = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([mt, mh, mw], dim=-1).reshape(-1, 3)


def coords_to_grid_mask(positions: torch.Tensor, grid: tuple[int, int, int],
                        valid_counts: torch.Tensor | None = None) -> torch.Tensor:
    """Rasterize continuous positions into a ``{-1, +1}`` occupancy mask.

    Args:
        positions: ``(B, L, 3)`` in ``[-1, 1]``.
        grid: ``(t, h, w)`` cell counts.
        valid_counts: ``(B,)`` active tokens per sample; padding slots are ignored.

    Returns:
        ``(B, G)`` mask with ``+1`` on occupied cells.
    """
    B, L, _ = positions.shape
    gt, gh, gw = grid
    G = gt * gh * gw

    pos = positions.clamp(-1, 1)
    idx = []
    for axis, size in enumerate((gt, gh, gw)):
        span = max(size - 1, 0)
        idx.append(torch.round((pos[..., axis] + 1) / 2 * span).long().clamp(0, span))
    flat_idx = idx[0] * (gh * gw) + idx[1] * gw + idx[2]  # (B, L)

    if valid_counts is not None:
        seq_idx = torch.arange(L, device=positions.device).unsqueeze(0)
        # Padding slots are routed to an overflow bin that is dropped afterwards.
        flat_idx = torch.where(seq_idx < valid_counts.unsqueeze(1), flat_idx, G)

    counts = positions.new_zeros(B, G + 1)
    counts.scatter_add_(1, flat_idx, torch.ones_like(counts[:, :L]))
    return torch.where(counts[:, :G] > 0, 1.0, -1.0)


def mask_to_positions(soft_mask: torch.Tensor, grid_coords: torch.Tensor,
                      token_counts: torch.Tensor) -> torch.Tensor:
    """Pick the ``k`` most-occupied cells per sample and return their centers.

    Top-k gives cells in descending mask value, which would place an arbitrary cell
    at slot 0. The content model was trained on positions in spatial-scan order, so
    the selection is re-sorted by flat grid index before being returned.

    Args:
        soft_mask: ``(B, G)`` or ``(B, G, 1)`` generated mask values.
        grid_coords: ``(G, 3)`` cell centers.
        token_counts: ``(B,)`` how many cells to keep per sample.

    Returns:
        ``(B, max_k, 3)`` positions, zero-padded beyond each sample's count.
    """
    if soft_mask.dim() == 3:
        soft_mask = soft_mask.squeeze(-1)

    G = soft_mask.shape[1]
    max_k = int(token_counts.max().item())
    _, topk_idx = torch.topk(soft_mask, max_k, dim=1)

    seq_idx = torch.arange(max_k, device=soft_mask.device).unsqueeze(0)
    valid = seq_idx < token_counts.unsqueeze(1)

    # Push filler entries past the end so they sort last, then zero them out.
    sort_keys = torch.where(valid, topk_idx, torch.full_like(topk_idx, G))
    sorted_idx, _ = sort_keys.sort(dim=1)
    sorted_idx = torch.where(sorted_idx == G, torch.zeros_like(sorted_idx), sorted_idx)

    positions = grid_coords[sorted_idx]
    return positions * valid.unsqueeze(-1).to(positions.dtype)


def prepend_register_positions(content_pos: torch.Tensor, n_registers: int) -> torch.Tensor:
    """Insert zero positions for the leading register tokens."""
    if n_registers == 0:
        return content_pos.clamp(-1, 1)
    B, _, _ = content_pos.shape
    out = content_pos.new_zeros(B, content_pos.shape[1] + n_registers, 3)
    out[:, n_registers:, :] = content_pos
    return out.clamp(-1, 1)
