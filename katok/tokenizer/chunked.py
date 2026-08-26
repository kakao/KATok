"""Tokenizing videos longer than the tokenizer's temporal window.

How many frames fit in one pass depends on the resolution: multi-resolution training
paired large spatial sizes with short clips (64 frames only at 256^2, 16 at 512^2 --
see :data:`KNOWN_GOOD_SHAPES`). Past that envelope the 3D RoPE indices were never
trained, and reconstruction does not degrade gracefully -- content ends up visibly
displaced rather than merely blurred.

:func:`reconstruct_video` therefore switches to a sliding window once a clip exceeds
its resolution's envelope. Each window is encoded and decoded on its own, so it stays
inside the trained range, and neighbouring windows can be cross-faded over an overlap.

Two costs follow from windows being independent:

* Token savings come from redundancy *within* a window, so a shorter window sees less
  of it and the per-frame token count rises. Compression and context length trade off
  directly.
* Overlapping frames are tokenized twice, and that is paid for directly.

Overlap defaults to zero. The tokenizer is deterministic and adjacent windows
reconstruct their shared frames near-identically, so seams stay at the noise floor
without blending; overlap is cheap insurance for clips where neighbouring windows
might disagree, such as fast motion or a cut landing on a boundary.

This is a practical way to run long inputs, not the adaptive-compression result the
paper reports.
"""

import torch

from .model import KATok

# (H, W, T) input configurations the tokenizer is known to handle in one pass: the
# multi-resolution training shapes, plus 512^2 x 32 -- the reconstruction setting the
# paper evaluates, which holds up despite exceeding the trained 512^2 x 16. The frame
# budget shrinks as resolution grows; 64 frames were only ever seen at 256^2.
KNOWN_GOOD_SHAPES = [
    (256, 256, 16), (256, 256, 24), (256, 256, 32), (256, 256, 64),
    (512, 512, 8), (512, 512, 16), (512, 512, 32),
    (368, 640, 16), (640, 368, 16),
    (240, 416, 32), (416, 240, 32),
]

# Preferred window length when a clip has to be split, provided the resolution allows
# it. Short enough to sit inside the envelope, long enough to exploit redundancy.
DEFAULT_CHUNK_FRAMES = 32

# Windows agree closely at their boundaries, so no cross-fade is needed by default.
DEFAULT_OVERLAP = 0


def max_single_pass_frames(height: int, width: int) -> int:
    """Longest clip that runs in one pass at this resolution.

    A shape whose spatial extent covers the input also covers its position indices,
    so the limit is the largest frame count among covering shapes. Inputs larger than
    anything in the table fall back to the most conservative trained length.
    """
    fits = [t for h, w, t in KNOWN_GOOD_SHAPES if h >= height and w >= width]
    return max(fits) if fits else min(t for _, _, t in KNOWN_GOOD_SHAPES)


def chunk_plan(n_frames: int, chunk_frames: int, overlap: int, patch_t: int = 8) -> list[tuple[int, int]]:
    """Lay out sliding windows over ``n_frames``.

    Args:
        n_frames: total frames to cover.
        chunk_frames: window length; must be a multiple of ``patch_t``.
        overlap: frames shared between neighbouring windows.
        patch_t: encoder temporal patch size. Window length must be a multiple of it.

    Returns:
        ``(start, end)`` frame ranges covering ``[0, n_frames)``. The final window is
        pulled back to end exactly at ``n_frames``, so it may overlap its predecessor
        by more than ``overlap``.
    """
    if chunk_frames % patch_t:
        raise ValueError(f"chunk_frames={chunk_frames} must be a multiple of the temporal patch {patch_t}")
    if not 0 <= overlap < chunk_frames:
        raise ValueError(f"overlap={overlap} must be in [0, {chunk_frames})")
    if n_frames < chunk_frames:
        raise ValueError(f"video has {n_frames} frames, shorter than one {chunk_frames}-frame window")

    stride = chunk_frames - overlap
    starts = list(range(0, n_frames - chunk_frames + 1, stride))
    if starts[-1] + chunk_frames < n_frames:
        starts.append(n_frames - chunk_frames)  # pull the tail back so nothing is missed
    return [(s, s + chunk_frames) for s in starts]


def blend_weights(plan: list[tuple[int, int]], n_frames: int, device=None,
                  dtype=torch.float32) -> list[torch.Tensor]:
    """Per-frame cross-fade weights, one ``(chunk_frames,)`` tensor per window.

    Each window ramps up where it overlaps its left neighbour and down where it
    overlaps its right one, using a raised-cosine taper. Weights are normalized so
    they sum to 1 at every frame, which keeps brightness flat even where the pulled-back
    final window overlaps more than the others.
    """
    windows = []
    for i, (start, end) in enumerate(plan):
        length = end - start
        w = torch.ones(length, device=device, dtype=dtype)

        if i > 0:
            left = max(0, plan[i - 1][1] - start)  # frames shared with the previous window
            if left > 1:
                ramp = torch.linspace(0, torch.pi, left, device=device, dtype=dtype)
                w[:left] = (1 - torch.cos(ramp)) / 2
        if i < len(plan) - 1:
            right = max(0, end - plan[i + 1][0])
            if right > 1:
                ramp = torch.linspace(0, torch.pi, right, device=device, dtype=dtype)
                w[length - right :] = (1 + torch.cos(ramp)) / 2
        windows.append(w)

    total = torch.zeros(n_frames, device=device, dtype=dtype)
    for (start, end), w in zip(plan, windows):
        total[start:end] += w
    total = total.clamp(min=1e-8)

    return [w / total[start:end] for (start, end), w in zip(plan, windows)]


@torch.no_grad()
def encode_long(model: KATok, video: torch.Tensor, chunk_frames: int = 24, overlap: int = 8,
                patch_t: int = 8, sample: bool = False):
    """Encode a long video window by window, without decoding.

    Returns ``(plan, latents)`` where ``latents[i]`` is the
    :class:`LatentProcessorOutput` for window ``plan[i]``. Useful for inspecting token
    allocation across a long clip.
    """
    if video.ndim != 5 or video.shape[0] != 1:
        raise ValueError(f"expected a single video of shape (1, C, T, H, W), got {tuple(video.shape)}")

    plan = chunk_plan(video.shape[2], chunk_frames, overlap, patch_t)
    return plan, [model.encode(video[:, :, s:e], sample=sample) for s, e in plan]


def window_owner(plan: list[tuple[int, int]], n_frames: int) -> list[int]:
    """Index of the window whose blend weight is largest at each frame.

    Overlapping frames are reconstructed from two windows at once. For visualizing a
    per-frame quantity that is not blendable -- a binary token mask, say -- attribute
    each frame to the window that dominates it.
    """
    weights = blend_weights(plan, n_frames)
    best = torch.full((n_frames,), -1.0)
    owner = [0] * n_frames
    for i, ((start, end), w) in enumerate(zip(plan, weights)):
        better = w > best[start:end]
        best[start:end] = torch.where(better, w, best[start:end])
        for j in better.nonzero().flatten().tolist():
            owner[start + j] = i
    return owner


@torch.no_grad()
def reconstruct_video(model: KATok, video: torch.Tensor, overlap: int = DEFAULT_OVERLAP,
                      chunk_frames: int | None = None, patch_t: int = 8, sample: bool = False):
    """Reconstruct a video of any length, splitting it only when necessary.

    Clips within the resolution's known-good envelope (:func:`max_single_pass_frames`)
    go through the model in one pass -- 64 frames at 256^2, but only 32 at 512^2, where
    longer clips were never trained. Anything longer is reconstructed with a sliding
    window sized to fit that envelope.

    Args:
        model: a loaded :class:`KATok`.
        video: ``(1, C, T, H, W)`` in ``[-1, 1]``.
        overlap: frames shared between windows. Defaults to none.
        chunk_frames: window length. ``None`` picks the largest of
            :data:`DEFAULT_CHUNK_FRAMES` and the resolution's envelope that fits; pass
            a value to force a window length, or ``0`` to force a single pass.
        patch_t: encoder temporal patch size.
        sample: draw latents from the posterior instead of using its mean.

    Returns:
        ``(recon, info)``. ``info`` reports ``chunked``, the window ``plan``, per-window
        ``token_counts``, ``total_tokens``, ``frames`` and ``tokens_per_frame``.
    """
    n_frames, height, width = video.shape[2], video.shape[3], video.shape[4]
    limit = max_single_pass_frames(height, width)
    force_single = chunk_frames == 0
    needs_split = n_frames > limit and not force_single

    if not needs_split:
        recon, latents = model.reconstruct(video, sample=sample)
        # Counts include the register tokens, matching the paper's accounting.
        n = int(latents.mask.sum().item())
        return recon, {
            "chunked": False,
            "plan": [(0, n_frames)],
            "token_counts": [n],
            "total_tokens": n,
            "frames": n_frames,
            "tokens_per_frame": n / n_frames,
        }

    if chunk_frames is None:
        # As long a window as the resolution's envelope allows, capped at the default.
        chunk_frames = min(DEFAULT_CHUNK_FRAMES, (limit // patch_t) * patch_t)

    recon, info = reconstruct_long(
        model, video, chunk_frames=chunk_frames,
        overlap=overlap, patch_t=patch_t, sample=sample,
    )
    return recon, {"chunked": True, **info}


@torch.no_grad()
def reconstruct_long(model: KATok, video: torch.Tensor, chunk_frames: int = DEFAULT_CHUNK_FRAMES,
                     overlap: int = DEFAULT_OVERLAP, patch_t: int = 8, sample: bool = False,
                     return_chunks: bool = False):
    """Reconstruct a long video with a sliding window.

    Args:
        model: a loaded :class:`KATok`.
        video: ``(1, C, T, H, W)`` in ``[-1, 1]``.
        chunk_frames: window length, a multiple of ``patch_t``.
        overlap: frames shared between neighbouring windows.
        patch_t: encoder temporal patch size.
        sample: draw latents from the posterior instead of using its mean.
        return_chunks: also return the per-window latents.

    Returns:
        ``(recon, info)`` where ``recon`` matches ``video``'s shape and ``info`` holds
        the plan, per-window token counts, and the total tokens spent.
    """
    if video.ndim != 5 or video.shape[0] != 1:
        raise ValueError(f"expected a single video of shape (1, C, T, H, W), got {tuple(video.shape)}")

    n_frames = video.shape[2]
    plan = chunk_plan(n_frames, chunk_frames, overlap, patch_t)
    weights = blend_weights(plan, n_frames, device=video.device, dtype=video.dtype)

    recon = torch.zeros_like(video)
    counts, chunks = [], []

    for (start, end), w in zip(plan, weights):
        latents = model.encode(video[:, :, start:end], sample=sample)
        piece = model.decode(latents)

        recon[:, :, start:end] += piece * w.view(1, 1, -1, 1, 1)

        # Register tokens count toward each window's cost, as in the paper's tables.
        counts.append(int(latents.mask.sum().item()))
        if return_chunks:
            chunks.append(latents)

    info = {
        "plan": plan,
        "token_counts": counts,
        "total_tokens": sum(counts),
        "frames": n_frames,
        "tokens_per_frame": sum(counts) / n_frames,
    }
    if return_chunks:
        info["chunks"] = chunks
    return recon, info
