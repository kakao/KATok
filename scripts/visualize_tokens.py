"""Visualize which patches the token selector keeps.

    python scripts/visualize_tokens.py assets/sample.mp4 -w weights/ -o out/

Each latent token corresponds to one spatio-temporal patch of the input. Dropped
tokens are drawn as black tiles over the frames, so the mask can be read directly
against the content: static or homogeneous regions go dark, motion-rich ones stay.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from katok.tokenizer import KATok, token_counts  # noqa: E402
from katok.tokenizer.chunked import (  # noqa: E402
    DEFAULT_CHUNK_FRAMES,
    DEFAULT_OVERLAP,
    max_single_pass_frames,
)
from katok.utils import parse_resolution, prepare, read_video, save_frames_grid, write_video  # noqa: E402


def mask_to_pixels(mask: torch.Tensor, grid: tuple[int, int, int], shape: tuple[int, int, int]) -> torch.Tensor:
    """Expand a per-token mask to a ``(1, T, H, W)`` pixel mask.

    Args:
        mask: ``(L,)`` content-token mask, in (t h w) order.
        grid: the encoder patch grid ``(t, h, w)`` the tokens live on.
        shape: target ``(T, H, W)`` in pixels.
    """
    gt, gh, gw = grid
    T, H, W = shape
    m = mask.reshape(1, 1, gt, gh, gw)
    m = torch.nn.functional.interpolate(m, size=(T, H, W), mode="nearest")
    return m[0]


def chunked_pixel_mask(model, x, chunk_frames, overlap, shape):
    """Pixel mask for a sliding-window encode.

    A binary mask cannot be cross-faded without turning the overlap grey, so each
    frame takes the mask of whichever window dominates it. Register tokens are
    excluded, as in the single-window path.
    """
    from katok.tokenizer.chunked import encode_long, window_owner

    T, H, W = shape
    plan, latents = encode_long(model, x, chunk_frames=chunk_frames, overlap=overlap)
    owner = window_owner(plan, T)

    print(f"windows: {len(plan)} x {chunk_frames}f, overlap {overlap}   {plan}")

    per_window = []
    masks = []
    for lat in latents:
        n_reg = lat.tokens.n_registers
        _, gt, gh, gw = lat.tokens.original_shape
        content = lat.mask[0, n_reg:, 0]
        per_window.append(int(lat.mask.sum().item()))  # registers included, as in the paper
        masks.append(mask_to_pixels(content, (gt, gh, gw), (chunk_frames, H, W)).to(x.dtype))

    print(f"tokens : {sum(per_window)} total   {per_window}   ({sum(per_window) / T:.1f}/frame, registers included)")

    out = torch.zeros(1, T, H, W, dtype=x.dtype, device=x.device)
    for t in range(T):
        i = owner[t]
        out[:, t] = masks[i][:, t - plan[i][0]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="input video file")
    ap.add_argument("-w", "--weights", required=True, help="safetensors file or directory")
    ap.add_argument("-o", "--out", default="out", help="output directory")
    ap.add_argument("-c", "--config", default=None, help="config YAML (defaults to configs/tokenizer.yaml)")
    ap.add_argument("--resolution", default="256", help="'256', '368x640', or 'none' to keep the source size")
    ap.add_argument("--frames", type=int, default=16, help="number of frames")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth source frame")
    ap.add_argument("--chunk", type=int, default=None,
                    help="sliding-window length; clips longer than the resolution's trained "
                         "envelope are split automatically (see reconstruct.py)")
    ap.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP,
                    help="frames shared between windows (default: none)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--grid-every", type=int, default=2, help="frame step in the PNG contact sheet")
    ap.add_argument("--dim", type=float, default=0.0, help="brightness of dropped tiles, 0 = black")
    ap.add_argument("--suffix", default="tokens", help="output filename suffix")
    ap.add_argument("--fps", type=float, default=24.0, help="fps of the written video")
    args = ap.parse_args()

    model = KATok.from_pretrained(args.weights, config=args.config, device=args.device)

    video = read_video(args.video, n_frames=args.frames * args.stride, stride=args.stride)
    x = prepare(video, resolution=parse_resolution(args.resolution), n_frames=args.frames).to(args.device)
    _, _, T, H, W = x.shape
    print(f"input  : {tuple(x.shape)}")

    chunk = args.chunk
    limit = max_single_pass_frames(H, W)
    if chunk is None and T > limit:
        chunk = min(DEFAULT_CHUNK_FRAMES, (limit // 8) * 8)
    if chunk:
        pixel_mask = chunked_pixel_mask(model, x, chunk, args.overlap, (T, H, W))
    else:
        latents = model.encode(x)
        n_reg = latents.tokens.n_registers
        _, gt, gh, gw = latents.tokens.original_shape
        content_mask = latents.mask[0, n_reg:, 0]

        n_active = int(token_counts(latents).item()) + n_reg
        n_total = gt * gh * gw + n_reg
        print(f"grid   : {gt} x {gh} x {gw} content cells (+ {n_reg} registers)")
        print(f"tokens : {n_active} / {n_total}   ({n_active / n_total:.1%} kept, registers included)")
        print(f"per temporal chunk: {[int(c) for c in content_mask.reshape(gt, gh * gw).sum(dim=1).tolist()]}")

        pixel_mask = mask_to_pixels(content_mask, (gt, gh, gw), (T, H, W)).to(x.dtype)

    # Dropped tiles go to `--dim` in [-1, 1] space; 0.0 maps to mid grey, -1 to black.
    floor = torch.full_like(x[0], args.dim * 2 - 1)
    masked = x[0] * pixel_mask + floor * (1 - pixel_mask)

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.video))[0]
    mp4_path = os.path.join(args.out, f"{stem}_{args.suffix}.mp4")
    png_path = os.path.join(args.out, f"{stem}_{args.suffix}.png")
    write_video(masked, mp4_path, fps=args.fps)
    save_frames_grid(masked, png_path, every=args.grid_every)
    print(f"wrote  : {mp4_path}\n         {png_path}")


if __name__ == "__main__":
    main()
