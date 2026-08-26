"""Reconstruct a video with the KATok tokenizer.

    python scripts/reconstruct.py assets/sample.mp4 -w weights/ -o out/

Prints how many tokens the clip cost and writes the reconstruction next to the input.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from katok.tokenizer import KATok  # noqa: E402
from katok.tokenizer.chunked import DEFAULT_OVERLAP, reconstruct_video  # noqa: E402
from katok.utils import parse_resolution, prepare, read_video, write_video  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="input video file")
    ap.add_argument("-w", "--weights", required=True, help="safetensors file or directory")
    ap.add_argument("-o", "--out", default="out", help="output directory")
    ap.add_argument("-c", "--config", default=None, help="config YAML (defaults to configs/tokenizer.yaml)")
    ap.add_argument("--resolution", default="256", help="'256', '368x640', or 'none' to keep the source size")
    ap.add_argument("--frames", type=int, default=16, help="number of frames to reconstruct")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth source frame")
    ap.add_argument("--chunk", type=int, default=None,
                    help="sliding-window length. Clips longer than the resolution's trained "
                         "envelope (64 frames at 256^2, 32 at 512^2) are split automatically; "
                         "pass a value to override, or 0 to force a single pass")
    ap.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP,
                    help="frames shared between windows (default: none)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fps", type=float, default=24.0, help="fps of the written video")
    args = ap.parse_args()

    model = KATok.from_pretrained(args.weights, config=args.config, device=args.device)

    video = read_video(args.video, n_frames=args.frames * args.stride, stride=args.stride)
    x = prepare(video, resolution=parse_resolution(args.resolution), n_frames=args.frames).to(args.device)
    print(f"input  : {tuple(x.shape)}  from {args.video}")

    _, _, T, H, W = x.shape
    channels = model.latent_processor.common_params.latent_token_dim

    recon, info = reconstruct_video(model, x, overlap=args.overlap, chunk_frames=args.chunk)
    n_active = info["total_tokens"]

    # Token accounting includes the register tokens, matching the paper's tables
    # (366.24 average out of a 514 maximum at 256^2 x 16).
    if info["chunked"]:
        length = info["plan"][0][1] - info["plan"][0][0]
        print(f"windows: {len(info['plan'])} x {length}f, overlap {args.overlap}   {info['plan']}")
        print(f"tokens : {n_active} total   {info['token_counts']}   (registers included)")
    else:
        n_total = (T // 8) * (H // 16) * (W // 16) + model.latent_processor.n_to_keep
        print(f"tokens : {n_active} / {n_total}   ({n_active / n_total:.1%} kept, registers included)")

    ratio = (T * H * W * 3) / (n_active * channels)
    mse = torch.mean((recon.clamp(-1, 1) - x) ** 2).item()
    psnr = 10 * torch.log10(torch.tensor(4.0 / mse)).item()  # data range is [-1, 1]

    print(f"comp.  : {ratio:.1f}x   at {channels} channels per token   ({n_active / T:.1f} tokens/frame)")
    print(f"psnr   : {psnr:.2f} dB")

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.video))[0]
    recon_path = os.path.join(args.out, f"{stem}_recon.mp4")
    input_path = os.path.join(args.out, f"{stem}_input.mp4")
    write_video(recon[0], recon_path, fps=args.fps)
    write_video(x[0], input_path, fps=args.fps)
    print(f"wrote  : {recon_path}\n         {input_path}")


if __name__ == "__main__":
    main()
