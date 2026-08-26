"""Generate videos with KATok.

    python scripts/generate.py -g weights/cascade/ -t weights/tokenizer/ -o out/ \\
        --classes 7 21 --tokens 370

A token budget must be chosen; there is no default. ``--tokens`` fixes one count for
every sample and doubles as a control signal -- fewer tokens give simpler, low-motion
clips, more give richer motion and detail. ``--token-counts`` draws each sample's
budget from real clips of its own class (scripts/token_counts.py), which matches the
distribution the model was trained on and is what the paper's evaluation uses.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from katok.diffusion import load_generator  # noqa: E402
from katok.tokenizer import KATok  # noqa: E402
from katok.utils import save_frames_grid, write_video  # noqa: E402


def draw_token_counts(data: dict, class_labels: list[int], seed: int, device) -> tuple[torch.Tensor, str]:
    """Draw one token budget per sample from real-clip counts.

    Token count and class are correlated -- complex actions cost more tokens -- and
    the model was trained on that joint distribution. So when the counts JSON knows
    which class each clip came from, every sample draws from clips of *its own*
    class; the global pool is only a fallback for classes the scan never reached.
    """
    pool = data["counts"]
    class_index = data.get("class_index") or []
    rng = torch.Generator().manual_seed(seed)

    by_class: dict[int, list[int]] = {}
    for cls, count in zip(class_index, pool):
        if cls is not None:
            by_class.setdefault(cls, []).append(count)

    def draw(candidates):
        return candidates[int(torch.randint(len(candidates), (1,), generator=rng))]

    if not by_class:
        picks = [draw(pool) for _ in class_labels]
        summary = f"{picks} sampled from {len(pool)} real clips (no class info in JSON)"
    else:
        picks, missing = [], set()
        for label in class_labels:
            candidates = by_class.get(label)
            if not candidates:
                missing.add(label)
                candidates = pool
            picks.append(draw(candidates))
        summary = f"{picks} matched to each sample's class ({len(by_class)} classes in JSON)"
        if missing:
            summary += f"\n           WARNING: no clips for classes {sorted(missing)}; used the global pool"

    return torch.tensor(picks, device=device, dtype=torch.long), summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-g", "--generator", required=True, help="diffusion weights directory")
    ap.add_argument("-t", "--tokenizer", required=True, help="tokenizer weights directory")
    ap.add_argument("-o", "--out", default="out", help="output directory")
    ap.add_argument("--classes", type=int, nargs="+", default=[7], help="class indices to generate")
    ap.add_argument("--tokens", type=int, default=None,
                    help="fixed token budget per sample, registers included")
    ap.add_argument("--token-counts", default=None,
                    help="JSON from scripts/token_counts.py: draw a per-sample budget from real "
                         "clips instead of pinning every sample to one length")
    ap.add_argument("--cfg-scale", type=float, default=None, help="classifier-free guidance scale")
    ap.add_argument("--num-steps", type=int, default=None, help="ODE solver steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fps", type=float, default=8.0, help="fps of the written video")
    ap.add_argument("--grid-every", type=int, default=2, help="frame step in the PNG contact sheet")
    args = ap.parse_args()

    gen = load_generator(args.generator, device=args.device)
    tokenizer = KATok.from_pretrained(args.tokenizer, device=args.device)

    sampling = gen.cfg["sampling"]
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else sampling["cfg_scale"]
    num_steps = args.num_steps or sampling["num_steps"]

    labels = torch.tensor(args.classes, device=args.device, dtype=torch.long)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    if args.token_counts:
        with open(args.token_counts) as f:
            data = json.load(f)
        tokens, summary = draw_token_counts(data, args.classes, args.seed, args.device)
    elif args.tokens:
        tokens = args.tokens
        summary = f"{tokens} ({tokens - gen.n_registers} content + {gen.n_registers} register)"
    else:
        ap.error(
            "choose a token budget: --tokens N for a fixed count, or --token-counts counts.json "
            "to match real clips (see README, 'Choosing the token budget')"
        )

    print(f"variant  : {gen.cfg['variant']}")
    print(f"classes  : {args.classes}")
    print(f"tokens   : {summary}")
    print(f"cfg      : {cfg_scale}   steps: {num_steps}   seed: {args.seed}")

    kwargs = {}
    if gen.cfg["variant"] == "joint":
        for k in ("content_mu", "content_sigma", "pos_mu", "pos_sigma"):
            if k in sampling:
                kwargs[k] = sampling[k]

    latents, valid_counts = gen.sample(
        labels, tokens=tokens, cfg_scale=cfg_scale, num_steps=num_steps,
        generator=generator, **kwargs,
    )
    videos = gen.decode(latents, valid_counts, tokenizer)
    print(f"decoded  : {tuple(videos.shape)}")

    os.makedirs(args.out, exist_ok=True)
    counts = valid_counts.tolist()
    for i, (label, video) in enumerate(zip(args.classes, videos)):
        stem = f"class{label:03d}_tok{counts[i]}_seed{args.seed}_{i}"
        write_video(video, os.path.join(args.out, f"{stem}.mp4"), fps=args.fps)
        save_frames_grid(video, os.path.join(args.out, f"{stem}.png"), every=args.grid_every)
        print(f"wrote    : {os.path.join(args.out, stem)}.mp4 / .png")


if __name__ == "__main__":
    main()
