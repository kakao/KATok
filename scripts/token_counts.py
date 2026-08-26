"""Measure how many tokens real clips cost, for use as a generation budget.

Evaluating generation quality against a dataset means generating clips of the same
length distribution the tokenizer produces on that dataset -- pinning every sample to
one token count would change the distribution being compared. This script encodes a
folder of videos and writes the per-clip counts.

    python scripts/token_counts.py path/to/videos -w weights/tokenizer/ -o counts.json
    python scripts/generate.py -g weights/cascade/ -t weights/tokenizer/ --token-counts counts.json

The counts include the register tokens, matching what the samplers expect.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from katok.tokenizer import KATok, token_counts  # noqa: E402
from katok.utils import parse_resolution, prepare, read_video  # noqa: E402

VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def find_videos(root: str) -> list[str]:
    if os.path.isfile(root):
        return [root]
    out = []
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.lower().endswith(VIDEO_SUFFIXES):
                out.append(os.path.join(dirpath, name))
    return sorted(out)


def class_index_map(root: str) -> dict[str, int]:
    """Class-name -> index over *all* class directories under ``root``.

    Class-conditional datasets are laid out as ``root/<ClassName>/clip.avi`` with
    indices assigned by alphabetical order of the class directories. The mapping is
    built from every subdirectory -- not only the ones a ``--limit`` run happened to
    reach -- so indices always match the full dataset.
    """
    if not os.path.isdir(root):
        return {}
    dirs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    return {d: i for i, d in enumerate(dirs)}


def class_of(path: str, root: str, mapping: dict[str, int]) -> int | None:
    rel = os.path.relpath(path, root)
    head = rel.split(os.sep)[0]
    return mapping.get(head) if os.sep in rel else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", help="video file or directory to scan recursively")
    ap.add_argument("-w", "--weights", required=True, help="tokenizer weights directory")
    ap.add_argument("-o", "--out", default="counts.json", help="output JSON file")
    ap.add_argument("-c", "--config", default=None, help="tokenizer config YAML")
    ap.add_argument("--resolution", default="256")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="stop after this many clips")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    paths = find_videos(args.videos)
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no videos found under {args.videos}")
    print(f"found {len(paths)} clips")

    model = KATok.from_pretrained(args.weights, config=args.config, device=args.device)
    resolution = parse_resolution(args.resolution)

    counts, kept, skipped = [], [], []
    batch, batch_paths = [], []

    def flush():
        if not batch:
            return
        x = torch.cat(batch, dim=0).to(args.device)
        latents = model.encode(x)
        # Registers are always kept, so add them back to get the sampler's budget.
        n = token_counts(latents).long() + latents.tokens.n_registers
        counts.extend(n.tolist())
        kept.extend(batch_paths)
        batch.clear()
        batch_paths.clear()

    for i, path in enumerate(paths):
        try:
            video = read_video(path, n_frames=args.frames * args.stride, stride=args.stride)
            batch.append(prepare(video, resolution=resolution, n_frames=args.frames))
            batch_paths.append(path)
        except Exception as exc:  # noqa: BLE001 - a short clip should not stop the scan
            skipped.append((path, str(exc)))
            continue

        if len(batch) == args.batch_size:
            flush()
            print(f"  {len(counts)}/{len(paths)} encoded", end="\r", flush=True)
    flush()

    if not counts:
        raise SystemExit("no clips could be encoded")

    t = torch.tensor(counts, dtype=torch.float)
    print(f"\nencoded {len(counts)} clips" + (f", skipped {len(skipped)}" if skipped else ""))
    print(f"  tokens: mean {t.mean():.1f}  std {t.std():.1f}  min {int(t.min())}  max {int(t.max())}")
    print(f"  quartiles: {[int(q) for q in t.quantile(torch.tensor([0.25, 0.5, 0.75])).tolist()]}")

    # Class-conditional layouts (root/<ClassName>/clip.avi) keep the class of every
    # clip, so generate.py can draw each sample's budget from clips of its own class.
    mapping = class_index_map(args.videos)
    class_indices = [class_of(p, args.videos, mapping) for p in kept]
    n_classed = sum(c is not None for c in class_indices)
    if n_classed:
        covered = len({c for c in class_indices if c is not None})
        print(f"  classes: {covered} of {len(mapping)} covered ({n_classed} clips attributed)")

    with open(args.out, "w") as f:
        json.dump(
            {
                "counts": counts,
                "includes_registers": True,
                "resolution": args.resolution,
                "frames": args.frames,
                "files": [os.path.relpath(p, args.videos) for p in kept],
                "classes": sorted(mapping, key=mapping.get),
                "class_index": class_indices,
            },
            f,
        )
    print(f"wrote {args.out}")
    if skipped:
        print(f"skipped {len(skipped)}, e.g. {skipped[0][0]}: {skipped[0][1]}")


if __name__ == "__main__":
    main()
