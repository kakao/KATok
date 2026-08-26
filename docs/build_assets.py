"""Collect the project page's media from ``out/`` into ``docs/static/``.

The page ships with its assets committed, so this only needs re-running when the
underlying clips are regenerated.

    python docs/build_assets.py                 # copy what already exists in out/
    python docs/build_assets.py --fonts         # re-vendor the webfont
    python docs/build_assets.py --generate \\
        -g weights/sky/ -t weights/tokenizer/   # also render the budget demo

The token-budget section shows two 4x4 grids of unconditional SkyTimelapse samples,
one per budget, which no other script produces as a side effect; ``--generate``
renders them and needs a GPU and the released weights.
"""

import argparse
import re
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
STATIC = Path(__file__).resolve().parent / "static"

# Pages published from docs/ cannot reach the repo's assets/ directory, so everything
# the page shows is mirrored into docs/static/. assets/ is the single source of truth;
# regenerate its media via the scripts, then rerun this to refresh the mirror.
COPIES = {
    "../assets/cat_input.mp4": "recon_input.mp4",
    "../assets/cat_recon.mp4": "recon_output.mp4",
    "../assets/cat_tokens.mp4": "recon_tokens.mp4",
    "../assets/all_clips_512_96.mp4": "all_clips_512_96.mp4",
    "../assets/token_allocation.png": "token_allocation.png",
    "../assets/overview.png": "overview.png",
    "../assets/cascaded_gen.png": "cascaded_gen.png",
    "../assets/joint_gen.png": "joint_gen.png",
    "../assets/misalign.png": "misalign.png",
    "../assets/sky_200tok_4x4.mp4": "sky_200tok_4x4.mp4",
    "../assets/sky_400tok_4x4.mp4": "sky_400tok_4x4.mp4",
}

# Token budgets compared in the emergent-control section (sky, unconditional).
BUDGETS = [200, 400]
BUDGET_GRID = 4          # 4x4 samples per budget
BUDGET_CELL, BUDGET_PAD = 256, 4

# The webfont is vendored so the published page makes no third-party requests. DM Sans
# is SIL Open Font License, which permits that; Google Sans, the other candidate, does
# not, which is why it is not the one here.
FONT_NAME = "DM Sans"
FONT_SLUG = "dm-sans"
FONT_CSS_URL = (
    "https://fonts.googleapis.com/css2"
    "?family=DM+Sans:ital,opsz,wght@0,9..40,400..700;1,9..40,400..700&display=swap"
)
# Google serves woff2 only to agents that advertise support, hence the spoofed browser.
FONT_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
# The page is English-only; other scripts would be dead weight in the repo.
FONT_SUBSETS = {"latin", "latin-ext"}


def copy_files() -> list[str]:
    missing = []
    for src, dst in COPIES.items():
        path = OUT / src
        if not path.exists():
            missing.append(src)
            continue
        shutil.copyfile(path, STATIC / dst)
    return missing


def fetch_fonts() -> None:
    """Download the woff2 subsets and print the @font-face block for style.css.

    The rules are printed rather than written: style.css is hand-maintained, and a
    silent rewrite would be easy to miss in review. Paste over the block at the top of
    the file only when a filename or unicode-range actually changed.
    """
    request = urllib.request.Request(FONT_CSS_URL, headers={"User-Agent": FONT_AGENT})
    with urllib.request.urlopen(request) as response:
        css = response.read().decode()

    fonts = STATIC / "fonts"
    fonts.mkdir(parents=True, exist_ok=True)
    rules = []
    for subset, body in re.findall(r"/\* ([\w-]+) \*/\s*@font-face \{(.*?)\}", css, re.S):
        if subset not in FONT_SUBSETS:
            continue
        style = re.search(r"font-style:\s*(\w+)", body).group(1)
        url = re.search(r"url\((\S+?)\)", body).group(1)
        urange = re.search(r"unicode-range:\s*([^;]+);", body).group(1).strip()
        name = f"{FONT_SLUG}-{subset}-{style}.woff2"
        with urllib.request.urlopen(url) as src, open(fonts / name, "wb") as dst:
            shutil.copyfileobj(src, dst)
        print(f"  fonts/{name}  {(fonts / name).stat().st_size / 1024:.0f} KB")
        rules.append(
            "@font-face {\n"
            f"  font-family: '{FONT_NAME}';\n"
            f"  font-style: {style};\n"
            "  font-weight: 400 700;\n"
            "  font-display: swap;\n"
            f"  src: url(static/fonts/{name}) format('woff2');\n"
            f"  unicode-range: {urange};\n"
            "}"
        )
    if not rules:
        raise SystemExit(f"no {sorted(FONT_SUBSETS)} faces in the stylesheet Google returned")
    print("\n--- @font-face rules for docs/style.css ---\n" + "\n".join(rules))


def generate_budget_clips(generator: str, tokenizer: str, seed: int) -> None:
    """Render a 4x4 grid of unconditional sky samples per budget.

    ``generator`` must point at the SkyTimelapse cascade weights (num_classes 1);
    label 0 is the single class. Sampling is seeded per (budget, batch) so reruns
    reproduce the same grids.
    """
    import av
    import numpy as np
    import torch
    from fractions import Fraction

    sys.path.insert(0, str(ROOT))
    from katok.diffusion import load_generator
    from katok.tokenizer import KATok
    from katok.utils import write_video

    tok = KATok.from_pretrained(tokenizer, device="cuda")
    gen = load_generator(generator, device="cuda")
    n = BUDGET_GRID * BUDGET_GRID

    for budget in BUDGETS:
        scratch = OUT / f"page_tok{budget}"
        scratch.mkdir(parents=True, exist_ok=True)
        clips = []
        for b in range(0, n, 8):
            k = min(8, n - b)
            labels = torch.zeros(k, dtype=torch.long, device="cuda")
            g = torch.Generator(device="cuda").manual_seed(seed * 1000 + budget + b)
            latents, vc = gen.sample(labels, tokens=budget, cfg_scale=4.0, num_steps=50, generator=g)
            videos = gen.decode(latents, vc, tok)
            for v in videos:
                path = scratch / f"sky_tok{budget}_{len(clips):02d}.mp4"
                write_video(v, str(path), fps=8)
                clips.append(((v.clamp(-1, 1) + 1) * 127.5).round().byte()
                             .permute(1, 2, 3, 0).cpu().numpy())
            del latents, videos
            torch.cuda.empty_cache()

        nf = clips[0].shape[0]
        w = BUDGET_GRID * BUDGET_CELL + (BUDGET_GRID - 1) * BUDGET_PAD
        grid = np.full((nf, w, w, 3), 255, dtype=np.uint8)
        for i, clip in enumerate(clips):
            r, c = divmod(i, BUDGET_GRID)
            y, x = r * (BUDGET_CELL + BUDGET_PAD), c * (BUDGET_CELL + BUDGET_PAD)
            grid[:, y:y + BUDGET_CELL, x:x + BUDGET_CELL] = clip
        out_path = STATIC / f"sky_{budget}tok_4x4.mp4"
        with av.open(str(out_path), "w") as container:
            stream = container.add_stream("libx264", rate=Fraction(8, 1))
            stream.width, stream.height = w, w
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "20"}
            for frame in grid:
                container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
            container.mux(stream.encode())
        shutil.copyfile(out_path, ROOT / "assets" / out_path.name)
        print(f"wrote {out_path} (+ assets/)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fonts", action="store_true", help="re-download the vendored webfont")
    ap.add_argument("--generate", action="store_true", help="also render the token-budget clips")
    ap.add_argument("-g", "--generator", help="diffusion weights directory (with --generate)")
    ap.add_argument("-t", "--tokenizer", help="tokenizer weights directory (with --generate)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    STATIC.mkdir(parents=True, exist_ok=True)
    missing = copy_files()

    if args.fonts:
        fetch_fonts()

    if args.generate:
        if not (args.generator and args.tokenizer):
            raise SystemExit("--generate needs -g and -t")
        generate_budget_clips(args.generator, args.tokenizer, args.seed)

    for name in sorted(f.name for f in STATIC.iterdir() if f.is_file()):
        print(f"  {name}  {(STATIC / name).stat().st_size / 1024:.0f} KB")
    if missing:
        print("\nnot found under out/ (left as-is in docs/static/):")
        for name in missing:
            print(f"  {name}")


if __name__ == "__main__":
    main()
