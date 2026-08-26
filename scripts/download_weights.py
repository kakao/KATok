"""Download the released KATok weights from the Hugging Face Hub.

    python scripts/download_weights.py -o weights/

Fetches ``model.safetensors`` and ``config.yaml`` into the output directory, which
can then be passed to ``KATok.from_pretrained``.
"""

import argparse
import os
import shutil

# The weights will be published on the Hugging Face Hub once internal review
# completes; this stays empty until then. TODO: fill in the released repo id.
HF_REPO_ID = os.environ.get("KATOK_HF_REPO", "")

FILES = ["model.safetensors", "config.yaml"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default="weights", help="output directory")
    ap.add_argument("-r", "--repo-id", default=HF_REPO_ID, help="Hugging Face repo id")
    ap.add_argument("--revision", default=None, help="branch, tag or commit to fetch")
    args = ap.parse_args()

    if not args.repo_id:
        raise SystemExit(
            "No Hugging Face repo id configured -- the weights release is pending\n"
            "internal review. Pass --repo-id or set KATOK_HF_REPO once published."
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("huggingface_hub is required: pip install huggingface_hub")

    os.makedirs(args.out, exist_ok=True)
    for filename in FILES:
        print(f"downloading {filename} from {args.repo_id} ...")
        cached = hf_hub_download(repo_id=args.repo_id, filename=filename, revision=args.revision)
        shutil.copyfile(cached, os.path.join(args.out, filename))

    print(f"weights ready in {args.out}")


if __name__ == "__main__":
    main()
