"""Convert a training checkpoint into the safetensors weights this repo loads.

Handles both kinds of checkpoint and picks the right one automatically:

* **Tokenizer** -- a PyTorch Lightning checkpoint with the raw weights under
  ``model.*`` and the EMA shadow copy under ``model_ema.*``.
* **Diffusion** -- a SiT checkpoint with ``model`` / ``ema`` state dicts.

Both export the EMA weights by default, which is what the paper evaluates.

    python scripts/convert_checkpoint.py path/to/tokenizer.ckpt -o weights/tokenizer/
    python scripts/convert_checkpoint.py path/to/0100000.pt -o weights/cascade/ --config configs/diffusion_cascade.yaml

The output directory can be passed straight to ``KATok.from_pretrained`` or
``load_generator``.
"""

import argparse
import os
import shutil

import torch
from safetensors.torch import save_file

CONFIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs")

# LitEma keeps shadow parameters as buffers, and buffer names cannot contain dots,
# so it stores them under ``name.replace(".", "")``. These two are its own state.
_EMA_INTERNAL = {"decay", "num_updates"}

# Training-schedule buffers that the inference tokenizer does not define.
_TRAINING_ONLY = {
    "latent_processor.token_dropper.gumbel_softmax.tau",
    "latent_processor.token_dropper.gumbel_softmax.current_decay_step",
}


def detect_kind(ckpt: dict) -> str:
    if "state_dict" in ckpt:
        return "tokenizer"
    if "ema" in ckpt or "model" in ckpt:
        return "diffusion"
    raise RuntimeError("unrecognized checkpoint: expected 'state_dict' or 'ema'/'model'")


def extract_tokenizer(ckpt: dict, use_ema: bool) -> dict[str, torch.Tensor]:
    state_dict = ckpt["state_dict"]
    model_sd = {k[len("model.") :]: v for k, v in state_dict.items() if k.startswith("model.")}
    if not model_sd:
        raise RuntimeError("no 'model.*' keys found")
    print(f"  model keys  : {len(model_sd)}")

    if use_ema:
        ema_sd = {k[len("model_ema.") :]: v for k, v in state_dict.items() if k.startswith("model_ema.")}
        if not ema_sd:
            raise RuntimeError("--ema requested but no 'model_ema.*' keys present")

        flat_to_name = {name.replace(".", ""): name for name in model_sd}
        unmatched = [k for k in ema_sd if k not in _EMA_INTERNAL and k not in flat_to_name]
        if unmatched:
            raise RuntimeError(f"EMA keys with no matching parameter: {unmatched[:5]}")

        # EMA tracks parameters only; anything else stays at its raw value.
        mapped = {flat_to_name[k]: v for k, v in ema_sd.items() if k not in _EMA_INTERNAL}
        model_sd = {**model_sd, **mapped}
        print(f"  ema applied : {len(mapped)} / {len(model_sd)} tensors")

    dropped = sorted(set(model_sd) & _TRAINING_ONLY)
    for k in dropped:
        model_sd.pop(k)
    if dropped:
        print(f"  dropped     : {dropped}")
    return model_sd


def extract_diffusion(ckpt: dict, use_ema: bool) -> dict[str, torch.Tensor]:
    key = "ema" if use_ema else "model"
    if key not in ckpt:
        raise RuntimeError(f"checkpoint has no '{key}' state dict (found: {sorted(ckpt)})")
    sd = ckpt[key]
    print(f"  using       : {key}  ({len(sd)} tensors)")
    return dict(sd)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", help="training checkpoint (.ckpt or .pt)")
    ap.add_argument("-o", "--out", default="weights", help="output directory")
    ap.add_argument("--raw", action="store_true", help="export raw weights instead of EMA")
    ap.add_argument("--config", default=None, help="config YAML to copy alongside the weights")
    ap.add_argument("--kind", default=None, choices=["tokenizer", "diffusion"], help="override auto-detection")
    args = ap.parse_args()

    print(f"reading {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    kind = args.kind or detect_kind(ckpt)
    print(f"  kind        : {kind}")
    if "global_step" in ckpt:
        print(f"  global_step : {ckpt['global_step']}")

    use_ema = not args.raw
    if kind == "tokenizer":
        state_dict = extract_tokenizer(ckpt, use_ema)
        default_config = os.path.join(CONFIGS, "tokenizer.yaml")
    else:
        state_dict = extract_diffusion(ckpt, use_ema)
        default_config = None

    config = args.config or default_config
    if config is None:
        raise SystemExit(
            "diffusion checkpoints need --config, e.g. --config configs/diffusion_cascade.yaml"
        )

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "model.safetensors")
    save_file({k: v.contiguous() for k, v in state_dict.items()}, out_path)

    config_dst = os.path.join(args.out, "config.yaml")
    if not (os.path.exists(config_dst) and os.path.samefile(config, config_dst)):
        shutil.copyfile(config, config_dst)

    total = sum(v.numel() for v in state_dict.values())
    print(f"wrote {out_path}  ({len(state_dict)} tensors, {total / 1e6:.2f}M parameters)")
    print(f"      {os.path.join(args.out, 'config.yaml')}")

    if kind == "diffusion":
        # Generation needs the tokenizer's latent statistics to denormalize.
        stats_src = os.path.join(CONFIGS, "latent_stats.safetensors")
        shutil.copyfile(stats_src, os.path.join(args.out, "latent_stats.safetensors"))
        print(f"      {os.path.join(args.out, 'latent_stats.safetensors')}")


if __name__ == "__main__":
    main()
