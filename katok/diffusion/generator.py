"""Shared plumbing for the generation pipelines.

Holds the pieces both the cascaded and joint variants need: latent normalization,
checkpoint loading, and turning generated latents back into pixels through the
tokenizer.
"""

import os
from typing import Any

import torch
import torch.nn as nn

from .sit import SiT


class LatentStats:
    """Channel statistics used to whiten tokenizer latents for the diffusion model.

    Training standardizes latents as ``(z - mean) / std`` before flow matching, so
    generated latents come back in that whitened space and have to be undone before
    the tokenizer can decode them.

    The statistics are **per channel**, not per position. There is one set for content
    tokens and a separate set for each register token, because registers sit on a very
    different scale (std ~0.2 against ~1.4). The content set is broadcast across
    however many content tokens a sequence happens to have.
    """

    def __init__(self, register_mean: torch.Tensor, register_std: torch.Tensor,
                 content_mean: torch.Tensor, content_std: torch.Tensor):
        self.register_mean = register_mean
        self.register_std = register_std
        self.content_mean = content_mean
        self.content_std = content_std

    @property
    def n_registers(self) -> int:
        return self.register_mean.shape[0]

    @classmethod
    def load(cls, path: str) -> "LatentStats":
        from safetensors.torch import load_file

        d = load_file(path)
        return cls(d["register_mean"], d["register_std"], d["content_mean"], d["content_std"])

    def to(self, device, dtype=None) -> "LatentStats":
        return LatentStats(*(t.to(device, dtype) for t in
                             (self.register_mean, self.register_std, self.content_mean, self.content_std)))

    def _expand(self, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Build ``(length, channels)`` stats: register rows first, then content."""
        n_reg = min(self.n_registers, length)
        n_content = length - n_reg
        mean = torch.cat([self.register_mean[:n_reg], self.content_mean.expand(n_content, -1)])
        std = torch.cat([self.register_std[:n_reg], self.content_std.expand(n_content, -1)])
        return mean, std

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        mean, std = self._expand(z.shape[-2])
        return (z - mean) / std

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        mean, std = self._expand(z.shape[-2])
        return z * std + mean


def build_sit(cfg: dict[str, Any]) -> SiT:
    """Build the backbone described by a ``configs/diffusion_*.yaml`` dict."""
    model_cfg = dict(cfg["model"])
    prior_grid = model_cfg.pop("prior_grid", None)
    if prior_grid is not None:
        model_cfg["prior_grid"] = tuple(prior_grid)
    return SiT(**model_cfg)


class Generator(nn.Module):
    """Base class for the generation pipelines.

    Subclasses implement :meth:`sample`, which returns latent tokens plus the number
    of active tokens per sample; :meth:`decode` hands those to a KATok tokenizer.
    """

    def __init__(self, model: SiT, stats: LatentStats, cfg: dict[str, Any]):
        super().__init__()
        self.model = model
        self.stats = stats
        self.cfg = cfg
        self.n_registers = cfg["sampling"]["n_registers"]
        self.num_classes = cfg["model"]["num_classes"]

    # ---- construction ----

    @classmethod
    def from_pretrained(cls, weights: str, config: str | dict[str, Any] | None = None,
                        stats: str | None = None, device: str | torch.device = "cpu") -> "Generator":
        """Load a released generation checkpoint.

        Args:
            weights: ``.safetensors`` file, or a directory holding ``model.safetensors``,
                ``config.yaml`` and ``latent_stats.safetensors``.
            config: config path or parsed dict; defaults to a ``config.yaml`` beside
                the weights.
            stats: latent-statistics file; defaults to ``latent_stats.safetensors``
                beside the weights.
            device: device to move the model to.
        """
        import yaml
        from safetensors.torch import load_file

        if os.path.isdir(weights):
            weights_dir = weights
            weights = os.path.join(weights_dir, "model.safetensors")
        else:
            weights_dir = os.path.dirname(os.path.abspath(weights))

        if config is None:
            config = os.path.join(weights_dir, "config.yaml")
        if isinstance(config, str):
            with open(config) as f:
                config = yaml.safe_load(f)

        if stats is None:
            stats = os.path.join(weights_dir, "latent_stats.safetensors")

        model = build_sit(config)
        model.load_state_dict(load_file(weights), strict=True)

        model.eval()
        gen = cls(model, LatentStats.load(stats), config)
        return gen.to(device)

    def to(self, *args, **kwargs):
        out = super().to(*args, **kwargs)
        device = self.model.x_embedder.weight.device
        out.stats = out.stats.to(device)
        return out

    # ---- sampling helpers ----

    def _labels(self, class_labels: torch.Tensor, use_cfg: bool) -> torch.Tensor:
        """Append the unconditional label half used by classifier-free guidance."""
        if not use_cfg:
            return class_labels
        null = torch.full_like(class_labels, self.num_classes)
        return torch.cat([class_labels, null], dim=0)

    def _resolve_counts(self, tokens, batch: int, device, seq_len: int | None = None):
        """Turn a token budget into ``(valid_counts, seq_len)``.

        ``tokens`` is either an int -- every sample gets the same length, which is
        how the paper's token-count sweeps are run -- or a per-sample tensor of
        counts. Per-sample counts are what the FVD evaluation uses: the counts are
        taken from encoding real clips, so the generated length distribution matches
        the data instead of being pinned to one value.

        Slots past a sample's count are padding: excluded from attention and dropped
        before decoding, so ``seq_len`` only has to be at least the largest count.
        """
        if isinstance(tokens, int):
            counts = torch.full((batch,), tokens, device=device, dtype=torch.long)
        else:
            counts = torch.as_tensor(tokens, device=device, dtype=torch.long).reshape(-1)
            if len(counts) != batch:
                raise ValueError(f"got {len(counts)} token counts for {batch} labels")

        longest = int(counts.max().item())
        if counts.min().item() <= self.n_registers:
            raise ValueError(f"token counts must exceed the {self.n_registers} register tokens")

        if seq_len is None:
            seq_len = longest
        elif seq_len < longest:
            raise ValueError(f"seq_len {seq_len} is shorter than the largest token count {longest}")
        return counts, seq_len

    def sample(self, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, valid_counts: torch.Tensor, tokenizer) -> torch.Tensor:
        """Denormalize generated latents and decode them to a video in ``[-1, 1]``.

        Args:
            latents: ``(B, L, C)`` normalized latents from :meth:`sample`.
            valid_counts: ``(B,)`` active tokens per sample, registers included.
            tokenizer: a :class:`katok.tokenizer.KATok` instance.
        """
        from ..tokenizer.interface import LatentProcessorOutput, TokenData

        latents = self.stats.denormalize(latents)

        B, L, _ = latents.shape
        mask = (torch.arange(L, device=latents.device).unsqueeze(0) < valid_counts.unsqueeze(1)).to(latents.dtype)
        mask = mask.unsqueeze(-1)

        grid = self.cfg["sampling"]["patch_grid"]
        tokens = TokenData(
            tensor=latents * mask,
            original_shape=(0, *grid),
            original_data_shape=(0, 0, 0, 0),
            n_registers=self.n_registers,
        )
        lpo = LatentProcessorOutput(
            tokens=tokens, mask=mask, mu=latents, log_var=torch.zeros_like(latents), logit=None
        )
        return tokenizer.decode(lpo)
