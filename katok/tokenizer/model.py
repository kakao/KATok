"""KATok: Keep-or-Drop? Adaptive Tokenizer for Compact Video Representation."""

import logging
import os
from typing import Any

import torch
import torch.nn as nn

from .common import get_patch_ids
from .decoder import RopeDecoder
from .encoder import RopeEncoder
from .interface import (
    AutoEncoderOutput,
    AutoEncoderParams,
    DecoderAbstract,
    EncoderAbstract,
    LatentProcessorAbstract,
    LatentProcessorOutput,
    PatchifierAbstract,
    UnpatchifierAbstract,
)
from .patchify import LinearPatchifier, LinearUnpatchifier
from .token_selector import AdaptiveTokenSelector, TokenImportanceNet

logpy = logging.getLogger(__name__)

# Buffers that only exist to drive the training-time Gumbel-Softmax schedule.
# They carry no information needed at inference.
_TRAINING_ONLY_KEYS = {
    "latent_processor.token_dropper.gumbel_softmax.tau",
    "latent_processor.token_dropper.gumbel_softmax.current_decay_step",
}


class KATok(nn.Module):
    """Transformer VAE with an adaptive keep-or-drop token selector.

    The pipeline is ``patchify -> encode -> select -> decode -> unpatchify``. The
    selector predicts a keep/drop decision per latent token, so the number of tokens
    an input costs is decided by its content rather than by a fixed budget.
    """

    def __init__(
        self,
        patchifier: PatchifierAbstract,
        encoder: EncoderAbstract,
        latent_processor: LatentProcessorAbstract,
        decoder: DecoderAbstract,
        unpatchifier: UnpatchifierAbstract,
    ):
        super().__init__()
        self.patchifier = patchifier
        self.encoder = encoder
        self.latent_processor = latent_processor
        self.decoder = decoder
        self.unpatchifier = unpatchifier

    def forward(self, batch: dict) -> AutoEncoderOutput:
        # batch["vid"] assumed to have range [-1, 1]
        patchifier_output = self.patchifier(batch)
        encoder_output = self.encoder(patchifier_output)
        latent_processor_output = self.latent_processor(encoder_output)
        decoder_output = self.decoder(latent_processor_output)
        unpatchifier_output = self.unpatchifier(decoder_output)
        return AutoEncoderOutput(
            patchifier_output=patchifier_output,
            encoder_output=encoder_output,
            latent_processor_output=latent_processor_output,
            decoder_output=decoder_output,
            unpatchifier_output=unpatchifier_output,
        )

    # ---- construction ----

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "KATok":
        """Build an (untrained) model from a parsed ``configs/*.yaml`` dict."""
        common = dict(cfg["common"])
        rope_base = common.get("rope_base_spatial_resolution")
        common["rope_base_spatial_resolution"] = tuple(rope_base) if rope_base is not None else None
        params = AutoEncoderParams(**common)

        patch, unpatch = cfg["patchifier"], cfg["unpatchifier"]
        sel = cfg["token_selector"]
        return cls(
            patchifier=LinearPatchifier(
                patch["input_dim"], params.hidden_dim, tuple(patch["scale_factors"])
            ),
            encoder=RopeEncoder(params, **cfg["encoder"]),
            latent_processor=AdaptiveTokenSelector(
                params, TokenImportanceNet(params.hidden_dim), **sel
            ),
            decoder=RopeDecoder(params, **cfg["decoder"]),
            unpatchifier=LinearUnpatchifier(
                unpatch["output_dim"], params.hidden_dim, tuple(unpatch["scale_factors"])
            ),
        )

    @classmethod
    def from_pretrained(
        cls,
        weights: str,
        config: str | dict[str, Any] | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> "KATok":
        """Load a released checkpoint.

        Args:
            weights: path to a ``.safetensors`` file, or to a directory holding
                ``model.safetensors`` and ``config.yaml``.
            config: path to a config YAML, or an already-parsed dict. Defaults to a
                ``config.yaml`` next to the weights, else ``configs/tokenizer.yaml``.
            device: device to move the model to.
            dtype: cast the model to this dtype. The released weights are fp32; note
                that reconstructions are not bit-identical in lower precision.
        """
        import yaml
        from safetensors.torch import load_file

        if os.path.isdir(weights):
            weights_dir = weights
            weights = os.path.join(weights_dir, "model.safetensors")
        else:
            weights_dir = os.path.dirname(os.path.abspath(weights))

        if config is None:
            sidecar = os.path.join(weights_dir, "config.yaml")
            config = sidecar if os.path.exists(sidecar) else _default_config_path()
        if isinstance(config, str):
            with open(config) as f:
                config = yaml.safe_load(f)

        model = cls.from_config(config)
        state_dict = load_file(weights)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        unexpected = [k for k in unexpected if k not in _TRAINING_ONLY_KEYS]
        if missing or unexpected:
            raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")

        model.eval()
        if dtype is not None:
            model = model.to(dtype=dtype)
        return model.to(device)

    # ---- inference API ----

    @torch.no_grad()
    def encode(self, vid: torch.Tensor, sample: bool = False) -> LatentProcessorOutput:
        """Encode a video into sparse latent tokens.

        Args:
            vid: ``(B, C, T, H, W)`` in ``[-1, 1]``.
            sample: draw latents with the reparameterization trick instead of using
                the posterior mean. Deterministic (``False``) by default.

        Returns a :class:`LatentProcessorOutput` whose ``tokens.tensor`` is
        ``(B, L, latent_dim)`` with dropped rows zeroed, and whose ``mask`` is
        ``(B, L, 1)``. Register tokens occupy the first ``tokens.n_registers`` rows.
        """
        patchifier_output = self.patchifier({"vid": vid})
        encoder_output = self.encoder(patchifier_output)
        return self.latent_processor(encoder_output, sample=sample)

    @torch.no_grad()
    def decode(self, latent_processor_output: LatentProcessorOutput) -> torch.Tensor:
        """Decode sparse latent tokens back to a video in ``[-1, 1]``."""
        decoder_output = self.decoder(latent_processor_output)
        return self.unpatchifier(decoder_output).recon

    @torch.no_grad()
    def reconstruct(self, vid: torch.Tensor, sample: bool = False) -> tuple[torch.Tensor, LatentProcessorOutput]:
        """Encode then decode. Returns ``(recon, latent_processor_output)``."""
        latent_processor_output = self.encode(vid, sample=sample)
        return self.decode(latent_processor_output), latent_processor_output


def _default_config_path() -> str:
    """``configs/tokenizer.yaml`` at the repo root."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "..", "configs", "tokenizer.yaml")


def token_counts(latent_processor_output: LatentProcessorOutput, include_registers: bool = False) -> torch.Tensor:
    """Number of active tokens per sample, ``N_eff(X)``.

    Register tokens are always kept and are excluded by default so the count reflects
    the content tokens actually spent on the video.
    """
    mask = latent_processor_output.mask
    if not include_registers:
        mask = mask[:, latent_processor_output.tokens.n_registers :]
    return mask.squeeze(-1).sum(dim=1)


def token_positions(latent_processor_output: LatentProcessorOutput) -> list[torch.Tensor]:
    """Grid coordinates ``(t, h, w)`` of the kept content tokens, per sample.

    Returns one ``(n_kept_i, 3)`` tensor per batch element. Useful for visualizing
    which patches survived and for conditioning downstream models on token positions.
    """
    tokens = latent_processor_output.tokens
    n_reg = tokens.n_registers
    mask = latent_processor_output.mask[:, n_reg:, 0]  # (B, L)

    ids = get_patch_ids(tokens.original_shape, device=mask.device)  # (L, 3)
    return [ids[m.bool()] for m in mask]
