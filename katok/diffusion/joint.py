"""Joint content-position generation with timestep decoupling.

The alternative to the cascade: instead of a separate prior, the position of each
token is carried as three extra channels alongside its 64 content channels and
denoised in the same pass. Content and position follow independent noise schedules,
implemented as a logit-normal shift of the shared solver time, so the spatial layout
can settle before fine content detail does.

This variant reaches slightly better FVD on SkyTimelapse but is sensitive to the
choice of the two schedules; the paper adopts the cascade as the default.
"""

import torch

from .flow_matching import decoupled_ode_sample
from .generator import Generator


class JointGenerator(Generator):
    """Single-pass sampler over concatenated content and position channels."""

    @torch.no_grad()
    def sample(self, class_labels: torch.Tensor, tokens, cfg_scale: float = 4.0,
               num_steps: int = 50, content_mu: float = 0.0, content_sigma: float = 1.0,
               pos_mu: float = 2.0, pos_sigma: float = 0.3, seq_len: int | None = None,
               generator: torch.Generator | None = None, return_positions: bool = False):
        """Generate latent tokens and their positions together.

        Args:
            class_labels: ``(B,)`` class indices.
            tokens: how many tokens to generate, including register tokens. An int
                applies one length to every sample; a ``(B,)`` tensor gives
                per-sample lengths, as used for FVD evaluation.
            cfg_scale: classifier-free guidance scale.
            num_steps: ODE solver output steps.
            content_mu / content_sigma: logit-normal shift for the content channels.
                The default ``(0.0, 1.0)`` is the identity schedule.
            pos_mu / pos_sigma: shift for the position channels. The solver runs
                ``t = 0`` (noise) to ``t = 1`` (data), so a positive ``pos_mu``
                advances positions ahead of content, resolving layout first.
            seq_len: padded sequence length; defaults to the largest token count.
            generator: optional RNG for reproducible noise.
            return_positions: also return the generated ``(t, h, w)`` coordinates.

        Returns:
            ``(latents, valid_counts)``, or ``(latents, valid_counts, positions)``.
        """
        device = class_labels.device
        B = len(class_labels)

        total_dim = self.model.in_channels
        content_dim = total_dim - 3
        if content_dim <= 0:
            raise ValueError("joint generation needs 3 position channels on top of the content channels")

        valid_counts, seq_len = self._resolve_counts(tokens, B, device, seq_len)
        noise = torch.randn(B, seq_len, total_dim, device=device, generator=generator)

        use_cfg = cfg_scale != 1.0
        state = torch.cat([noise, noise], dim=0) if use_cfg else noise
        model_fn = self.model.forward_with_cfg if use_cfg else self.model.forward

        kwargs = dict(y=self._labels(class_labels, use_cfg), valid_counts=valid_counts, token_len=valid_counts)
        if use_cfg:
            kwargs["cfg_scale"] = cfg_scale

        out = decoupled_ode_sample(
            state, model_fn, content_dim=content_dim, num_steps=num_steps,
            content_mu=content_mu, content_sigma=content_sigma,
            pos_mu=pos_mu, pos_sigma=pos_sigma, **kwargs,
        )[-1]
        if use_cfg:
            out = out.chunk(2, dim=0)[0]

        latents, positions = out[..., :content_dim], out[..., content_dim:]
        if return_positions:
            # The coordinate channels were whitened with per-axis stats during
            # training; undo that before interpreting them as (t, h, w) in [-1, 1].
            sampling = self.cfg["sampling"]
            pos_mean = positions.new_tensor(sampling.get("pos_channel_mean", [0.0, 0.0, 0.0]))
            pos_std = positions.new_tensor(sampling.get("pos_channel_std", [1.0, 1.0, 1.0]))
            positions = (positions * pos_std + pos_mean).clamp(-1, 1)
            return latents, valid_counts, positions
        return latents, valid_counts
