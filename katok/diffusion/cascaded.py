"""Cascaded generation: mask prior first, then content.

This is the paper's default strategy. A lightweight mask prior generates an
occupancy mask over the token grid; the selected cells become explicit positional
conditioning for the main content model. Decoupling *where* tokens live from *what*
they contain is what removes the content-position misalignment that appears when a
diffusion model is trained naively on sparse latents.
"""

import torch

from .flow_matching import sample_ode
from .generator import Generator
from .mask_prior import mask_to_positions, prepend_register_positions


class CascadedGenerator(Generator):
    """Two-stage sampler: mask prior -> positions -> content."""

    @torch.no_grad()
    def sample_mask(self, class_labels: torch.Tensor, valid_counts: torch.Tensor,
                    num_steps: int = 50, generator: torch.Generator | None = None) -> torch.Tensor:
        """Run the mask prior and return the raw grid mask, ``(B, G)``.

        The prior is conditioned on the class label and the requested token count,
        but never on content, and is sampled without classifier-free guidance.
        """
        device = class_labels.device
        grid = self.model.prior_grid
        n_cells = grid[0] * grid[1] * grid[2]

        noise = torch.randn(len(class_labels), n_cells, 1, device=device, generator=generator)
        soft_mask = sample_ode(
            noise, self.model.forward_prior, num_steps=num_steps,
            y=class_labels, token_len=valid_counts.float(),
        )[-1]
        return soft_mask.squeeze(-1)

    @torch.no_grad()
    def sample(self, class_labels: torch.Tensor, tokens, cfg_scale: float = 4.0,
               num_steps: int = 50, seq_len: int | None = None,
               generator: torch.Generator | None = None, return_mask: bool = False):
        """Generate latent tokens for the given class labels.

        Args:
            class_labels: ``(B,)`` class indices.
            tokens: how many tokens to generate, **including** the leading register
                tokens. An int applies one length to every sample -- raising it
                yields more motion and finer detail, lowering it yields simpler,
                more static clips. A ``(B,)`` tensor gives per-sample lengths, which
                is what FVD evaluation uses so the generated length distribution
                matches the data.
            cfg_scale: classifier-free guidance scale. ``1.0`` disables guidance.
            num_steps: ODE solver output steps.
            seq_len: padded sequence length; defaults to the largest token count.
            generator: optional RNG for reproducible noise.
            return_mask: also return the mask prior's raw grid mask.

        Returns:
            ``(latents, valid_counts)``, or ``(latents, valid_counts, mask)``.
            ``latents`` are normalized; pass them to :meth:`Generator.decode`.
        """
        device = class_labels.device
        B = len(class_labels)
        content_dim = self.model.in_channels

        valid_counts, seq_len = self._resolve_counts(tokens, B, device, seq_len)
        content_counts = valid_counts - self.n_registers

        soft_mask = self.sample_mask(class_labels, valid_counts, num_steps=num_steps, generator=generator)
        positions = mask_to_positions(soft_mask, self.model.prior_grid_coords, content_counts)
        positions = prepend_register_positions(positions, self.n_registers)
        if positions.shape[1] < seq_len:
            pad = positions.new_zeros(B, seq_len - positions.shape[1], 3)
            positions = torch.cat([positions, pad], dim=1)

        noise = torch.randn(B, seq_len, content_dim, device=device, generator=generator)

        use_cfg = cfg_scale != 1.0
        if use_cfg:
            latents = sample_ode(
                torch.cat([noise, noise], dim=0), self.model.forward_with_cfg, num_steps=num_steps,
                y=self._labels(class_labels, True), cfg_scale=cfg_scale,
                valid_counts=valid_counts, token_len=valid_counts, gt_positions=positions,
            )[-1].chunk(2, dim=0)[0]
        else:
            latents = sample_ode(
                noise, self.model.forward, num_steps=num_steps,
                y=class_labels, valid_counts=valid_counts, token_len=valid_counts,
                gt_positions=positions,
            )[-1]

        if return_mask:
            return latents, valid_counts, soft_mask
        return latents, valid_counts
