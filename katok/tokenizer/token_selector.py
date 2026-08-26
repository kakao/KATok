"""Adaptive token selector: the keep-or-drop mechanism of KATok.

The selector receives the encoder embeddings ``e_i`` and produces both the diagonal
Gaussian parameters ``(mu_i, sigma_i) = f_theta(e_i)`` and the keep/drop logits
``alpha_i = g_theta(e_i) in R^2``.

During training the mask is sampled with a Gumbel-Softmax relaxation so the selection
stays differentiable. This module is inference-only: the relaxation is replaced by the
hard mask ``m_i = 1[alpha_i0 >= alpha_i1]`` used at inference time in the paper.
"""

import logging

import torch
import torch.nn as nn

from .common import reparameterize
from .interface import AutoEncoderParams, EncoderOutput, LatentProcessorAbstract, LatentProcessorOutput

logpy = logging.getLogger(__name__)


class TokenImportanceNet(nn.Module):
    """``g_theta``: a linear head predicting per-token keep/drop logits.

    Class 0 is *keep*, class 1 is *drop*.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.ffn_prob = nn.Linear(hidden_dim, 2)

    def forward(self, encoded_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        # The selection head runs in fp32 so that the argmax is not decided by
        # low-precision noise when the rest of the model runs under autocast.
        with torch.autocast(device_type=encoded_tensor.device.type, enabled=False):
            logit = self.ffn_prob(encoded_tensor.float())

        mask = (logit.argmax(dim=-1, keepdim=True) == 0).to(logit.dtype)
        return {"mask": mask, "logit": logit}


class AdaptiveTokenSelector(LatentProcessorAbstract):
    """Turns encoder embeddings into gated continuous latent tokens.

    Args:
        common_params: shared model dimensions.
        token_dropper: the token-importance network ``g_theta``.
        n_to_keep: number of leading tokens always kept. Register tokens sit at the
            front of the sequence and are never dropped, so this equals the number
            of register tokens.
        min_prob: keep-probability floor; mask values below it are clamped to 0.
        latent_sample: if True, sample latents via the reparameterization trick as in
            training. Inference is deterministic and uses ``mu`` directly.
    """

    def __init__(
        self,
        common_params: AutoEncoderParams,
        token_dropper: nn.Module,
        n_to_keep: int = 0,
        min_prob: float = 0.0,
        latent_sample: bool = False,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        self.token_dropper = token_dropper
        self.n_to_keep = n_to_keep
        self.min_prob = min_prob
        self.latent_sample = latent_sample
        self.common_params = common_params

        if hidden_dim is None:
            hidden_dim = common_params.hidden_dim
        self.ffn_mu = nn.Linear(hidden_dim, common_params.latent_token_dim)
        self.ffn_logvar = nn.Linear(hidden_dim, common_params.latent_token_dim)

    def forward(self, encoder_output: EncoderOutput, sample: bool | None = None) -> LatentProcessorOutput:
        encoded_tensor = encoder_output.tokens.tensor
        dropper_output = self.token_dropper(encoded_tensor)

        mu = self.ffn_mu(encoded_tensor)
        log_var = self.ffn_logvar(encoded_tensor)

        if self.latent_sample if sample is None else sample:
            latent_tensor = reparameterize(mu, log_var)
        else:
            latent_tensor = mu

        mask = self.post_process_mask(dropper_output["mask"])
        latent_tensor = latent_tensor * mask

        return LatentProcessorOutput(
            tokens=encoder_output.tokens.replace_tensor(new_tensor=latent_tensor),
            mask=mask,
            mu=mu,
            log_var=log_var,
            logit=dropper_output["logit"],
        )

    def post_process_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if self.min_prob > 0.0:
            mask = mask * (mask > self.min_prob).to(mask.dtype)

        if self.n_to_keep > 0:
            mask = torch.cat([torch.ones_like(mask[:, : self.n_to_keep]), mask[:, self.n_to_keep :]], dim=1)

        return mask
