"""Dataclasses and abstract base classes shared by the tokenizer components."""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

Shape4D = tuple[int, int, int, int]

# ==== data classes ==== #


@dataclass(frozen=True, slots=True)
class TokenData:
    """A flattened token sequence plus the grid it was derived from.

    ``original_shape`` is the patch grid ``(C, T, H, W)`` the tokens were laid out on,
    and ``original_data_shape`` is the input video shape ``(C, T, H, W)`` in pixels.
    """

    tensor: torch.Tensor
    original_shape: Shape4D
    original_data_shape: Shape4D
    n_registers: int = 0

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @property
    def bs(self) -> int:
        return self.tensor.shape[0]

    @property
    def shape(self) -> torch.Size:
        return self.tensor.shape

    @property
    def dtype(self) -> torch.dtype:
        return self.tensor.dtype

    def replace_tensor(self, new_tensor: torch.Tensor, n_registers: int | None = None) -> "TokenData":
        if n_registers is None:
            n_registers = self.n_registers
        return TokenData(
            tensor=new_tensor,
            original_shape=self.original_shape,
            original_data_shape=self.original_data_shape,
            n_registers=n_registers,
        )


@dataclass(frozen=True, slots=True)
class PatchifierOutput:
    tokens: TokenData


@dataclass(frozen=True, slots=True)
class EncoderOutput:
    tokens: TokenData


@dataclass(frozen=True, slots=True)
class LatentProcessorOutput:
    """Output of the adaptive token selector.

    ``tokens`` holds the gated latents (already multiplied by ``mask``), ``mask`` is
    the per-token keep mask of shape ``(B, L, 1)``, and ``logit`` holds the raw
    keep/drop logits ``alpha_i`` of shape ``(B, L, 2)``.
    """

    tokens: TokenData
    mask: torch.Tensor
    mu: torch.Tensor
    log_var: torch.Tensor
    logit: torch.Tensor


@dataclass(frozen=True, slots=True)
class DecoderOutput:
    tokens: TokenData


@dataclass(frozen=True, slots=True)
class UnpatchifierOutput:
    recon: torch.Tensor


@dataclass(frozen=True, slots=True)
class AutoEncoderOutput:
    patchifier_output: PatchifierOutput
    encoder_output: EncoderOutput
    latent_processor_output: LatentProcessorOutput
    decoder_output: DecoderOutput
    unpatchifier_output: UnpatchifierOutput


@dataclass
class AutoEncoderParams:
    """Parameters shared by the encoder, latent processor and decoder."""

    hidden_dim: int
    num_heads: int
    mlp_ratio: float
    qk_scale: float | None
    dropout: float
    latent_token_dim: int
    # When set, RoPE spatial ids are rescaled to this base grid so that a model
    # trained at one resolution extrapolates to another.
    rope_base_spatial_resolution: tuple[int, int] | None = None


# ==== abstract classes ==== #


class PatchifierAbstract(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, batch: dict) -> PatchifierOutput:
        pass


class EncoderAbstract(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, patchifier_output: PatchifierOutput) -> EncoderOutput:
        pass

    @abstractmethod
    def prepare_pos_emb(self, patches: TokenData, **kwargs: Any) -> torch.Tensor:
        pass

    @abstractmethod
    def prepare_register_tokens(self, patches: TokenData, pe: torch.Tensor) -> tuple[TokenData, torch.Tensor]:
        pass


class LatentProcessorAbstract(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, encoder_output: EncoderOutput) -> LatentProcessorOutput:
        pass


class DecoderAbstract(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, latent_processor_output: LatentProcessorOutput) -> DecoderOutput:
        pass

    @abstractmethod
    def prepare_pos_emb(self, tokens: TokenData, latent_mask: torch.Tensor, patch_shape: Shape4D) -> torch.Tensor:
        pass


class UnpatchifierAbstract(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, decoder_output: DecoderOutput) -> UnpatchifierOutput:
        pass
