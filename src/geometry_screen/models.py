"""One support-aware backbone with two end-to-end predictive families."""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from irts_model import ConditionalDiffusion, LatentGridDenoiser, interpolate_latent_grid


FAMILIES = frozenset({"conditional_diffusion", "direct_gaussian"})


class SupportAwareDenoiser(LatentGridDenoiser):
    """The common backbone for both time encodings and both families."""

    @staticmethod
    def context_support(times: Tensor, valid: Tensor, grid_size: int) -> Tensor:
        if times.shape != valid.shape or times.ndim != 2 or grid_size < 2:
            raise ValueError("times/valid must share [B,L] and grid_size must exceed one")
        valid = valid.bool()
        if not valid.any(dim=1).all():
            raise ValueError("every sample needs observed context")
        lower = times.masked_fill(~valid, float("inf")).min(dim=1).values
        upper = times.masked_fill(~valid, float("-inf")).max(dim=1).values
        grid = torch.linspace(0.0, 1.0, grid_size, device=times.device, dtype=times.dtype)
        support = (grid[None] >= lower[:, None]) & (grid[None] <= upper[:, None])
        # A single event or sub-grid interval must still occupy its nearest cell.
        empty = ~support.any(dim=1)
        if empty.any():
            midpoint = (lower[empty] + upper[empty]) / 2
            nearest = (grid[None] - midpoint[:, None]).abs().argmin(dim=1)
            rows = empty.nonzero(as_tuple=False).flatten()
            support[rows, nearest] = True
        return support

    def encode_events(self, x: Tensor, x_time: Tensor, x_mask: Tensor) -> Tensor:
        if x.shape != x_mask.shape or x.ndim != 3 or x_time.shape != x.shape[:2]:
            raise ValueError("expected x/x_mask [B,L,C] and x_time [B,L]")
        if x.shape[-1] != self.channels:
            raise ValueError("input channel count does not match model")
        if not torch.isfinite(x_mask).all() or not torch.all((x_mask == 0) | (x_mask == 1)):
            raise ValueError("x_mask must be finite and binary")
        valid = x_mask.bool().any(dim=-1)
        if not valid.any(dim=1).all():
            raise ValueError("each sample needs at least one observed condition")
        features = torch.cat((x * x_mask, x_mask), dim=-1)
        latent = self.event_projection(features) + self.time_embedding(x_time)
        grid = interpolate_latent_grid(x_time, self.event_norm(latent), valid, self.grid_size)
        support = self.context_support(x_time, valid, self.grid_size).unsqueeze(-1)
        grid = grid * support
        for block in self.frequency_blocks:
            grid = block(grid) * support
        return grid


def _masked_channel_macro(loss: Tensor, mask: Tensor, reduction: str) -> Tensor:
    weights = mask.to(loss.dtype)
    counts = weights.sum(dim=1)
    active = counts > 0
    if not active.any(dim=1).all():
        raise ValueError("each sample needs at least one scored channel")
    per_channel = (loss * weights).sum(dim=1) / counts.clamp_min(1)
    per_sample = (per_channel * active).sum(dim=1) / active.sum(dim=1)
    if reduction == "none":
        return per_sample
    if reduction != "mean":
        raise ValueError("reduction must be mean or none")
    return per_sample.mean()


class HeteroscedasticGaussian(nn.Module):
    """Factorized direct Gaussian diagnostic, not a diffusion replacement claim."""

    sampler_implementation = "direct_heteroscedastic_gaussian_v2"
    distribution_family = "direct_gaussian"

    def __init__(self, denoiser: SupportAwareDenoiser, min_scale: float = 1e-4) -> None:
        super().__init__()
        if min_scale <= 0:
            raise ValueError("min_scale must be positive")
        self.denoiser = denoiser
        self.channels = denoiser.channels
        self.min_scale = float(min_scale)
        denoiser.output = nn.Linear(denoiser.dim, 2 * denoiser.channels)

    def distribution(
        self, x: Tensor, x_time: Tensor, x_mask: Tensor, y_time: Tensor, y_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        empty = x.new_zeros((x.shape[0], y_time.shape[1], self.channels))
        step = x.new_zeros((x.shape[0],))
        output = self.denoiser(empty, step, x, x_time, x_mask, y_time, y_mask)
        mean, raw_scale = output.chunk(2, dim=-1)
        return mean, F.softplus(raw_scale) + self.min_scale

    def training_loss(
        self,
        target: Tensor,
        target_mask: Tensor,
        x: Tensor,
        x_time: Tensor,
        x_mask: Tensor,
        y_time: Tensor,
        *,
        generator: torch.Generator | None = None,
        reduction: str = "mean",
        **_: object,
    ) -> Tensor:
        del generator
        mean, scale = self.distribution(x, x_time, x_mask, y_time, target_mask)
        nll = 0.5 * ((target - mean) / scale).square() + scale.log() + 0.5 * math.log(2 * math.pi)
        return _masked_channel_macro(nll, target_mask, reduction)

    @torch.inference_mode()
    def sample(
        self,
        x: Tensor,
        x_time: Tensor,
        x_mask: Tensor,
        y_time: Tensor,
        *,
        y_mask: Tensor,
        channels: int,
        n_samples: int,
        generator: torch.Generator | None = None,
        **_: object,
    ) -> Tensor:
        if channels != self.channels or n_samples < 2:
            raise ValueError("channels must match and n_samples must be at least two")
        mean, scale = self.distribution(x, x_time, x_mask, y_time, y_mask)
        random_device = x.device if generator is None else torch.device(generator.device)
        noise = torch.randn(
            (x.shape[0], n_samples, y_time.shape[1], channels),
            device=random_device,
            dtype=x.dtype,
            generator=generator,
        )
        if random_device != x.device:
            noise = noise.to(x.device)
        return mean[:, None] + scale[:, None] * noise


def build_model(channels: int, distribution_family: str, config: Mapping[str, object]) -> nn.Module:
    if distribution_family not in FAMILIES:
        raise ValueError(f"unknown predictive distribution family {distribution_family!r}")
    denoiser = SupportAwareDenoiser(
        channels=channels,
        dim=int(config["dim"]),
        grid_size=int(config["grid_size"]),
        layers=int(config["layers"]),
        heads=int(config["heads"]),
        ff_dim=int(config["ff_dim"]),
        dropout=float(config["dropout"]),
        variant="full" if distribution_family == "conditional_diffusion" else "deterministic_head",
    )
    if distribution_family == "conditional_diffusion":
        model = ConditionalDiffusion(denoiser, train_steps=int(config["train_steps"]))
        model.distribution_family = distribution_family
        return model
    return HeteroscedasticGaussian(denoiser, float(config["gaussian_min_scale"]))
