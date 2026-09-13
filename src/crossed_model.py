"""Cross only embedding and raster coordinates; retain the R5 task and weights.

History time is explicitly [B,L,2]: [...,0] is embedding (E), [...,1] is
raster/support (R). Query time remains [B,P], using E. The original future
query mask is intentionally retained: this is a retrospective component
experiment, not a forecast-origin-available task repair.
"""

from typing import Mapping

import torch
from torch import Tensor, nn

from geometry_screen.models import FAMILIES, HeteroscedasticGaussian, SupportAwareDenoiser
from irts_model import ConditionalDiffusion, interpolate_latent_grid


class CrossedSupportAwareDenoiser(SupportAwareDenoiser):
    """No new parameters; override the shared training and cached-sampling path."""

    def encode_events(self, x: Tensor, x_time: Tensor, x_mask: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape != x_mask.shape or x_time.shape != (*x.shape[:2], 2):
            raise ValueError("expected x/x_mask [B,L,C] and explicit E/R times [B,L,2]")
        if x.shape[-1] != self.channels:
            raise ValueError("input channel count does not match model")
        if not torch.isfinite(x_mask).all() or not torch.all((x_mask == 0) | (x_mask == 1)):
            raise ValueError("x_mask must be finite and binary")
        valid = x_mask.bool().any(dim=-1)
        if not valid.any(dim=1).all():
            raise ValueError("each sample needs at least one observed condition")
        embedding_time, raster_time = x_time.unbind(dim=-1)
        valid_embedding = embedding_time[valid]
        if not torch.isfinite(valid_embedding).all() or valid_embedding.min() < 0 or valid_embedding.max() > 1:
            raise ValueError("valid embedding times must be finite and normalized to [0,1]")
        features = torch.cat((x * x_mask, x_mask), dim=-1)
        latent = self.event_projection(features) + self.time_embedding(embedding_time)
        grid = interpolate_latent_grid(raster_time, self.event_norm(latent), valid, self.grid_size)
        support = self.context_support(raster_time, valid, self.grid_size).unsqueeze(-1)
        grid = grid * support
        for block in self.frequency_blocks:
            grid = block(grid) * support
        return grid


def build_crossed_model(channels: int, distribution_family: str, config: Mapping[str, object]) -> nn.Module:
    """Use the original family construction with only the E/R-aware subclass."""
    if distribution_family not in FAMILIES:
        raise ValueError(f"unknown predictive distribution family {distribution_family!r}")
    denoiser = CrossedSupportAwareDenoiser(
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
