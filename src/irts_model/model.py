"""Latent-grid denoiser for irregular multivariate time series.

Frequency-domain operations are applied only after irregular events have been
embedded and linearly materialized on a uniform latent grid.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


VARIANT_DESCRIPTIONS = {
    "full": "latent interpolation followed by frequency-gated attention with full complex projections",
    "time_attention": "frequency blocks replaced by standard temporal MHA; FF width adjusted to match parameters",
    "diagonal_complex": "full complex frequency projections replaced by diagonal complex projections",
    "raw_interpolation": "masked raw values and masks interpolated before event encoding; capacity identical to full",
    "deterministic_head": "single direct masked-MSE regression with no diffusion noise or probabilistic samples",
}


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int, scale: float = 1_000.0) -> None:
        super().__init__()
        if dim < 4 or dim % 2:
            raise ValueError("embedding dimension must be even and at least four")
        half = dim // 2
        frequencies = torch.exp(
            -math.log(10_000.0) * torch.arange(half, dtype=torch.float32) / max(half - 1, 1)
        )
        self.register_buffer("frequencies", frequencies)
        self.scale = scale

    def forward(self, values: Tensor) -> Tensor:
        angles = values.unsqueeze(-1) * self.scale * self.frequencies
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


def _merge_duplicate_times(times: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
    unique, inverse, counts = torch.unique_consecutive(
        times, return_inverse=True, return_counts=True
    )
    if unique.numel() == times.numel():
        return times, values
    merged = values.new_zeros((unique.numel(), values.shape[-1]))
    merged.index_add_(0, inverse, values)
    return unique, merged / counts.to(values.dtype).unsqueeze(-1)


def interpolate_latent_grid(
    times: Tensor,
    latent: Tensor,
    valid_events: Tensor,
    grid_size: int,
) -> Tensor:
    """Piecewise-linear latent interpolation onto [0,1], with boundary holding."""
    if times.ndim != 2 or latent.ndim != 3 or valid_events.shape != times.shape:
        raise ValueError("expected times/valid [B,L] and latent [B,L,D]")
    if latent.shape[:2] != times.shape:
        raise ValueError("latent and times must share [B,L]")
    if grid_size < 2:
        raise ValueError("grid_size must be at least two")

    grid = torch.linspace(0.0, 1.0, grid_size, device=times.device, dtype=times.dtype)
    outputs: list[Tensor] = []
    for batch_index in range(times.shape[0]):
        keep = valid_events[batch_index].bool()
        event_times = times[batch_index, keep]
        event_latent = latent[batch_index, keep]
        if event_times.numel() == 0:
            outputs.append(latent.new_zeros((grid_size, latent.shape[-1])))
            continue
        if not torch.isfinite(event_times).all() or event_times.min() < 0 or event_times.max() > 1:
            raise ValueError("valid event times must be finite and normalized to [0, 1]")

        order = torch.argsort(event_times, stable=True)
        event_times, event_latent = _merge_duplicate_times(
            event_times[order], event_latent[order]
        )
        if event_times.numel() == 1:
            outputs.append(event_latent.expand(grid_size, -1))
            continue

        right = torch.searchsorted(event_times, grid, right=False)
        right = right.clamp(1, event_times.numel() - 1)
        left = right - 1
        left_time = event_times[left]
        right_time = event_times[right]
        weight = ((grid - left_time) / (right_time - left_time).clamp_min(1e-8)).unsqueeze(-1)
        interpolated = event_latent[left] + weight * (event_latent[right] - event_latent[left])
        interpolated = torch.where(
            (grid <= event_times[0]).unsqueeze(-1), event_latent[0], interpolated
        )
        interpolated = torch.where(
            (grid >= event_times[-1]).unsqueeze(-1), event_latent[-1], interpolated
        )
        outputs.append(interpolated)
    return torch.stack(outputs)


class ComplexLinear(nn.Module):
    """Full complex affine map; off-diagonal parameters participate in the forward pass."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        scale = dim**-0.5
        self.real_weight = nn.Parameter(torch.empty(dim, dim).normal_(std=scale))
        self.imag_weight = nn.Parameter(torch.empty(dim, dim).normal_(std=scale))
        self.real_bias = nn.Parameter(torch.zeros(dim))
        self.imag_bias = nn.Parameter(torch.zeros(dim))

    def forward(self, values: Tensor) -> Tensor:
        real = torch.einsum("bfd,dh->bfh", values.real, self.real_weight)
        real = real - torch.einsum("bfd,dh->bfh", values.imag, self.imag_weight)
        imag = torch.einsum("bfd,dh->bfh", values.real, self.imag_weight)
        imag = imag + torch.einsum("bfd,dh->bfh", values.imag, self.real_weight)
        return torch.complex(real + self.real_bias, imag + self.imag_bias)


class DiagonalComplexLinear(nn.Module):
    """Channel-wise complex affine map used by the diagonal projection ablation."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        scale = dim**-0.5
        self.real_weight = nn.Parameter(torch.empty(dim).normal_(std=scale))
        self.imag_weight = nn.Parameter(torch.empty(dim).normal_(std=scale))
        self.real_bias = nn.Parameter(torch.zeros(dim))
        self.imag_bias = nn.Parameter(torch.zeros(dim))

    def forward(self, values: Tensor) -> Tensor:
        real = values.real * self.real_weight - values.imag * self.imag_weight
        imag = values.real * self.imag_weight + values.imag * self.real_weight
        return torch.complex(real + self.real_bias, imag + self.imag_bias)


class FrequencyGatedAttention(nn.Module):
    def __init__(
        self, dim: int, heads: int, grid_size: int, dropout: float, diagonal: bool = False
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.grid_size = grid_size
        n_frequencies = grid_size // 2 + 1
        gate_init = math.log(math.e - 1.0)
        self.query_gate = nn.Parameter(torch.full((n_frequencies, dim), gate_init))
        self.key_gate = nn.Parameter(torch.full((n_frequencies, dim), gate_init))
        self.value_gate = nn.Parameter(torch.full((n_frequencies, dim), gate_init))
        projection = DiagonalComplexLinear if diagonal else ComplexLinear
        self.query_projection = projection(dim)
        self.key_projection = projection(dim)
        self.value_projection = projection(dim)
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, dim)
        self.dropout = float(dropout)

    def _project(self, spectrum: Tensor, gate: Tensor, projection: ComplexLinear) -> Tensor:
        gated = spectrum * F.softplus(gate).unsqueeze(0)
        projected = projection(gated)
        return torch.fft.irfft(projected, n=self.grid_size, dim=1, norm="ortho")

    def _heads(self, values: Tensor) -> Tensor:
        batch, length, _ = values.shape
        return values.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, values: Tensor) -> Tensor:
        residual = values
        normalized = self.norm(values).float()
        spectrum = torch.fft.rfft(normalized, n=self.grid_size, dim=1, norm="ortho")
        query = self._heads(self._project(spectrum, self.query_gate, self.query_projection))
        key = self._heads(self._project(spectrum, self.key_gate, self.key_projection))
        value = self._heads(self._project(spectrum, self.value_gate, self.value_projection))
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view_as(residual)
        return residual + F.dropout(self.output(attended), self.dropout, self.training)


class FrequencyBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        grid_size: int,
        ff_dim: int,
        dropout: float,
        diagonal: bool = False,
    ) -> None:
        super().__init__()
        self.attention = FrequencyGatedAttention(dim, heads, grid_size, dropout, diagonal)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
        )
        self.dropout = dropout

    def forward(self, values: Tensor) -> Tensor:
        values = self.attention(values)
        return values + F.dropout(self.ff(self.ff_norm(values)), self.dropout, self.training)


def _matched_time_ff_dim(dim: int, grid_size: int, ff_dim: int) -> int:
    """Widen the temporal FFN by the parameter gap left by standard real-valued MHA."""
    frequencies = grid_size // 2 + 1
    attention_gap = 3 * dim * dim + 3 * dim + 3 * frequencies * dim
    return ff_dim + round(attention_gap / (2 * dim + 1))


class TimeAttentionBlock(nn.Module):
    """Standard time-domain self-attention with block-level parameter matching."""

    def __init__(
        self, dim: int, heads: int, grid_size: int, ff_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(dim)
        matched_ff_dim = _matched_time_ff_dim(dim, grid_size, ff_dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, matched_ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(matched_ff_dim, dim),
        )
        self.dropout = dropout

    def forward(self, values: Tensor) -> Tensor:
        normalized = self.attention_norm(values)
        update = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        values = values + F.dropout(update, self.dropout, self.training)
        return values + F.dropout(self.ff(self.ff_norm(values)), self.dropout, self.training)


class QueryBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.cross_norm = nn.LayerNorm(dim)
        self.ff_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, dim)
        )
        self.dropout = dropout

    def forward(
        self, query: Tensor, context: Tensor, query_valid: Tensor | None = None
    ) -> Tensor:
        key_padding_mask = None
        if query_valid is not None:
            if query_valid.shape != query.shape[:2]:
                raise ValueError("query_valid must have shape [B,L]")
            if not query_valid.bool().any(dim=1).all():
                raise ValueError("every sample needs at least one valid query")
            key_padding_mask = ~query_valid.bool()
        normalized = self.self_norm(query)
        update = self.self_attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        query = query + F.dropout(update, self.dropout, self.training)
        update = self.cross_attention(
            self.cross_norm(query), self.context_norm(context), self.context_norm(context), need_weights=False
        )[0]
        query = query + F.dropout(update, self.dropout, self.training)
        return query + F.dropout(self.ff(self.ff_norm(query)), self.dropout, self.training)


class LatentGridDenoiser(nn.Module):
    def __init__(
        self,
        channels: int,
        dim: int = 64,
        grid_size: int = 32,
        layers: int = 3,
        heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
        variant: str = "full",
    ) -> None:
        super().__init__()
        if variant not in VARIANT_DESCRIPTIONS:
            raise ValueError(f"unknown model variant {variant!r}")
        self.channels = channels
        self.dim = dim
        self.grid_size = grid_size
        self.variant = variant
        self.variant_description = VARIANT_DESCRIPTIONS[variant]
        self.event_projection = nn.Linear(2 * channels, dim)
        self.time_embedding = SinusoidalEmbedding(dim)
        if variant == "deterministic_head":
            self.noisy_projection = None
            self.step_embedding = None
            self.query_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.noisy_projection = nn.Linear(channels, dim)
            self.step_embedding = SinusoidalEmbedding(dim)
            self.register_parameter("query_bias", None)
        self.event_norm = nn.LayerNorm(dim)
        if variant == "time_attention":
            blocks = [
                TimeAttentionBlock(dim, heads, grid_size, ff_dim, dropout)
                for _ in range(layers)
            ]
        else:
            blocks = [
                FrequencyBlock(
                    dim,
                    heads,
                    grid_size,
                    ff_dim,
                    dropout,
                    diagonal=variant == "diagonal_complex",
                )
                for _ in range(layers)
            ]
        self.frequency_blocks = nn.ModuleList(blocks)
        self.query_blocks = nn.ModuleList(
            [QueryBlock(dim, heads, ff_dim, dropout) for _ in range(layers)]
        )
        self.output_norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, channels)

    def encode_events(self, x: Tensor, x_time: Tensor, x_mask: Tensor) -> Tensor:
        if x.shape != x_mask.shape or x.ndim != 3 or x_time.shape != x.shape[:2]:
            raise ValueError("expected x/x_mask [B,L,C] and x_time [B,L]")
        if x.shape[-1] != self.channels:
            raise ValueError("input channel count does not match model")
        if not torch.isfinite(x_mask).all() or not torch.all((x_mask == 0) | (x_mask == 1)):
            raise ValueError("x_mask must be finite and binary")
        if not x_mask.bool().flatten(1).any(dim=1).all():
            raise ValueError("each sample needs at least one observed condition")
        features = torch.cat((x * x_mask, x_mask), dim=-1)
        valid = x_mask.bool().any(dim=-1)
        if self.variant == "raw_interpolation":
            raw_grid = interpolate_latent_grid(x_time, features, valid, self.grid_size)
            grid_time = torch.linspace(
                0.0, 1.0, self.grid_size, device=x_time.device, dtype=x_time.dtype
            ).expand(x.shape[0], -1)
            grid = self.event_norm(
                self.event_projection(raw_grid) + self.time_embedding(grid_time)
            )
        else:
            latent = self.event_projection(features) + self.time_embedding(x_time)
            grid = interpolate_latent_grid(
                x_time, self.event_norm(latent), valid, self.grid_size
            )
        for block in self.frequency_blocks:
            grid = block(grid)
        return grid

    def forward_with_context(
        self,
        noisy_target: Tensor,
        diffusion_step: Tensor,
        context: Tensor,
        y_time: Tensor,
        y_mask: Tensor | None = None,
    ) -> Tensor:
        """Evaluate query tokens against an already encoded conditioning context."""
        if noisy_target.ndim != 3 or y_time.shape != noisy_target.shape[:2]:
            raise ValueError("expected noisy_target [B,L,C] and y_time [B,L]")
        if noisy_target.shape[-1] != self.channels:
            raise ValueError("noisy_target channel count does not match model")
        if diffusion_step.shape != (noisy_target.shape[0],):
            raise ValueError("diffusion_step must have shape [B]")
        if context.shape != (noisy_target.shape[0], self.grid_size, self.dim):
            raise ValueError("context must have shape [B,grid_size,dim]")
        if y_mask is None:
            query_valid = torch.ones_like(y_time, dtype=torch.bool)
        else:
            if y_mask.shape != noisy_target.shape:
                raise ValueError("y_mask must match noisy_target [B,L,C]")
            query_valid = y_mask.bool().any(dim=-1)
        valid_times = y_time[query_valid]
        if (
            not torch.isfinite(valid_times).all()
            or valid_times.min() < 0
            or valid_times.max() > 1
        ):
            raise ValueError("valid query times must be finite and normalized to [0, 1]")
        if self.variant == "deterministic_head":
            query = self.time_embedding(y_time) + self.query_bias
        else:
            assert self.noisy_projection is not None and self.step_embedding is not None
            query = self.noisy_projection(noisy_target) + self.time_embedding(y_time)
            query = query + self.step_embedding(diffusion_step).unsqueeze(1)
        for block in self.query_blocks:
            query = block(query, context, query_valid)
        return self.output(self.output_norm(query))

    def forward(
        self,
        noisy_target: Tensor,
        diffusion_step: Tensor,
        x: Tensor,
        x_time: Tensor,
        x_mask: Tensor,
        y_time: Tensor,
        y_mask: Tensor | None = None,
    ) -> Tensor:
        context = self.encode_events(x, x_time, x_mask)
        return self.forward_with_context(
            noisy_target, diffusion_step, context, y_time, y_mask
        )
