"""Masked epsilon-objective diffusion and deterministic DDIM sampling."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def cosine_betas(steps: int, offset: float = 0.008) -> Tensor:
    points = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    cumulative = torch.cos(((points / steps + offset) / (1 + offset)) * math.pi / 2) ** 2
    cumulative = cumulative / cumulative[0]
    return (1 - cumulative[1:] / cumulative[:-1]).clamp(1e-5, 0.999).float()


def _extract(values: Tensor, indices: Tensor, target: Tensor) -> Tensor:
    return values[indices].view(indices.shape[0], *([1] * (target.ndim - 1)))


def _random_device(reference: Tensor, generator: torch.Generator | None) -> torch.device:
    return reference.device if generator is None else torch.device(generator.device)


def _randn(
    shape: tuple[int, ...], reference: Tensor, generator: torch.Generator | None
) -> Tensor:
    device = _random_device(reference, generator)
    values = torch.randn(shape, device=device, dtype=reference.dtype, generator=generator)
    return values if device == reference.device else values.to(reference.device)


def _randint(
    high: int, shape: tuple[int, ...], reference: Tensor, generator: torch.Generator | None
) -> Tensor:
    device = _random_device(reference, generator)
    values = torch.randint(high, shape, device=device, generator=generator)
    return values if device == reference.device else values.to(reference.device)


def _validate_target_mask(target: Tensor, target_mask: Tensor) -> None:
    if target.shape != target_mask.shape or target.ndim != 3:
        raise ValueError("target and target_mask must share [B,L,C]")
    if not torch.isfinite(target_mask).all() or not torch.all(
        (target_mask == 0) | (target_mask == 1)
    ):
        raise ValueError("target_mask must be finite and binary")
    if not target_mask.bool().flatten(1).any(dim=1).all():
        raise ValueError("target_mask must select at least one value per sample")


def _masked_mse(
    prediction: Tensor,
    target: Tensor,
    target_mask: Tensor,
    reduction: str,
) -> Tensor:
    mask = target_mask.to(target.dtype)
    squared = (prediction - target).square() * mask
    channel_count = mask.sum(dim=1)
    active_channel = channel_count > 0
    channel_mse = squared.sum(dim=1) / channel_count.clamp_min(1)
    sample_loss = (channel_mse * active_channel).sum(dim=1) / active_channel.sum(dim=1)
    if reduction == "none":
        return sample_loss
    if reduction != "mean":
        raise ValueError("reduction must be mean or none")
    return sample_loss.mean()


class ConditionalDiffusion(nn.Module):
    sampler_implementation = "ddim_context_cache_v1"

    def __init__(self, denoiser: nn.Module, train_steps: int = 1_000) -> None:
        super().__init__()
        if train_steps < 2:
            raise ValueError("train_steps must be at least two")
        betas = cosine_betas(train_steps)
        alphas = 1.0 - betas
        self.denoiser = denoiser
        self.variant = str(getattr(denoiser, "variant", "full"))
        self.variant_description = str(getattr(denoiser, "variant_description", ""))
        self.train_steps = train_steps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", torch.cumprod(alphas, dim=0))

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
        step: Tensor | None = None,
        noise: Tensor | None = None,
        reduction: str = "mean",
    ) -> Tensor:
        _validate_target_mask(target, target_mask)
        batch = target.shape[0]
        if step is None:
            step = _randint(self.train_steps, (batch,), target, generator)
        else:
            if step.shape != (batch,) or step.dtype not in (torch.int32, torch.int64):
                raise ValueError("step must be an integer tensor with shape [B]")
            step = step.to(target.device, dtype=torch.long)
            if torch.any((step < 0) | (step >= self.train_steps)):
                raise ValueError("step values must be in [0, train_steps)")
        if noise is None:
            noise = _randn(tuple(target.shape), target, generator)
        else:
            if noise.shape != target.shape or not torch.isfinite(noise).all():
                raise ValueError("noise must be finite and match target [B,L,C]")
            noise = noise.to(target.device, dtype=target.dtype)
        alpha_bar = _extract(self.alpha_bars, step, target)
        noisy = alpha_bar.sqrt() * target + (1.0 - alpha_bar).sqrt() * noise
        predicted = self.denoiser(
            noisy,
            step.to(target.dtype) / (self.train_steps - 1),
            x,
            x_time,
            x_mask,
            y_time,
            target_mask,
        )
        return _masked_mse(predicted, noise, target_mask, reduction)

    def _sampling_steps(self, count: int) -> list[int]:
        if not 1 <= count <= self.train_steps:
            raise ValueError("ddim_steps must be in [1, train_steps]")
        values = torch.linspace(self.train_steps - 1, 0, count).round().long().tolist()
        return list(dict.fromkeys(values))

    @torch.inference_mode()
    def sample(
        self,
        x: Tensor,
        x_time: Tensor,
        x_mask: Tensor,
        y_time: Tensor,
        *,
        y_mask: Tensor | None = None,
        channels: int,
        n_samples: int = 100,
        ddim_steps: int = 50,
        sample_chunk: int = 10,
        generator: torch.Generator | None = None,
        x0_clip: float | None = None,
    ) -> Tensor:
        if n_samples < 2 or sample_chunk < 1:
            raise ValueError("n_samples must be at least two and sample_chunk positive")
        if x0_clip is not None and (not math.isfinite(x0_clip) or x0_clip <= 0):
            raise ValueError("x0_clip must be finite and positive")
        if hasattr(self.denoiser, "channels") and channels != self.denoiser.channels:
            raise ValueError("channels must match the denoiser")
        batch, target_length = y_time.shape
        if y_mask is None:
            y_mask = torch.ones(
                (batch, target_length, channels),
                dtype=torch.bool,
                device=y_time.device,
            )
        elif y_mask.shape != (batch, target_length, channels):
            raise ValueError("y_mask must have shape [B,L,C]")
        if not torch.isfinite(y_mask).all() or not torch.all((y_mask == 0) | (y_mask == 1)):
            raise ValueError("y_mask must be finite and binary")
        if not y_mask.bool().flatten(1).any(dim=1).all():
            raise ValueError("y_mask must select at least one query per sample")
        schedule = self._sampling_steps(ddim_steps)
        chunks: list[Tensor] = []
        initial_noise = _randn(
            (batch, n_samples, target_length, channels), x, generator
        )
        was_training = self.training
        self.eval()
        try:
            cache_supported = callable(getattr(self.denoiser, "encode_events", None)) and callable(
                getattr(self.denoiser, "forward_with_context", None)
            )
            context = (
                self.denoiser.encode_events(x, x_time, x_mask) if cache_supported else None
            )
            for start in range(0, n_samples, sample_chunk):
                draws = min(sample_chunk, n_samples - start)
                repeat = lambda value: value.repeat_interleave(draws, dim=0)
                query_time = repeat(y_time)
                query_mask = repeat(y_mask)
                repeated_context = repeat(context) if context is not None else None
                if repeated_context is None:
                    condition_x = repeat(x)
                    condition_time = repeat(x_time)
                    condition_mask = repeat(x_mask)
                current = initial_noise[:, start : start + draws].reshape(
                    batch * draws, target_length, channels
                )
                for index, step_value in enumerate(schedule):
                    step = torch.full(
                        (batch * draws,),
                        step_value / (self.train_steps - 1),
                        device=x.device,
                        dtype=x.dtype,
                    )
                    if repeated_context is None:
                        predicted_noise = self.denoiser(
                            current,
                            step,
                            condition_x,
                            condition_time,
                            condition_mask,
                            query_time,
                            query_mask,
                        )
                    else:
                        predicted_noise = self.denoiser.forward_with_context(
                            current, step, repeated_context, query_time, query_mask
                        )
                    alpha_bar = self.alpha_bars[step_value].to(current.dtype)
                    clean = (current - (1.0 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
                    if x0_clip is not None:
                        clean = clean.clamp(-x0_clip, x0_clip)
                        predicted_noise = (
                            current - alpha_bar.sqrt() * clean
                        ) / (1.0 - alpha_bar).sqrt().clamp_min(1e-8)
                    if index == len(schedule) - 1:
                        current = clean
                    else:
                        previous = self.alpha_bars[schedule[index + 1]].to(current.dtype)
                        current = previous.sqrt() * clean + (1.0 - previous).sqrt() * predicted_noise
                chunks.append(current.view(batch, draws, target_length, channels))
        finally:
            self.train(was_training)
        return torch.cat(chunks, dim=1)


class DeterministicRegressor(nn.Module):
    """Single-output direct regression; deliberately exposes no predictive distribution."""

    sampler_implementation = "deterministic_predict_v1"

    def __init__(self, denoiser: nn.Module) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.variant = "deterministic_head"
        self.variant_description = str(getattr(denoiser, "variant_description", ""))

    def _prediction(
        self,
        x: Tensor,
        x_time: Tensor,
        x_mask: Tensor,
        y_time: Tensor,
        y_mask: Tensor | None,
    ) -> Tensor:
        channels = int(getattr(self.denoiser, "channels"))
        empty_target = x.new_zeros((x.shape[0], y_time.shape[1], channels))
        step = x.new_zeros((x.shape[0],))
        return self.denoiser(
            empty_target, step, x, x_time, x_mask, y_time, y_mask
        )

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
        step: Tensor | None = None,
        noise: Tensor | None = None,
        reduction: str = "mean",
    ) -> Tensor:
        del generator, step, noise
        _validate_target_mask(target, target_mask)
        prediction = self._prediction(x, x_time, x_mask, y_time, target_mask)
        return _masked_mse(prediction, target, target_mask, reduction)

    @torch.inference_mode()
    def predict(
        self,
        x: Tensor,
        x_time: Tensor,
        x_mask: Tensor,
        y_time: Tensor,
        *,
        y_mask: Tensor | None = None,
        channels: int,
    ) -> Tensor:
        if channels != getattr(self.denoiser, "channels", None):
            raise ValueError("channels must match the deterministic regressor")
        was_training = self.training
        self.eval()
        try:
            return self._prediction(x, x_time, x_mask, y_time, y_mask)
        finally:
            self.train(was_training)

    @torch.inference_mode()
    def sample(
        self,
        x: Tensor,
        x_time: Tensor,
        x_mask: Tensor,
        y_time: Tensor,
        *,
        y_mask: Tensor | None = None,
        channels: int,
        n_samples: int = 1,
        ddim_steps: int = 1,
        sample_chunk: int = 1,
        generator: torch.Generator | None = None,
        x0_clip: float | None = None,
    ) -> Tensor:
        del ddim_steps, sample_chunk, generator, x0_clip
        if n_samples != 1:
            raise ValueError(
                "deterministic_head is non-probabilistic; use predict and leave CRPS null"
            )
        return self.predict(
            x, x_time, x_mask, y_time, y_mask=y_mask, channels=channels
        ).unsqueeze(1)
