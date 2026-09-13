"""r5 marginal metrics and limited joint diagnostics for development draws."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


QUANTILE_METHOD = "weibull"
QUANTILE_PROBABILITIES = (0.025, 0.975)
VARIOGRAM_POWER = 0.5


def _validate(samples: object, target: object, mask: object, expected_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = np.asarray(samples, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    raw_mask = np.asarray(mask)
    if target.ndim != 3 or raw_mask.shape != target.shape:
        raise ValueError("target/mask must share [B,L,C]")
    if samples.ndim != 4 or samples.shape[0] != target.shape[0] or samples.shape[2:] != target.shape[1:]:
        raise ValueError("samples must have [B,S,L,C] matching target")
    if samples.shape[1] != expected_samples:
        raise ValueError(f"expected exactly {expected_samples} trajectories")
    if not np.isfinite(samples).all() or not np.isfinite(target).all() or not np.isfinite(raw_mask).all():
        raise ValueError("samples, targets, and masks must be finite")
    if not np.isin(raw_mask, (0, 1)).all():
        raise ValueError("mask must be binary")
    mask = raw_mask.astype(bool, copy=False)
    if not mask.reshape(mask.shape[0], -1).any(axis=1).all():
        raise ValueError("each trajectory needs a scored dimension")
    return samples, target, mask


def empirical_crps_pointwise(samples: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Exact empirical-CDF marginal CRPS for [B,S,L,C] draws."""
    count = samples.shape[1]
    ordered = np.sort(samples, axis=1)
    weights = (2 * np.arange(count) - count + 1).reshape(1, count, 1, 1)
    return np.abs(samples - target[:, None]).mean(axis=1) - (ordered * weights).sum(axis=1) / count**2


def masked_dimension_normalized_energy(
    samples: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Trajectory energy score per batch item, normalized by sqrt(scored dimensions)."""
    values: list[float] = []
    for draws, truth, scored in zip(samples, target, mask):
        keep = scored.reshape(-1)
        x = draws.reshape(draws.shape[0], -1)[:, keep]
        y = truth.reshape(-1)[keep]
        dimension = x.shape[1]
        scale = math.sqrt(dimension)
        first = np.linalg.norm(x - y[None], axis=1).mean() / scale
        squared = np.square(x).sum(axis=1)
        pairwise_squared = np.maximum(squared[:, None] + squared[None] - 2 * (x @ x.T), 0.0)
        second = np.sqrt(pairwise_squared).mean() / scale
        values.append(float(first - 0.5 * second))
    return np.asarray(values, dtype=np.float64)


def lag1_temporal_variogram_pointwise(
    samples: np.ndarray, target: np.ndarray, mask: np.ndarray, power: float = VARIOGRAM_POWER
) -> tuple[np.ndarray, np.ndarray]:
    """Squared lag-1 variogram discrepancy; same-channel temporal pairs only."""
    if not 0 < power <= 2:
        raise ValueError("variogram power must be in (0,2]")
    valid = mask[:, 1:] & mask[:, :-1]
    observed = np.abs(target[:, 1:] - target[:, :-1]) ** power
    predicted = (np.abs(samples[:, :, 1:] - samples[:, :, :-1]) ** power).mean(axis=1)
    return np.square(observed - predicted), valid


class TrajectoryMetrics:
    def __init__(self, expected_samples: int = 100, nominal_coverage: float = 0.95) -> None:
        if expected_samples < 2 or not 0 < nominal_coverage < 1:
            raise ValueError("invalid sample count or nominal coverage")
        self.expected_samples = int(expected_samples)
        self.nominal = float(nominal_coverage)
        self.count: np.ndarray | None = None
        self.crps_sum: np.ndarray | None = None
        self.crps_first_half_sum: np.ndarray | None = None
        self.crps_last_half_sum: np.ndarray | None = None
        self.coverage_sum: np.ndarray | None = None
        self.coverage_first_half_sum: np.ndarray | None = None
        self.coverage_last_half_sum: np.ndarray | None = None
        self.width_sum: np.ndarray | None = None
        self.interval_sum: np.ndarray | None = None
        self.variogram_count: np.ndarray | None = None
        self.variogram_sum: np.ndarray | None = None
        self.energy_sum = 0.0
        self.energy_count = 0

    def update(self, samples: object, target: object, mask: object) -> None:
        samples, target, mask = _validate(samples, target, mask, self.expected_samples)
        channels = target.shape[-1]
        if self.count is None:
            self.count = np.zeros(channels, dtype=np.int64)
            self.crps_sum = np.zeros(channels)
            self.crps_first_half_sum = np.zeros(channels)
            self.crps_last_half_sum = np.zeros(channels)
            self.coverage_sum = np.zeros(channels)
            self.coverage_first_half_sum = np.zeros(channels)
            self.coverage_last_half_sum = np.zeros(channels)
            self.width_sum = np.zeros(channels)
            self.interval_sum = np.zeros(channels)
            self.variogram_count = np.zeros(channels, dtype=np.int64)
            self.variogram_sum = np.zeros(channels)
        elif len(self.count) != channels:
            raise ValueError("channel count changed")
        alpha = 1.0 - self.nominal
        midpoint = self.expected_samples // 2
        if self.expected_samples != 100 or midpoint != 50:
            raise ValueError("MC sensitivity is frozen to first50 versus last50 of 100 draws")
        lower, upper = np.quantile(
            samples, QUANTILE_PROBABILITIES, axis=1, method=QUANTILE_METHOD
        )
        first_lower, first_upper = np.quantile(
            samples[:, :midpoint], QUANTILE_PROBABILITIES, axis=1, method=QUANTILE_METHOD
        )
        last_lower, last_upper = np.quantile(
            samples[:, midpoint:], QUANTILE_PROBABILITIES, axis=1, method=QUANTILE_METHOD
        )
        crps = empirical_crps_pointwise(samples, target)
        crps_first = empirical_crps_pointwise(samples[:, :midpoint], target)
        crps_last = empirical_crps_pointwise(samples[:, midpoint:], target)
        covered = (target >= lower) & (target <= upper)
        covered_first = (target >= first_lower) & (target <= first_upper)
        covered_last = (target >= last_lower) & (target <= last_upper)
        width = upper - lower
        interval = width.copy()
        interval += (2.0 / alpha) * (lower - target) * (target < lower)
        interval += (2.0 / alpha) * (target - upper) * (target > upper)
        self.count += mask.sum(axis=(0, 1))
        self.crps_sum += (crps * mask).sum(axis=(0, 1))
        self.crps_first_half_sum += (crps_first * mask).sum(axis=(0, 1))
        self.crps_last_half_sum += (crps_last * mask).sum(axis=(0, 1))
        self.coverage_sum += (covered * mask).sum(axis=(0, 1))
        self.coverage_first_half_sum += (covered_first * mask).sum(axis=(0, 1))
        self.coverage_last_half_sum += (covered_last * mask).sum(axis=(0, 1))
        self.width_sum += (width * mask).sum(axis=(0, 1))
        self.interval_sum += (interval * mask).sum(axis=(0, 1))
        variogram, valid_pairs = lag1_temporal_variogram_pointwise(samples, target, mask)
        self.variogram_count += valid_pairs.sum(axis=(0, 1))
        self.variogram_sum += (variogram * valid_pairs).sum(axis=(0, 1))
        energy = masked_dimension_normalized_energy(samples, target, mask)
        self.energy_sum += float(energy.sum())
        self.energy_count += len(energy)

    def finalize(self, *, target_weighted: bool = False) -> dict[str, object]:
        if self.count is None or self.energy_count == 0 or not (self.count > 0).any():
            raise ValueError("no scored trajectories")
        active = self.count > 0
        if not (self.variogram_count[active] > 0).all():
            raise ValueError("every active channel needs at least one lag-1 temporal pair")
        if target_weighted:
            count = self.count[active].sum()
            pair_count = self.variogram_count[active].sum()
            crps = float(self.crps_sum[active].sum() / count)
            first_crps = float(self.crps_first_half_sum[active].sum() / count)
            last_crps = float(self.crps_last_half_sum[active].sum() / count)
            coverage = float(self.coverage_sum[active].sum() / count)
            coverage_first = float(self.coverage_first_half_sum[active].sum() / count)
            coverage_last = float(self.coverage_last_half_sum[active].sum() / count)
            width = float(self.width_sum[active].sum() / count)
            interval = float(self.interval_sum[active].sum() / count)
            variogram = float(self.variogram_sum[active].sum() / pair_count)
        else:
            crps = self.crps_sum[active] / self.count[active]
            first_crps = self.crps_first_half_sum[active] / self.count[active]
            last_crps = self.crps_last_half_sum[active] / self.count[active]
            coverage = self.coverage_sum[active] / self.count[active]
            coverage_first = self.coverage_first_half_sum[active] / self.count[active]
            coverage_last = self.coverage_last_half_sum[active] / self.count[active]
            width = self.width_sum[active] / self.count[active]
            interval = self.interval_sum[active] / self.count[active]
            variogram = self.variogram_sum[active] / self.variogram_count[active]
        result = {
            "aggregation": "target_weighted" if target_weighted else "channel_macro",
            "marginal_crps": float(np.mean(crps)),
            "marginal_crps_first50": float(np.mean(first_crps)),
            "marginal_crps_last50": float(np.mean(last_crps)),
            "mc_half_crps_abs_difference": float(abs(np.mean(first_crps) - np.mean(last_crps))),
            "coverage_95": float(np.mean(coverage)),
            "coverage_abs_error_95": float(np.mean(np.abs(coverage - self.nominal))),
            "coverage_95_first50": float(np.mean(coverage_first)),
            "coverage_95_last50": float(np.mean(coverage_last)),
            "coverage_abs_error_95_first50": float(np.mean(np.abs(coverage_first - self.nominal))),
            "coverage_abs_error_95_last50": float(np.mean(np.abs(coverage_last - self.nominal))),
            "mc_half_coverage_abs_error_difference": float(abs(
                np.mean(np.abs(coverage_first - self.nominal))
                - np.mean(np.abs(coverage_last - self.nominal))
            )),
            "interval_width_95": float(np.mean(width)),
            "interval_score_95": float(np.mean(interval)),
            "trajectory_energy_score": self.energy_sum / self.energy_count,
            "lag1_temporal_variogram_score": float(np.mean(variogram)),
            "active_channel_indices": np.flatnonzero(active).tolist(),
            "target_count": int(self.count[active].sum()),
            "evaluated_window_count": int(self.energy_count),
            "lag1_pair_count": int(self.variogram_count[active].sum()),
        }
        numeric = [value for value in result.values() if isinstance(value, (int, float))]
        if not np.isfinite(numeric).all():
            raise ValueError("nonfinite metric")
        return result


class EqualStratumMetrics:
    def __init__(
        self,
        strata: Sequence[str],
        expected_samples: int = 100,
        nominal_coverage: float = 0.95,
    ) -> None:
        if len(strata) != 4 or len(set(strata)) != 4:
            raise ValueError("exactly four registered strata are required")
        self.strata = tuple(strata)
        self.by_stratum = {
            name: TrajectoryMetrics(expected_samples, nominal_coverage) for name in self.strata
        }
        self.target_weighted = TrajectoryMetrics(expected_samples, nominal_coverage)

    def update(self, name: str, samples: object, target: object, mask: object) -> None:
        if name not in self.by_stratum:
            raise ValueError(f"unregistered stratum {name!r}")
        self.by_stratum[name].update(samples, target, mask)
        self.target_weighted.update(samples, target, mask)

    def finalize(self) -> dict[str, object]:
        strata = {name: accumulator.finalize() for name, accumulator in self.by_stratum.items()}
        metric_names = (
            "marginal_crps",
            "coverage_95",
            "coverage_abs_error_95",
            "interval_width_95",
            "interval_score_95",
            "trajectory_energy_score",
            "lag1_temporal_variogram_score",
        )
        equal = {
            metric: float(np.mean([strata[name][metric] for name in self.strata]))
            for metric in metric_names
        }
        first50 = float(np.mean([strata[name]["marginal_crps_first50"] for name in self.strata]))
        last50 = float(np.mean([strata[name]["marginal_crps_last50"] for name in self.strata]))
        equal["marginal_crps_first50"] = first50
        equal["marginal_crps_last50"] = last50
        equal["mc_half_crps_abs_difference"] = abs(first50 - last50)
        for metric in (
            "coverage_95_first50",
            "coverage_95_last50",
            "coverage_abs_error_95_first50",
            "coverage_abs_error_95_last50",
        ):
            equal[metric] = float(np.mean([strata[name][metric] for name in self.strata]))
        equal["mc_half_coverage_abs_error_difference"] = abs(
            equal["coverage_abs_error_95_first50"] - equal["coverage_abs_error_95_last50"]
        )
        if not np.isfinite(list(equal.values())).all():
            raise ValueError("nonfinite equal-stratum metric")
        return {
            "aggregation": "channel_macro_within_stratum_then_equal_stratum",
            "nominal_coverage": 0.95,
            "trajectory_samples": next(iter(self.by_stratum.values())).expected_samples,
            "quantile_method": QUANTILE_METHOD,
            "energy_normalization": "euclidean_norm_divided_by_sqrt_scored_dimensions",
            "variogram_scope": "lag1_same_channel_temporal_only",
            "variogram_power": VARIOGRAM_POWER,
            "units": "standardized_only",
            "primary_equal_stratum": equal,
            "strata": strata,
            "secondary_target_weighted": self.target_weighted.finalize(target_weighted=True),
        }
