"""Fair finite-ensemble scores and auditable per-window sufficient statistics."""
from __future__ import annotations
import math
import numpy as np


def score_terms(samples, target, mask):
    samples = np.asarray(samples, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask)
    if samples.ndim != 4 or target.ndim != 3 or samples.shape[0] != target.shape[0] or samples.shape[2:] != target.shape[1:] or mask.shape != target.shape:
        raise ValueError('expected samples [B,S,L,C], target/mask [B,L,C]')
    if not np.isfinite(samples).all() or not np.isfinite(target).all() or not np.isin(mask, [0, 1]).all():
        raise ValueError('nonfinite values or nonbinary mask')
    mask = mask.astype(bool)
    size = samples.shape[1]
    if size < 2 or not mask.reshape(len(mask), -1).any(1).all():
        raise ValueError('at least two draws and one scored dimension per window required')
    first = np.abs(samples - target[:, None]).mean(1)
    ordered = np.sort(samples, axis=1)
    weights = (2 * np.arange(size) - size + 1).reshape(1, size, 1, 1)
    half_pair_sum = (ordered * weights).sum(1)
    empirical = first - half_pair_sum / size**2
    fair = first - half_pair_sum / (size * (size - 1))
    energy_empirical, energy_fair = [], []
    for draws, truth, scored in zip(samples, target, mask):
        flat = draws.reshape(size, -1)[:, scored.reshape(-1)]
        truth = truth[scored]
        scale = math.sqrt(len(truth))
        distance = np.linalg.norm(flat - truth, axis=1).mean() / scale
        squares = (flat * flat).sum(1)
        pair = np.sqrt(np.maximum(squares[:, None] + squares[None] - 2 * flat @ flat.T, 0)) / scale
        np.fill_diagonal(pair, 0.0)
        energy_empirical.append(float(distance - 0.5 * pair.sum() / size**2))
        energy_fair.append(float(distance - 0.5 * pair.sum() / (size * (size - 1))))
    lower, upper = np.quantile(samples, [0.025, 0.975], axis=1, method='weibull')
    covered = (target >= lower) & (target <= upper)
    return {
        'channel_count': mask.sum(1).tolist(),
        'empirical_crps_channel_sum': (empirical * mask).sum(1).tolist(),
        'fair_crps_channel_sum': (fair * mask).sum(1).tolist(),
        'coverage_channel_sum': (covered * mask).sum(1).tolist(),
        'width_channel_sum': ((upper - lower) * mask).sum(1).tolist(),
        'empirical_energy': energy_empirical,
        'fair_energy': energy_fair,
    }


def gaussian_terms(mean, scale, target, mask):
    mean, scale, target = (np.asarray(x, dtype=np.float64) for x in (mean, scale, target))
    mask = np.asarray(mask, dtype=bool)
    if mean.shape != target.shape or scale.shape != target.shape or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('invalid Gaussian parameters')
    z = (target - mean) / scale
    erf = np.fromiter((math.erf(float(x) / math.sqrt(2)) for x in z.flat), dtype=float).reshape(z.shape)
    phi = np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    crps = scale * (z * erf + 2 * phi - 1 / math.sqrt(math.pi))
    covered = np.abs(z) <= 1.959963984540054
    return {
        'analytic_gaussian_crps_channel_sum': (crps * mask).sum(1).tolist(),
        'analytic_gaussian_coverage_channel_sum': (covered * mask).sum(1).tolist(),
        'analytic_gaussian_width_channel_sum': (2 * 1.959963984540054 * scale * mask).sum(1).tolist(),
    }


def summarize_rows(rows):
    """Same channel-within-stratum weights; block summaries remain descriptive."""
    summaries = {}
    for stratum in sorted({r['stratum'] for r in rows}):
        selected = [r for r in rows if r['stratum'] == stratum]
        counts = np.asarray([r['channel_count'] for r in selected]).sum(0)
        active = counts > 0
        result = {'windows': len(selected), 'channel_count': counts.tolist(), 'source_blocks': len({r['source_block_digest'] for r in selected})}
        for name in selected[0]:
            if name.endswith('_channel_sum'):
                sums = np.asarray([r[name] for r in selected]).sum(0)
                result[name.removesuffix('_channel_sum')] = float(np.mean(sums[active] / counts[active]))
        for name in ('empirical_energy', 'fair_energy', 'support_fraction'):
            result[name] = float(np.mean([r[name] for r in selected]))
        summaries[stratum] = result
    equal = {name: float(np.mean([s[name] for s in summaries.values()])) for name in next(iter(summaries.values())) if name not in ('windows', 'channel_count', 'source_blocks')}
    return {'strata': summaries, 'equal_stratum': equal, 'fair_assumption': 'independent exchangeable draws conditional on this fitted model and input; training seeds are not predictive draws', 'scope': 'retrospective development diagnostic, not independent confirmation'}


def self_test():
    rng = np.random.default_rng(51)
    samples = rng.normal(size=(3, 5, 4, 2))
    target = rng.normal(size=(3, 4, 2))
    mask = rng.random((3, 4, 2)) > 0.2
    terms = score_terms(samples, target, mask)
    pair = np.abs(samples[:, :, None] - samples[:, None, :]).sum((1, 2))
    exact = np.abs(samples - target[:, None]).mean(1) - pair / (2 * 5 * 4)
    np.testing.assert_allclose(terms['fair_crps_channel_sum'], (exact * mask).sum(1), rtol=1e-12, atol=1e-12)
    for i in range(3):
        x = samples[i][:, mask[i]]
        first = np.linalg.norm(x - target[i][mask[i]], axis=1).mean()
        second = np.linalg.norm(x[:, None] - x[None, :], axis=-1).sum()
        expected = (first - second / (2 * 5 * 4)) / math.sqrt(mask[i].sum())
        np.testing.assert_allclose(terms['fair_energy'][i], expected, atol=1e-12)
    g = gaussian_terms(np.zeros((1, 1, 1)), np.ones((1, 1, 1)), np.zeros((1, 1, 1)), np.ones((1, 1, 1)))
    np.testing.assert_allclose(g['analytic_gaussian_crps_channel_sum'], [[(math.sqrt(2) - 1) / math.sqrt(math.pi)]], atol=1e-12)
    print('PASS: fair CRPS/ES brute-force identities and analytic Gaussian origin')


if __name__ == '__main__':
    self_test()
