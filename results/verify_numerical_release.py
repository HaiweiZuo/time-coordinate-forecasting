"""Verify this aggregate-only release using Python's standard library.

No models, private files, downloads, raw targets, or scientific reruns are used.
Run from any directory: python verify_numerical_release.py --package <directory>.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median

ARMS = ('LL', 'SL', 'LS', 'SS')
KEYS = ('E_at_R_L', 'E_at_R_S', 'R_at_E_L', 'R_at_E_S', 'interaction')
PRIMARY = 'primary_empirical_crps_archived_corners'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def close(a, b):
    require(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), 'Numeric mismatch')


def differences(values):
    ll, sl, ls, ss = (values[k] for k in ARMS)
    return dict(zip(KEYS, (sl-ll, ss-ls, ls-ll, ss-sl, ss-sl-ls+ll)))


def verify_matrix(data, temporal):
    groups = defaultdict(dict)
    for cell in data['cells']:
        key = (cell['family'], cell['seed']) if temporal else (cell['dataset'], cell['family'], cell['seed'])
        require(cell['arm'] not in groups[key], 'Duplicate arm')
        groups[key][cell['arm']] = cell
    require(len(data['cells']) == (24 if temporal else 48), 'Incorrect cell count')
    require(all(set(arms) == set(ARMS) for arms in groups.values()), 'Missing arm')
    pooled = defaultdict(list)
    effects = data['per_seed_effects' if temporal else 'per_seed_factorial_effects']
    for row in effects:
        key = (row['family'], row['seed']) if temporal else (row['dataset'], row['family'], row['seed'])
        arms = groups[key]
        values = {}
        for arm, cell in arms.items():
            if row['metric'] == PRIMARY:
                values[arm] = cell['primary_2x2_empirical_crps'][row['stratum']]
            else:
                scope = row.get('scope', 'overall')
                summary = cell['summary'] if scope == 'overall' else cell['months' if scope == 'month' else 'blocks'][row['group']]
                stratum = summary['equal_stratum'] if row['stratum'] == 'equal_stratum' else summary['strata'][row['stratum']]
                values[arm] = stratum[row['metric']]
            close(values[arm], row['absolute_by_arm'][arm])
        for name, value in differences(values).items():
            close(value, row[name])
        if temporal:
            close(values['SS']-values['LL'], row['diagonal_SS_minus_LL'])
            close(100*(1-values['SS']/values['LL']), row['diagonal_relative_improvement_percent'])
        skey = (row['family'], row['scope'], row['group'], row['stratum'], row['metric']) if temporal else (row['dataset'], row['family'], row['stratum'], row['metric'])
        pooled[skey].append(row)
    summaries = data['three_seed_median_ranges' if temporal else 'three_seed_descriptive_summaries']
    for summary in summaries:
        skey = (summary['family'], summary['scope'], summary['group'], summary['stratum'], summary['metric']) if temporal else (summary['dataset'], summary['family'], summary['stratum'], summary['metric'])
        rows = pooled[skey]
        require(sorted(r['seed'] for r in rows) == [2024, 2025, 2026], 'Wrong seeds')
        metrics = (*KEYS, 'diagonal_SS_minus_LL', 'diagonal_relative_improvement_percent') if temporal else KEYS
        for name in metrics:
            values = [r[name] for r in rows]
            reported = summary[name] if temporal else summary['effects'][name]
            for label, fn in (('median', median), ('min', min), ('max', max)):
                close(fn(values), reported[label])
        for arm in ARMS:
            for label, fn in (('median', median), ('min', min), ('max', max)):
                close(fn([r['absolute_by_arm'][arm] for r in rows]), summary['absolute_by_arm'][arm][label])
    return len(effects), len(summaries)


def verify(package):
    inventory = read(package / 'MANIFEST.json')['files']
    require({p.name for p in package.iterdir() if p.is_file()} == set(inventory) | {'MANIFEST.json'}, 'Missing/unregistered package file')
    for name, record in inventory.items():
        require(Path(name).name == name, 'Unsafe package filename')
        path = package / name
        require(not path.is_symlink() and sha(path) == record['sha256'] and path.stat().st_size == record['bytes'], 'Package hash mismatch: ' + name)
    original = read(package / 'original_24_aggregates.json')
    cross = read(package / 'crossed_48_aggregates.json')
    temporal = read(package / 'temporal_24_aggregates.json')
    numeric = read(package / 'support_mask_training_summary.json')
    old = {c['cell_id']: c for c in original['cells']}
    require(len(old) == 24 and len(original['cells']) == 24, 'Original roster mismatch')
    history_rows = 0
    for cell in old.values():
        history = cell['history']
        require([r['epoch'] for r in history] == list(range(len(history))), 'History epoch order')
        selected = min(range(len(history)), key=lambda i: history[i]['early_stop_loss'])
        require(selected == cell['training_summary']['selected_epoch_zero_based'], 'Original history selection mismatch')
        close(history[selected]['early_stop_loss'], cell['training_summary']['selected_early_stop_loss'])
        history_rows += len(history)
    for cell in cross['cells']:
        if cell['mode'] == 'replay':
            original_cell = old[cell['baseline_cell_id']]
            close(cell['primary_2x2_empirical_crps']['equal_stratum'], original_cell['metrics']['primary_equal_stratum']['marginal_crps'])
            for stratum in cell['summary']['strata']:
                close(cell['primary_2x2_empirical_crps'][stratum], original_cell['metrics']['strata'][stratum]['marginal_crps'])
    crossed_counts = verify_matrix(cross, False)
    temporal_counts = verify_matrix(temporal, True)
    for row in numeric['support']:
        close(row['support_fraction']*row['grid_size'], row['mean_supported_positions_per_window'])
        close(row['support_fraction']*row['all_window_grid_positions'], row['supported_window_grid_positions'])
    new = [c for c in cross['cells'] if c['mode'] == 'train']
    require(len(new) == 24, 'New-fit roster mismatch')
    total = sum(c['training']['optimizer_steps_executed'] for c in new)
    require(total == numeric['new_training']['total_optimizer_updates'], 'New-only update mismatch')
    sensitivity = defaultdict(list)
    for d, is_temporal in ((cross, False), (temporal, True)):
        groups = defaultdict(dict)
        for c in d['cells']:
            groups[('weather2022' if is_temporal else c['dataset'], c['family'], c['seed'])][c['arm']] = c
        for (dataset, family, seed), arms in groups.items():
            for metric in ('empirical_crps', 'fair_crps', 'analytic_gaussian_crps', 'empirical_energy', 'fair_energy'):
                if metric in arms['LL']['summary']['equal_stratum']:
                    ll, ss = [arms[a]['summary']['equal_stratum'][metric] for a in ('LL', 'SS')]
                    sensitivity[dataset, family, metric].append(100*(1-ss/ll))
            if not is_temporal:
                ll, ss = [arms[a]['primary_2x2_empirical_crps']['equal_stratum'] for a in ('LL', 'SS')]
                sensitivity[dataset, family, 'archived_empirical_crps'].append(100*(1-ss/ll))
    for row in numeric['sensitivity_summaries']:
        values = sensitivity[row['dataset'], row['family'], row['metric']]
        for name, fn in (('median', median), ('min', min), ('max', max)):
            close(fn(values), row[name])
    return {'status': 'PASS_AGGREGATE_ARITHMETIC_AND_PACKAGE_HASHES', 'original_cells': 24,
            'original_history_rows': history_rows, 'crossed_effect_rows_and_summaries': crossed_counts,
            'temporal_effect_rows_and_summaries': temporal_counts, 'new_optimizer_updates': total,
            'boundary': 'No raw-target scoring, training, external-file hash verification, or full experimental reproduction.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parent)
    print(json.dumps(verify(parser.parse_args().package), indent=2))
