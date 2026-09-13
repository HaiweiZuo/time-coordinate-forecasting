"""Offline v2 supplement checks, including complete original and new histories."""
import argparse
import math
from pathlib import Path
import struct
import verify_numerical_release as base


def check_history(history, summary, policy):
    base.require([r['epoch'] for r in history] == list(range(len(history))), 'Noncontiguous history')
    best, epoch, stale = math.inf, None, 0
    for index, row in enumerate(history):
        base.require(set(row) == {'epoch', 'fit_loss', 'early_stop_loss'}, 'Unexpected history fields')
        base.require(math.isfinite(row['fit_loss']) and math.isfinite(row['early_stop_loss']), 'Nonfinite history')
        if row['early_stop_loss'] < best:
            best, epoch, stale = row['early_stop_loss'], index, 0
        else:
            stale += 1
        if index < len(history)-1:
            base.require(stale < policy['patience'], 'History continues beyond its stop')
    reason = 'epoch_cap' if len(history) == policy['max_epochs'] else 'patience'
    base.require(len(history) <= policy['max_epochs'] and (stale == policy['patience'] or reason == 'epoch_cap'), 'Incomplete stop')
    base.require(summary['completed_epochs'] == len(history) and summary['selected_epoch_zero_based'] == epoch,
                 'Summary epoch differs')
    base.require(summary['selected_epoch_one_based'] == epoch+1 and summary['stale_epochs'] == stale
                 and summary['stop_reason'] == reason, 'Summary stop differs')
    base.close(summary['selected_early_stop_loss'], best)
    base.close(summary['last_fit_loss'], history[-1]['fit_loss'])
    base.close(summary['last_early_stop_loss'], history[-1]['early_stop_loss'])
    return epoch, best


def verify(package):
    report = base.verify(package)
    contract = base.read(package / 'original_scientific_contract.json')
    old = base.read(package / 'original_24_aggregates.json')['cells']
    crossed = base.read(package / 'crossed_48_aggregates.json')
    new_export = base.read(package / 'new_24_training_histories.json')
    new = {c['cell_id']: c for c in new_export['cells']}
    train = {c['cell_id']: c for c in crossed['cells'] if c['mode'] == 'train'}
    base.require(set(new) == set(train) and len(new) == 24 and len(new_export['cells']) == 24, 'Incorrect new history roster')
    for cell in old:
        check_history(cell['history'], cell['training_summary'], contract['training'])
    old_by_dataset = {c['cell_id'].split('__')[0]: c for c in old}
    for name, cell in new.items():
        summary = train[name]['training']
        base.require(cell['training_summary'] == summary, 'New summary mismatch')
        base.require(cell['history_sha256'] == crossed['inputs'][name]['file_sha256']['history.json'], 'New history source hash link')
        base.require(cell['terminal_sha256'] == crossed['inputs'][name]['terminal_sha256'], 'New terminal hash link')
        epoch, loss = check_history(cell['history'], summary, contract['training'])
        selected = train[name]['selected_checkpoint']
        base.require(selected['mode'] == 'train' and selected['history_rule_verified'] is True
                     and selected['epoch_zero_based'] == epoch, 'Selected checkpoint identity/rule mismatch')
        base.close(selected['early_stop_loss'], loss)
        fit_counts = old_by_dataset[cell['dataset']]['partition_proof']['partition_counts']['fit']
        updates = math.ceil(sum(fit_counts.values()) / contract['training']['batch_size']) * len(cell['history'])
        base.require(updates == summary['optimizer_steps_executed'], 'Optimizer update derivation differs')
    old_rows = sum(len(c['history']) for c in old)
    new_rows = sum(len(c['history']) for c in new.values())
    base.require(old_rows == 1518 and new_rows == new_export['total_epoch_rows'] == 1467, 'Old/new history row count differs')
    captions = base.read(package / 'captions.json')
    expected = {f'training_{dataset}__{family}' for dataset in ('ushcn', 'weather_jena2020')
                for family in ('conditional_diffusion', 'direct_gaussian')}
    base.require(set(captions) == expected, 'Missing original training caption')
    for stem in expected:
        base.require('without smoothing' in captions[stem], 'Original caption scope changed')
        image = (package / (stem+'.png')).read_bytes()
        base.require(image[:8] == b'\x89PNG\r\n\x1a\n' and struct.unpack('>II', image[16:24]) == (2100, 900), 'PNG dimensions differ')
        base.require((package / (stem+'.svg')).is_file(), 'Missing SVG')
    rows = [line for line in (package / 'original_training_history_table.md').read_text().splitlines()
            if line.startswith('| ushcn__') or line.startswith('| weather_jena2020__')]
    lookup = {c['cell_id']: c['training_summary'] for c in old}
    base.require(len(rows) == 24, 'Original history table row count')
    seen = set()
    for row in rows:
        fields = [c.strip() for c in row.strip('|').split('|')]
        name, completed, best, early, last, steps, reason = fields
        s = lookup[name]
        base.require(name not in seen, 'Duplicate history table row')
        seen.add(name)
        base.require((int(completed), int(best), int(steps)) == (s['completed_epochs'], s['selected_epoch_one_based'], s['optimizer_steps_executed']), 'Table integer mismatch')
        base.require(early == format(s['selected_early_stop_loss'], '.8g') and last == format(s['last_early_stop_loss'], '.8g'), 'Table displayed loss mismatch')
        base.require(reason == ('epoch_cap' if s['stop_reason'] == 'epoch_cap' else 'early_stopping_patience'), 'Table stop mismatch')
    return {**report, 'status': 'PASS_V2_AGGREGATES_ALL_ORIGINAL_AND_NEW_HISTORIES', 'original_epoch_rows': old_rows,
            'new_epoch_rows': new_rows, 'original_training_curve_pairs': 4, 'original_full_cell_table_rows': 24,
            'new_histories_verified': 24, 'new_stop_reason': 'patience for all 24', 'histories_not_new_trajectories': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parent)
    print(base.json.dumps(verify(parser.parse_args().package), indent=2))
