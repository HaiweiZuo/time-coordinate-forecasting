#!/usr/bin/env python
"""Synthetic checks of the second-protocol code. No study data, no GPU, no training."""

from __future__ import annotations

import os
import tempfile
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from confirmation.common import CONFIG

import numpy as np
import torch

from confirmation import export, rosters
from confirmation.prepare_data import read_year_archives
from confirmation.score import stratum_summary
from confirmation2 import analyze
from confirmation2.common import ARMS, DATASET, FAMILY, RECORD_ID, ROSTER, SEEDS, build_roster
from confirmation2.orchestrate import ordered_cells
from confirmation2.prepare_data import read_archives
from confirmation2.validate import merge, same

COLUMNS = [f"c{index}" for index in range(21)]


def write_archive(path: Path, start: datetime, rows: int, generator, drop=(), duplicate=()) -> None:
    lines = ["Date Time," + ",".join(COLUMNS)]
    for index in range(rows):
        if index in drop:
            continue
        stamp = (start + index * timedelta(minutes=10)).strftime("%d.%m.%Y %H:%M:%S")
        for _ in range(2 if index in duplicate else 1):
            values = generator.normal(size=21).round(2)
            values[generator.random(21) < 0.02] = -9999.0
            lines.append(stamp + "," + ",".join(f"{value:.2f}" for value in values))
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr(path.stem + ".csv", ("\r\n".join(lines) + "\r\n").encode("latin-1"))


def check_matrix() -> None:
    cells = ordered_cells()
    assert len(cells) == 40 and len(set(cells)) == 40
    assert all(arm in ("LL", "SL") for arm, _ in cells[:20]) and all(arm in ("LS", "SS") for arm, _ in cells[20:])
    assert {seed for _, seed in cells} == set(range(2037, 2047)) and not set(SEEDS) & set(range(2024, 2037))
    print("PASS matrix: 40 registered cells, LL and SL first, ten new seeds")


def check_reader() -> None:
    generator = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        a, b = root / "mpi_roof_2023a.zip", root / "mpi_roof_2023b.zip"
        write_archive(a, datetime(2023, 1, 1, 0, 10), 400, generator, drop={17, 18}, duplicate={40})
        write_archive(b, datetime(2023, 1, 1, 0, 10) + 400 * timedelta(minutes=10), 300, generator, drop={5}, duplicate={7, 8})
        reference, reference_metadata = read_year_archives([a, b], COLUMNS, 2023)
        pair, pair_metadata = read_archives([a, b], COLUMNS, 2023)
        assert same(pair, reference) and pair_metadata == reference_metadata
        merged = root / "merged" / "mpi_roof_2023.zip"
        merged.parent.mkdir()
        merge([a, b], merged)
        single, single_metadata = read_archives([merged], COLUMNS, 2023)
        assert same(single, reference)
        assert all(single_metadata[key] == reference_metadata[key] for key in reference_metadata if key != "archives")
        assert reference_metadata["absent_timestamp_rows"] == 3 and reference_metadata["duplicate_extra_rows"] == 3
        outside = root / "mpi_roof_2022.zip"
        write_archive(outside, datetime(2023, 1, 1, 0, 10), 10, generator)
        try:
            read_archives([outside], COLUMNS, 2022)
            raise AssertionError("a date outside the archive year must be refused")
        except ValueError:
            pass
    print("PASS reader: same tensors and counts as the two-archive reader; a merged single archive gives the same record; other years refused")


def check_roster() -> None:
    length = 792 * 3 + 100
    values = torch.arange(length * 21, dtype=torch.float32).reshape(length, 21) / 100
    mask = torch.ones_like(values, dtype=torch.bool)
    mask[792 + 90:792 + 108] = False
    payload = RECORD_ID, torch.arange(length, dtype=torch.float32), values, mask
    metadata = {"lattice_rows": length, "first_timestamp": "2024-01-01T00:10:00"}
    arms = {arm: build_roster(payload, metadata, arm) for arm in CONFIG["coordinate_arms"]}
    rosters.paired_across_arms(arms)
    proof = arms["LL"].proof
    assert len(arms["LL"].dev_score) == 8 and proof["excluded_block_count"] == 2 and proof["source_block_count"]["score"] == 2
    assert proof["record_id"] == RECORD_ID and set(proof["calendar_month_block_counts"]) == {"2024-01"}
    try:
        build_roster(("weather:temporal2023", *payload[1:]), metadata, "LL")
        raise AssertionError("a record of another year must be refused")
    except ValueError:
        pass
    print("PASS roster: 2022 temporal rule on the 2024 record identity, all-or-none blocks, paired across arms")


def synthetic_delivery(gain, flip=()):
    """Delivery tables with a built-in arm effect; seeds in `flip` reverse the SL-versus-LL effect."""
    generator = np.random.default_rng(11)
    strata = ("a", "b", "c", "d")
    counts = generator.integers(0, 6, size=(12, 4, 3)) + (np.arange(3) == 0)
    base = generator.random((12, 4, 3)) + 0.5
    scores, blocks, sums = [], [], []
    for arm in ARMS:
        for seed in SEEDS:
            factor = (2 - gain[arm]) if (arm == "SL" and seed in flip) else gain[arm]
            noise = 1 + 0.01 * generator.standard_normal((12, 4, 3))
            rows = [{"stratum": strata[s], "source_block_digest": f"{b:016x}" + "0" * 48, "channel_count": counts[b, s].tolist(),
                     "empirical_crps_channel_sum": (counts[b, s] * base[b, s] * noise[b, s] * factor).tolist(),
                     "fair_crps_channel_sum": (counts[b, s] * base[b, s] * noise[b, s] * factor * 0.98).tolist()}
                    for b in range(12) for s in range(4)]
            cell = {"dataset": DATASET, "roster": ROSTER, "family": FAMILY, "arm": arm, "seed": seed}
            scores.extend({**cell, **row} for row in export.summary_rows(stratum_summary(rows)))
            block_scores, block_sums = export.block_tables(rows)
            blocks.extend({**cell, **row} for row in block_scores)
            sums.extend({**cell, **row} for row in block_sums)
    reference = [{"dataset": DATASET, "roster": ROSTER, "baseline": name, "stratum": "ALL", "crps_empirical": value, "crps_fair": value, "n_scored": 1}
                 for name, value in (("climatology", 2.0), ("persistence", 1.5))]
    return scores, blocks, sums, reference


def run_analysis(tables):
    scores, blocks, sums, reference = tables
    first = analyze.first
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        export.write_csv(out / "confirmation_scores.csv", export.CELL_FIELDS + export.SCORE_FIELDS, scores)
        export.write_csv(out / "confirmation_block_scores.csv", export.CELL_FIELDS + ["stratum", "source_block", "crps_empirical", "crps_fair", "n_scored", "n_windows"], blocks)
        export.write_csv(out / "confirmation_block_channel_sums.csv.gz", export.CELL_FIELDS + ["stratum", "source_block", "channel", "count", "empirical_sum", "fair_sum"], sums, compressed=True)
        export.write_csv(out / "baseline_scores.csv", ["dataset", "roster", "baseline"] + export.SCORE_FIELDS, reference)
        loaded = [first.read_csv(out / name) for name in ("confirmation_scores.csv", "confirmation_block_scores.csv", "confirmation_block_channel_sums.csv.gz")]
        results = analyze.analyse(*loaded, "empirical")
        arms = first.arm_table(loaded[0], first.read_csv(out / "baseline_scores.csv"))
    return results, arms


def check_analysis() -> None:
    gain = {"LL": 1.0, "SL": 0.9, "LS": 0.97, "SS": 1.04}
    results, arms = run_analysis(synthetic_delivery(gain))
    assert len(results) == 5 and len(arms) == 4 and all(row["n"] == 10 for row in arms)
    for entry in results:
        a, b = entry["contrast"].split(" vs ")
        assert entry["roster"] == ROSTER and entry["family"] == FAMILY and entry["seeds"] == list(SEEDS)
        assert abs(entry["median"] - (1 - gain[a] / gain[b])) < 0.02
    decisions = [entry for entry in results if "decision" in entry]
    assert len(decisions) == 1 and decisions[0]["contrast"] == "SL vs LL" and decisions[0]["label"].startswith("H3")
    assert decisions[0]["positive"] == 10 and decisions[0]["decision"] == "confirmed"
    # Nine positive seeds pass, eight do not.
    nine, _ = run_analysis(synthetic_delivery(gain, flip={2040}))
    eight, _ = run_analysis(synthetic_delivery(gain, flip={2040, 2045}))
    pick = lambda rows: next(entry for entry in rows if "decision" in entry)
    assert pick(nine)["positive"] == 9 and pick(nine)["decision"] == "confirmed"
    assert pick(eight)["positive"] == 8 and pick(eight)["decision"] == "not confirmed"
    text = analyze.markdown(eight, arms, [])
    assert "H3 decision by the registered rule: not confirmed" in text and "H2" not in text
    print("PASS analysis: first-protocol statistics on the second-protocol matrix; H3 decided at 9 of 10, not at 8")


if __name__ == "__main__":
    check_matrix()
    check_reader()
    check_roster()
    check_analysis()
