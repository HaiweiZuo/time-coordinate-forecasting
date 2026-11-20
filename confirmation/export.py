#!/usr/bin/env python
"""Collect the delivery files of protocol section 8 from finished runs.

Reads only the files written by confirmation/fit.py, score.py and
baselines.py. No model is loaded and nothing is rescored.

Files written:
  confirmation_scores.csv         per cell, roster and stratum, plus the
                                  equal-stratum primary score (stratum = ALL)
  confirmation_block_scores.csv   the same scores per source-time block
  confirmation_block_channel_sums.csv.gz
                                  per block, stratum and channel: scored
                                  count and CRPS sums (exact re-aggregation)
  baseline_scores.csv, baseline_block_scores.csv
  RUN_RECORD.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
from pathlib import Path

import numpy as np

from confirmation.common import (
    ARMS, CONFIRMATION_SEEDS, DATASETS, FAMILIES, PROTOCOL_SHA256, ROSTERS, cell_name, read_json, require, write_json,
)

CELL_FIELDS = ["dataset", "roster", "family", "arm", "seed"]
SCORE_FIELDS = ["stratum", "crps_empirical", "crps_fair", "n_scored"]


def block_tables(rows):
    """Channel-macro score within every (stratum, block), and the sums behind it."""
    grouped = {}
    for row in rows:
        entry = grouped.setdefault((row["stratum"], row["source_block_digest"]), {"windows": 0, "count": 0, "empirical": 0, "fair": 0})
        entry["windows"] += 1
        entry["count"] = entry["count"] + np.asarray(row["channel_count"], dtype=np.float64)
        entry["empirical"] = entry["empirical"] + np.asarray(row["empirical_crps_channel_sum"], dtype=np.float64)
        entry["fair"] = entry["fair"] + np.asarray(row["fair_crps_channel_sum"], dtype=np.float64)
    scores, sums = [], []
    for (stratum, block), entry in sorted(grouped.items()):
        active = entry["count"] > 0
        scores.append({"stratum": stratum, "source_block": block[:16],
                       "crps_empirical": float(np.mean(entry["empirical"][active] / entry["count"][active])),
                       "crps_fair": float(np.mean(entry["fair"][active] / entry["count"][active])),
                       "n_scored": int(entry["count"].sum()), "n_windows": entry["windows"]})
        for channel in np.flatnonzero(active):
            sums.append({"stratum": stratum, "source_block": block[:16], "channel": int(channel),
                         "count": int(entry["count"][channel]), "empirical_sum": float(entry["empirical"][channel]),
                         "fair_sum": float(entry["fair"][channel])})
    return scores, sums


def summary_rows(summary) -> list[dict]:
    rows = [{"stratum": name, **{key: entry[key] for key in ("crps_empirical", "crps_fair", "n_scored")}}
            for name, entry in summary["strata"].items()]
    rows.append({"stratum": "ALL", **summary["equal_stratum"]})
    return rows


def write_csv(path: Path, fields, rows, compressed: bool = False) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    data = buffer.getvalue().encode("utf-8")
    (gzip.open(path, "wb") if compressed else path.open("wb")).write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="run directory used by confirmation/orchestrate.py")
    parser.add_argument("--baselines", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--record", required=True, type=Path, help="JSON with hardware, software, timing and deviation notes")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    scores, blocks, sums, missing, fits = [], [], [], [], []
    for dataset in DATASETS:
        for family in FAMILIES:
            for arm in ARMS:
                for seed in CONFIRMATION_SEEDS:
                    name = cell_name(dataset, family, arm, seed)
                    terminal_path = args.root / "fits" / name / "terminal.json"
                    if not terminal_path.exists():
                        missing.append({"cell": name, "stage": "fit"})
                        continue
                    terminal = read_json(terminal_path)
                    fits.append({"cell": name, **{key: terminal[key] for key in ("epochs_run", "selected_epoch", "stop_reason", "checkpoint_sha256", "elapsed_seconds")}})
                    for roster in ROSTERS[dataset]:
                        directory = args.root / "scores" / name / roster
                        if not (directory / "summary.json").exists():
                            missing.append({"cell": name, "stage": f"score:{roster}"})
                            continue
                        summary = read_json(directory / "summary.json")
                        require(summary["protocol_sha256"] == PROTOCOL_SHA256 and summary["checkpoint_sha256"] == terminal["checkpoint_sha256"],
                                f"score of {name} does not belong to its fitted checkpoint")
                        cell = {"dataset": dataset, "roster": roster, "family": family, "arm": arm, "seed": seed}
                        for row in summary_rows(summary):
                            scores.append({**cell, **row, "selected_epoch": terminal["selected_epoch"], "stop_reason": terminal["stop_reason"]})
                        block_scores, block_sums = block_tables(read_json(directory / "window_diagnostics.json")["rows"])
                        blocks.extend({**cell, **row} for row in block_scores)
                        sums.extend({**cell, **row} for row in block_sums)
    write_csv(args.out / "confirmation_scores.csv", CELL_FIELDS + SCORE_FIELDS + ["selected_epoch", "stop_reason"], scores)
    write_csv(args.out / "confirmation_block_scores.csv", CELL_FIELDS + ["stratum", "source_block", "crps_empirical", "crps_fair", "n_scored", "n_windows"], blocks)
    write_csv(args.out / "confirmation_block_channel_sums.csv.gz",
              CELL_FIELDS + ["stratum", "source_block", "channel", "count", "empirical_sum", "fair_sum"], sums, compressed=True)
    baseline_scores, baseline_blocks = [], []
    for path in sorted(args.baselines.glob("*.json")):
        result = read_json(path)
        identity = {"dataset": result["dataset"], "roster": result["roster"], "baseline": result["baseline"]}
        baseline_scores.extend({**identity, **row} for row in summary_rows(result))
        baseline_blocks.extend({**identity, **row} for row in block_tables(result["rows"])[0])
    write_csv(args.out / "baseline_scores.csv", ["dataset", "roster", "baseline"] + SCORE_FIELDS, baseline_scores)
    write_csv(args.out / "baseline_block_scores.csv", ["dataset", "roster", "baseline", "stratum", "source_block", "crps_empirical", "crps_fair", "n_scored", "n_windows"], baseline_blocks)
    events = [__import__("json").loads(line) for line in (args.root / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    record = read_json(args.record)
    record.update({
        "protocol_sha256": PROTOCOL_SHA256,
        "preparation": read_json(args.prepared / "PREPARATION.json"),
        "fits_completed": len(fits), "fits_registered": len(DATASETS) * len(FAMILIES) * len(ARMS) * len(CONFIRMATION_SEEDS),
        "missing": missing, "fits": fits,
        "reruns": [event for event in events if event["event"] in ("rerun", "gave_up") or (event["event"] == "process_end" and event["returncode"] != 0)],
        "first_event_unix": events[0]["time_unix"] if events else None, "last_event_unix": events[-1]["time_unix"] if events else None,
    })
    write_json(args.out / "RUN_RECORD.json", record)
    print(f"Exported {len(fits)} fits, {len(scores)} score rows, {len(missing)} missing entries.")


if __name__ == "__main__":
    main()
