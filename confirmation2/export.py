#!/usr/bin/env python
"""Collect the delivery files of the second protocol (section 8) from finished runs.

Reads only the files written by confirmation2/fit.py, score.py and
baselines.py. No model is loaded and nothing is rescored. Tables and file
formats are those of confirmation/export.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from confirmation.common import cell_name, read_json, require, write_json
from confirmation.export import CELL_FIELDS, SCORE_FIELDS, block_tables, summary_rows, write_csv
from confirmation2.common import ARMS, DATASET, FAMILY, METADATA_FILE, PROTOCOL_SHA256, ROSTER, SEEDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="run directory used by confirmation2/orchestrate.py")
    parser.add_argument("--baselines", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--record", required=True, type=Path, help="JSON with hardware, software, timing and deviation notes")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    scores, blocks, sums, missing, fits = [], [], [], [], []
    for arm in ARMS:
        for seed in SEEDS:
            name = cell_name(DATASET, FAMILY, arm, seed)
            terminal_path = args.root / "fits" / name / "terminal.json"
            if not terminal_path.exists():
                missing.append({"cell": name, "stage": "fit"})
                continue
            terminal = read_json(terminal_path)
            fits.append({"cell": name, **{key: terminal[key] for key in ("epochs_run", "selected_epoch", "stop_reason", "checkpoint_sha256", "elapsed_seconds")}})
            directory = args.root / "scores" / name / ROSTER
            if not (directory / "summary.json").exists():
                missing.append({"cell": name, "stage": f"score:{ROSTER}"})
                continue
            summary = read_json(directory / "summary.json")
            require(summary["protocol_sha256"] == PROTOCOL_SHA256 and summary["checkpoint_sha256"] == terminal["checkpoint_sha256"],
                    f"score of {name} does not belong to its fitted checkpoint")
            cell = {"dataset": DATASET, "roster": ROSTER, "family": FAMILY, "arm": arm, "seed": seed}
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
    events = [json.loads(line) for line in (args.root / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    preparation = read_json(args.prepared / "PREPARATION.json")
    preparation["metadata_counts"] = {key: value for key, value in read_json(args.prepared / METADATA_FILE).items()
                                      if key in ("unique_timestamps", "sentinel_entries_before_averaging", "missing_entries_after_lattice", "off_lattice_unique_timestamps")}
    record = read_json(args.record)
    record.update({
        "protocol_sha256": PROTOCOL_SHA256, "preparation": preparation,
        "fits_completed": len(fits), "fits_registered": len(ARMS) * len(SEEDS), "missing": missing, "fits": fits,
        "reruns": [event for event in events if event["event"] in ("rerun", "gave_up") or (event["event"] == "process_end" and event["returncode"] != 0)],
        "first_event_unix": events[0]["time_unix"] if events else None, "last_event_unix": events[-1]["time_unix"] if events else None,
    })
    write_json(args.out / "RUN_RECORD.json", record)
    print(f"Exported {len(fits)} fits, {len(scores)} score rows, {len(missing)} missing entries.")


if __name__ == "__main__":
    main()
