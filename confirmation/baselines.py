#!/usr/bin/env python
"""Reference baselines without training (protocol section 6). CPU only.

Same rosters, target masks and aggregation as the model scores.

* Climatology: for each channel, 100 values are drawn once (NumPy
  default_rng(271828), channels in order, with replacement) from the observed
  standardized training-role values of that channel. The same 100 values form
  the predictive ensemble for every target of that channel.
* Persistence: the last observed value of each channel in the history window
  as presented to the model (for Weather, after the fixed history deletion).
  A channel without an observation in the window uses 0, the training mean on
  the standardized scale. The CRPS of a point forecast is its absolute error,
  so the empirical and fair columns hold the same number.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from confirmation.common import (
    CONFIG, PROTOCOL_SHA256, ROSTER_FILES, ROSTERS, read_json, require, require_protocol, sha256, write_json,
)

import numpy as np
import torch

from confirmation.rosters import build_roster
from confirmation.score import stratum_summary
from diagnostic_metrics import score_terms
from geometry_screen import data

CLIMATOLOGY_SEED = 271828
DRAWS = 100


def climatology_ensemble(train_payload, dataset: str) -> np.ndarray:
    records = data.validate_payload(train_payload, dataset)
    channels = records[0].values.shape[1]
    generator = np.random.default_rng(CLIMATOLOGY_SEED)
    ensemble = np.zeros((DRAWS, channels), dtype=np.float64)
    for channel in range(channels):
        observed = np.concatenate([record.values[:, channel][record.mask[:, channel]].numpy() for record in records])
        require(len(observed) > 0, "every channel needs observed training values")
        ensemble[:, channel] = generator.choice(observed, size=DRAWS, replace=True)
    return ensemble


def window_arrays(windows):
    target = np.stack([sample["y"].numpy() for sample in windows.samples]).astype(np.float64)
    mask = np.stack([sample["y_mask"].numpy() for sample in windows.samples])
    return target, mask


def identity(sample) -> dict:
    return {"sample_id": sample["sample_id"], "anchor_digest": sample["anchor_digest"],
            "source_block_digest": sample["source_block_digest"], "stratum": sample["stratum"]}


def climatology_rows(windows, ensemble: np.ndarray) -> list[dict]:
    target, mask = window_arrays(windows)
    rows = []
    for start in range(0, len(target), 64):
        chunk = slice(start, start + 64)
        samples = np.broadcast_to(ensemble[None, :, None, :], (len(target[chunk]), DRAWS, target.shape[1], target.shape[2]))
        terms = score_terms(samples, target[chunk], mask[chunk])
        for offset, sample in enumerate(windows.samples[chunk]):
            rows.append({**identity(sample), "channel_count": terms["channel_count"][offset],
                         "empirical_crps_channel_sum": terms["empirical_crps_channel_sum"][offset],
                         "fair_crps_channel_sum": terms["fair_crps_channel_sum"][offset]})
    return rows


def persistence_rows(windows) -> list[dict]:
    rows = []
    for sample in windows.samples:
        history, observed = sample["x"].numpy().astype(np.float64), sample["x_mask"].numpy().astype(bool)
        forecast = np.zeros(history.shape[1], dtype=np.float64)
        for channel in range(history.shape[1]):
            positions = np.flatnonzero(observed[:, channel])
            if len(positions):
                forecast[channel] = history[positions[-1], channel]
        target, mask = sample["y"].numpy().astype(np.float64), sample["y_mask"].numpy().astype(bool)
        error = (np.abs(target - forecast[None]) * mask).sum(0)
        rows.append({**identity(sample), "channel_count": mask.sum(0).tolist(),
                     "empirical_crps_channel_sum": error.tolist(), "fair_crps_channel_sum": error.tolist()})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--original-data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new directory")
    args = parser.parse_args()
    require_protocol()
    require(np.__version__ == CONFIG["evaluation"]["numpy_version"], "study aggregation requires NumPy 1.26.4")
    torch.set_num_threads(1)
    receipt = read_json(args.prepared / "PREPARATION.json")
    require(receipt["protocol_sha256"] == PROTOCOL_SHA256, "prepared payloads belong to another protocol")
    args.output.mkdir(parents=True, exist_ok=False)
    for dataset, rosters in ROSTERS.items():
        parent = CONFIG["datasets"][dataset]["parent_dataset"]
        train_path = args.original_data / parent / "train.pt"
        ensemble = climatology_ensemble(torch.load(train_path, map_location="cpu", weights_only=True), dataset)
        for roster in rosters:
            payload_path = args.prepared / ROSTER_FILES[roster]
            require(sha256(payload_path) == receipt["files"][payload_path.name], "prepared payload changed")
            payload = torch.load(payload_path, map_location="cpu", weights_only=True)
            metadata = read_json(args.prepared / "weather2023_metadata.json") if roster == "weather2023" else None
            bundle = build_roster(roster, payload, "LL", metadata)
            require(bundle.proof["anonymous_roster_digest"] == receipt["rosters"][roster]["anonymous_roster_digest"], "roster differs from the prepared roster")
            for name, rows in (("climatology", climatology_rows(bundle.dev_score, ensemble)), ("persistence", persistence_rows(bundle.dev_score))):
                write_json(args.output / f"{roster}__{name}.json", {
                    "dataset": dataset, "roster": roster, "baseline": name, "protocol_sha256": PROTOCOL_SHA256,
                    "train_payload_sha256": sha256(train_path), "rows": rows, **stratum_summary(rows),
                })
    print("Baselines written.")


if __name__ == "__main__":
    main()
