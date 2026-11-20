#!/usr/bin/env python
"""Reference baselines on the Weather 2024 roster (second protocol, section 6). CPU only.

Climatology and persistence are the functions of confirmation/baselines.py,
unchanged, applied to the same roster, target masks and aggregation as the
model scores.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from confirmation.common import CONFIG, read_json, require, sha256, write_json
from confirmation2.common import (
    DATASET, METADATA_FILE, PARENT_DATASET, PAYLOAD_FILE, PROTOCOL_SHA256, ROSTER, build_roster, require_protocol,
)

import numpy as np
import torch

from confirmation.baselines import climatology_ensemble, climatology_rows, persistence_rows
from confirmation.score import stratum_summary


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
    payload_path = args.prepared / PAYLOAD_FILE
    require(receipt["protocol_sha256"] == PROTOCOL_SHA256 and sha256(payload_path) == receipt["files"][PAYLOAD_FILE], "prepared payload changed")
    train_path = args.original_data / PARENT_DATASET / "train.pt"
    ensemble = climatology_ensemble(torch.load(train_path, map_location="cpu", weights_only=True), DATASET)
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    bundle = build_roster(payload, read_json(args.prepared / METADATA_FILE), "LL")
    require(bundle.proof["anonymous_roster_digest"] == receipt["rosters"][ROSTER]["anonymous_roster_digest"], "roster differs from the prepared roster")
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("climatology", climatology_rows(bundle.dev_score, ensemble)), ("persistence", persistence_rows(bundle.dev_score))):
        write_json(args.output / f"{ROSTER}__{name}.json", {
            "dataset": DATASET, "roster": ROSTER, "baseline": name, "protocol_sha256": PROTOCOL_SHA256,
            "train_payload_sha256": sha256(train_path), "rows": rows, **stratum_summary(rows),
        })
    print("Baselines written.")


if __name__ == "__main__":
    main()
