#!/usr/bin/env python
"""Score one checkpoint of the second protocol once on the Weather 2024 roster (section 5).

The scoring loop and the aggregation are confirmation.score.score_windows and
confirmation.score.stratum_summary, unchanged. An existing output directory
is refused, so a checkpoint is scored once.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from confirmation.common import CONFIG, read_json, require, sha256, write_json
from confirmation2.common import (
    ARMS, DATASET, FAMILY, METADATA_FILE, PAYLOAD_FILE, PROTOCOL_SHA256, ROSTER, SEEDS, build_roster, require_protocol,
)

import numpy as np
import torch

from confirmation.score import score_windows, stratum_summary
from crossed_model import build_crossed_model
from training import configure_determinism, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--prepared", required=True, type=Path, help="directory written by confirmation2/prepare_data.py")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new directory; existing paths are refused")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    require_protocol()
    require(not args.output.exists(), "output exists; a checkpoint is scored once")
    require(np.__version__ == CONFIG["evaluation"]["numpy_version"], "study aggregation requires NumPy 1.26.4")
    device = torch.device(args.device)
    require(device.type == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "scoring runs on CUDA with bfloat16")
    torch.cuda.set_device(device)
    configure_determinism(args.seed)
    payload_path = args.prepared / PAYLOAD_FILE
    receipt = read_json(args.prepared / "PREPARATION.json")
    require(receipt["protocol_sha256"] == PROTOCOL_SHA256 and sha256(payload_path) == receipt["files"][PAYLOAD_FILE], "prepared payload changed")
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    bundle = build_roster(payload, read_json(args.prepared / METADATA_FILE), args.arm)
    recorded = receipt["rosters"][ROSTER]
    for key in ("anonymous_roster_digest", "source_block_digest", "source_block_count", "partition_counts"):
        require(bundle.proof[key] == recorded[key], "roster differs from the prepared roster")
    model = build_crossed_model(bundle.channels, FAMILY, CONFIG["model"]).to(device)
    cell = {"dataset": DATASET, "family": FAMILY, "arm": args.arm, "seed": args.seed}
    stored = load_checkpoint(model, args.checkpoint, expected_cell=cell)
    args.output.mkdir(parents=True, exist_ok=False)
    predictions = args.output / "predictions"
    predictions.mkdir()
    started = time.time()
    rows, official = score_windows(model, bundle, FAMILY, device, predictions)
    summary = stratum_summary(rows)
    if "primary_equal_stratum" in official:
        difference = abs(official["primary_equal_stratum"]["marginal_crps"] - summary["equal_stratum"]["crps_empirical"])
        require(difference <= 1e-9, "exported CRPS differs from the unchanged study aggregate")
    write_json(args.output / "window_diagnostics.json", {"cell": cell, "roster": ROSTER, "rows": rows})
    write_json(args.output / "metrics.json", official)
    write_json(args.output / "summary.json", {
        "cell": cell, "roster": ROSTER, "protocol_sha256": PROTOCOL_SHA256,
        "checkpoint_sha256": sha256(args.checkpoint), "checkpoint_epoch": stored["epoch"],
        "payload_sha256": sha256(payload_path), "roster_digest": bundle.proof["anonymous_roster_digest"]["score"],
        "trajectories": 100, "sample_seed": int(CONFIG["evaluation"]["sample_seed"]),
        "torch": str(torch.__version__), "numpy": np.__version__, "device": torch.cuda.get_device_name(device),
        "elapsed_seconds": time.time() - started, **summary,
    })


if __name__ == "__main__":
    main()
