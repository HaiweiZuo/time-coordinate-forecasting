#!/usr/bin/env python
"""Fit one confirmation cell with a new seed (protocol section 3).

Orchestration only. Partitions, model, loss, optimizer, early stopping and
checkpoint selection are the unchanged functions of release v1.0.1; this
entry point exists because run_experiment.py accepts seeds 2024-2026 only.
The original development roster digests and counts are always required.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

from confirmation.common import (
    ARMS, CONFIG, CONFIRMATION_SEEDS, DATASETS, EXPLORATORY_SEEDS, FAMILIES, MANIFEST, PROTOCOL_SHA256, ROOT,
    require, require_protocol, sha256, write_json,
)

import numpy as np
import torch

from crossed_model import build_crossed_model
from training import configure_determinism, make_partitions, train

ROSTER_KEYS = ("anonymous_roster_digest", "source_block_digest", "source_block_count", "partition_counts", "fixed_tensor_shapes")


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=15).stdout.strip()
    except Exception:
        marker = ROOT / "CODE_COMMIT"
        return marker.read_text(encoding="utf-8").strip() if marker.exists() else "unknown"


def environment(device: torch.device) -> dict:
    return {
        "python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "platform": platform.platform(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--family", required=True, choices=FAMILIES)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--train-payload", required=True, type=Path)
    parser.add_argument("--val-payload", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new directory; existing paths are refused")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--development-check", action="store_true",
                        help="allow an original seed (2024-2026) for a pipeline health check on development data only")
    args = parser.parse_args()
    require_protocol()
    allowed = EXPLORATORY_SEEDS if args.development_check else CONFIRMATION_SEEDS
    require(args.seed in allowed, f"seed must be one of {allowed}")
    require(not args.output.exists(), "output already exists; a fit is never resumed or replaced")
    require(np.__version__ == CONFIG["evaluation"]["numpy_version"], "study aggregation requires NumPy 1.26.4")
    device = torch.device(args.device)
    require(device.type == "cuda" and torch.cuda.is_available(), "confirmation fits run on CUDA with bfloat16")
    torch.cuda.set_device(device)
    require(torch.cuda.is_bf16_supported(), "CUDA training requires bfloat16 support")
    configure_determinism(args.seed)
    train_payload = torch.load(args.train_payload, map_location="cpu", weights_only=True)
    val_payload = torch.load(args.val_payload, map_location="cpu", weights_only=True)
    partitions = make_partitions(train_payload, val_payload, args.dataset, CONFIG, args.arm)
    expected = MANIFEST["datasets"][args.dataset]
    require(all(partitions.proof[key] == expected[key] for key in ROSTER_KEYS), "fitting/early-stopping roster differs from the original study")
    model = build_crossed_model(partitions.channels, args.family, CONFIG["model"]).to(device)
    cell = {"dataset": args.dataset, "family": args.family, "arm": args.arm, "seed": args.seed}
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    write_json(args.output / "run.json", {
        "artifact_class": "development_check" if args.development_check else "confirmation_fit",
        "cell": cell, "protocol_sha256": PROTOCOL_SHA256, "code_commit": git_commit(),
        "config_sha256": sha256(ROOT / "configs" / "study.json"),
        "train_payload_sha256": sha256(args.train_payload), "val_payload_sha256": sha256(args.val_payload),
        "original_roster_matches": True, "environment": environment(device), "started_unix": started,
    })
    write_json(args.output / "partition_proof.json", partitions.proof)
    torch.cuda.reset_peak_memory_stats(device)
    checkpoint = train(model, partitions, CONFIG, args.output, cell, device)
    history = json.loads((args.output / "history.json").read_text(encoding="utf-8"))
    selected = min(history, key=lambda row: row["early_stop_loss"])
    stored = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(stored["epoch"] == selected["epoch"] and stored["early_stop_loss"] == selected["early_stop_loss"], "checkpoint is not the early-stopping minimum")
    write_json(args.output / "terminal.json", {
        "state": "complete", "cell": cell, "epochs_run": len(history), "selected_epoch": selected["epoch"],
        "selected_early_stop_loss": selected["early_stop_loss"],
        "stop_reason": "max_epochs" if len(history) >= int(CONFIG["training"]["max_epochs"]) else "patience",
        "checkpoint_sha256": sha256(checkpoint), "elapsed_seconds": time.time() - started,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(device)),
    })


if __name__ == "__main__":
    main()
