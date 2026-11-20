#!/usr/bin/env python
"""Score one checkpoint once on one roster (protocol section 5) and export statistics.

The sampling call, ensemble size, sampling seed, DDIM settings, clipping and
batch order repeat run_experiment.score. The draws go to the unchanged
EqualStratumMetrics and to diagnostic_metrics.score_terms, which returns the
per-window, per-channel sums of empirical and fair CRPS needed for the
per-block export. An existing output directory is refused, so a checkpoint is
scored once per roster.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from confirmation.common import (
    ARMS, CONFIG, CONFIRMATION_SEEDS, DATASETS, EXPLORATORY_SEEDS, FAMILIES, PROTOCOL_SHA256, ROSTER_FILES, ROSTERS,
    read_json, require, require_protocol, sha256, write_json,
)

import numpy as np
import torch

from confirmation.rosters import build_roster
from crossed_model import build_crossed_model
from diagnostic_metrics import gaussian_terms, score_terms
from geometry_screen.metrics import EqualStratumMetrics
from training import _amp_context, _finite_tensor, _generator, _loader, _move, configure_determinism, load_checkpoint


@torch.inference_mode()
def score_windows(model, bundle, family: str, device: torch.device, predictions: Path | None):
    """Same loop as run_experiment.score, plus per-window sufficient statistics."""
    model.eval()
    settings = CONFIG["evaluation"]
    windows = bundle.dev_score
    metadata = {sample["sample_id"]: sample for sample in windows.samples}
    accumulator = EqualStratumMetrics(windows.expected_strata, 100, 0.95)
    generator = _generator(device, int(settings["sample_seed"]))
    loader = _loader(windows, int(CONFIG["training"]["evaluation_batch_size"]), shuffle=False, seed=0)
    rows = []
    for number, raw in enumerate(loader):
        batch = _move(raw, device)
        with _amp_context(device, True):
            samples = model.sample(
                batch["x"], batch["x_mark"], batch["x_mask"], batch["y_mark"],
                y_mask=batch["y_mask"], channels=bundle.channels,
                n_samples=int(settings["trajectories"]), ddim_steps=int(settings["ddim_steps"]),
                sample_chunk=int(settings["sample_chunk"]), generator=generator, x0_clip=float(settings["x0_clip"]),
            )
        _finite_tensor(samples, "confirmation samples")
        draws = samples.float().cpu().numpy()
        target, mask = batch["y"].float().cpu().numpy(), batch["y_mask"].cpu().numpy()
        terms = score_terms(draws, target, mask)
        archive = {"samples": draws, "target_mask": mask, "sample_id": np.asarray(raw["sample_id"])}
        if family == "direct_gaussian":
            with _amp_context(device, True):
                mean, scale = model.distribution(batch["x"], batch["x_mark"], batch["x_mask"], batch["y_mark"], batch["y_mask"])
            mean, scale = mean.float().cpu().numpy(), scale.float().cpu().numpy()
            terms.update(gaussian_terms(mean, scale, target, mask))
            archive.update(gaussian_mean=mean, gaussian_scale=scale)
        if predictions is not None:
            with (predictions / f"batch_{number:04d}.npz").open("xb") as handle:
                np.savez(handle, **archive)
        for name in sorted(set(raw["stratum"])):
            indices = np.asarray([i for i, value in enumerate(raw["stratum"]) if value == name])
            accumulator.update(name, draws[indices], target[indices], mask[indices])
        for index, sample_id in enumerate(raw["sample_id"]):
            sample = metadata[sample_id]
            rows.append({"sample_id": sample_id, "anchor_digest": sample["anchor_digest"],
                         "source_block_digest": sample["source_block_digest"], "stratum": sample["stratum"],
                         **{name: value[index] for name, value in terms.items()}})
    try:
        official = accumulator.finalize()
    except ValueError as error:
        # The unchanged aggregate also needs a lag-1 pair in every active channel; CRPS does not.
        official = {"unavailable": str(error)}
    return rows, official


def stratum_summary(rows) -> dict:
    """Channel-macro within stratum, then equal weight over the four strata."""
    strata = {}
    for name in sorted({row["stratum"] for row in rows}):
        selected = [row for row in rows if row["stratum"] == name]
        counts = np.asarray([row["channel_count"] for row in selected]).sum(0)
        active = counts > 0
        entry = {"windows": len(selected), "n_scored": int(counts.sum()), "active_channels": int(active.sum())}
        for label, key in (("crps_empirical", "empirical_crps_channel_sum"), ("crps_fair", "fair_crps_channel_sum")):
            sums = np.asarray([row[key] for row in selected]).sum(0)
            entry[label] = float(np.mean(sums[active] / counts[active]))
        strata[name] = entry
    equal = {label: float(np.mean([entry[label] for entry in strata.values()])) for label in ("crps_empirical", "crps_fair")}
    equal["n_scored"] = int(sum(entry["n_scored"] for entry in strata.values()))
    return {"strata": strata, "equal_stratum": equal}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--family", required=True, choices=FAMILIES)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--roster", required=True)
    parser.add_argument("--prepared", required=True, type=Path, help="directory written by confirmation/prepare_data.py")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new directory; existing paths are refused")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-predictions", action="store_true")
    args = parser.parse_args()
    require_protocol()
    require(args.roster in ROSTERS[args.dataset], "roster does not belong to this dataset")
    require(args.seed in CONFIRMATION_SEEDS + EXPLORATORY_SEEDS, "unknown seed")
    require(not args.output.exists(), "output exists; a checkpoint is scored once per roster")
    require(np.__version__ == CONFIG["evaluation"]["numpy_version"], "study aggregation requires NumPy 1.26.4")
    device = torch.device(args.device)
    require(device.type == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "scoring runs on CUDA with bfloat16")
    torch.cuda.set_device(device)
    configure_determinism(args.seed)
    payload_path = args.prepared / ROSTER_FILES[args.roster]
    receipt = read_json(args.prepared / "PREPARATION.json")
    require(receipt["protocol_sha256"] == PROTOCOL_SHA256 and sha256(payload_path) == receipt["files"][payload_path.name], "prepared payload changed")
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    metadata = read_json(args.prepared / "weather2023_metadata.json") if args.roster == "weather2023" else None
    bundle = build_roster(args.roster, payload, args.arm, metadata)
    recorded = receipt["rosters"][args.roster]
    for key in ("anonymous_roster_digest", "source_block_digest", "source_block_count", "partition_counts"):
        require(bundle.proof[key] == recorded[key], "roster differs from the prepared roster")
    model = build_crossed_model(bundle.channels, args.family, CONFIG["model"]).to(device)
    cell = {"dataset": args.dataset, "family": args.family, "arm": args.arm, "seed": args.seed}
    stored = load_checkpoint(model, args.checkpoint, expected_cell=cell)
    args.output.mkdir(parents=True, exist_ok=False)
    predictions = None
    if not args.no_predictions:
        predictions = args.output / "predictions"
        predictions.mkdir()
    started = time.time()
    rows, official = score_windows(model, bundle, args.family, device, predictions)
    summary = stratum_summary(rows)
    if "primary_equal_stratum" in official:
        difference = abs(official["primary_equal_stratum"]["marginal_crps"] - summary["equal_stratum"]["crps_empirical"])
        require(difference <= 1e-9, "exported CRPS differs from the unchanged study aggregate")
    write_json(args.output / "window_diagnostics.json", {"cell": cell, "roster": args.roster, "rows": rows})
    write_json(args.output / "metrics.json", official)
    write_json(args.output / "summary.json", {
        "cell": cell, "roster": args.roster, "protocol_sha256": PROTOCOL_SHA256,
        "checkpoint_sha256": sha256(args.checkpoint), "checkpoint_epoch": stored["epoch"],
        "payload_sha256": sha256(payload_path), "roster_digest": bundle.proof["anonymous_roster_digest"]["score"],
        "trajectories": 100, "sample_seed": int(CONFIG["evaluation"]["sample_seed"]),
        "torch": str(torch.__version__), "numpy": np.__version__, "device": torch.cuda.get_device_name(device),
        "elapsed_seconds": time.time() - started, **summary,
    })


if __name__ == "__main__":
    main()
