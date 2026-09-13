"""Train one coordinate/family cell or score an existing checkpoint.

This is a portable adaptation, not the original executed study launcher.
Only prepared development train/validation payloads are accepted. Scoring
retains retrospective query times and realized target masks.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from crossed_model import build_crossed_model
from geometry_screen.metrics import EqualStratumMetrics
from training import (
    _amp_context, _finite_tensor, _generator, _loader, _move,
    configure_determinism, load_checkpoint, make_partitions, sha256, train, write_json,
)


@torch.inference_mode()
def score(model, partitions, config, device):
    model.eval()
    settings = config["evaluation"]
    metrics = EqualStratumMetrics(partitions.dev_score.expected_strata, 100, 0.95)
    generator = _generator(device, int(settings["sample_seed"]))
    loader = _loader(partitions.dev_score, int(config["training"]["evaluation_batch_size"]), shuffle=False, seed=0)
    for raw in loader:
        batch = _move(raw, device)
        with _amp_context(device, True):
            samples = model.sample(
                batch["x"], batch["x_mark"], batch["x_mask"], batch["y_mark"],
                y_mask=batch["y_mask"], channels=partitions.channels,
                n_samples=int(settings["trajectories"]), ddim_steps=int(settings["ddim_steps"]),
                sample_chunk=int(settings["sample_chunk"]), generator=generator, x0_clip=float(settings["x0_clip"]),
            )
        _finite_tensor(samples, "development samples")
        draws = samples.float().cpu().numpy()
        target, mask = batch["y"].float().cpu().numpy(), batch["y_mask"].cpu().numpy()
        for name in sorted(set(raw["stratum"])):
            indices = np.asarray([i for i, value in enumerate(raw["stratum"]) if value == name])
            metrics.update(name, draws[indices], target[indices], mask[indices])
    return metrics.finalize()


def main():
    config = json.loads((ROOT / "configs" / "study.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("train", "score"))
    parser.add_argument("--dataset", required=True, choices=tuple(config["datasets"]))
    parser.add_argument("--family", required=True, choices=("conditional_diffusion", "direct_gaussian"))
    parser.add_argument("--arm", required=True, choices=tuple(config["coordinate_arms"]), help="E then R; L=separate, S=shared")
    parser.add_argument("--seed", required=True, type=int, choices=config["seeds"])
    parser.add_argument("--train-payload", required=True, type=Path, help="prepared development train .pt file")
    parser.add_argument("--val-payload", required=True, type=Path, help="prepared development validation .pt file")
    parser.add_argument("--checkpoint", type=Path, help="required only for --mode score")
    parser.add_argument("--output", required=True, type=Path, help="new directory; existing paths are refused")
    parser.add_argument("--device", default="cpu", help="cpu or a local CUDA device such as cuda:0")
    parser.add_argument("--verify-study-roster", action="store_true", help="require original aggregate roster digests/counts")
    args = parser.parse_args()
    if (args.mode == "score") != (args.checkpoint is not None):
        parser.error("--checkpoint is required for score and is not accepted for train")
    if args.output.exists():
        parser.error("output already exists; choose a new directory")
    if args.train_payload.resolve() == args.val_payload.resolve():
        parser.error("training and validation files must be distinct")
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        parser.error("only CPU and CUDA devices are supported")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA is not available")
        torch.cuda.set_device(device)
        if not torch.cuda.is_bf16_supported():
            parser.error("CUDA training/scoring requires bfloat16 support")
    if np.__version__ != config["evaluation"]["numpy_version"]:
        parser.error("install requirements.txt: study aggregation requires NumPy 1.26.4")
    configure_determinism(args.seed)
    train_payload = torch.load(args.train_payload, map_location="cpu", weights_only=True)
    val_payload = torch.load(args.val_payload, map_location="cpu", weights_only=True)
    partitions = make_partitions(train_payload, val_payload, args.dataset, config, args.arm)
    expected = json.loads((ROOT / "configs" / "data_roster_manifest.json").read_text(encoding="utf-8"))["datasets"][args.dataset]
    keys = ("anonymous_roster_digest", "source_block_digest", "source_block_count", "partition_counts", "fixed_tensor_shapes")
    roster_match = all(partitions.proof[key] == expected[key] for key in keys)
    if args.verify_study_roster and not roster_match:
        parser.error("prepared payload roster differs from the original study")
    model = build_crossed_model(partitions.channels, args.family, config["model"]).to(device)
    cell = {"dataset": args.dataset, "family": args.family, "arm": args.arm, "seed": args.seed}
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, expected_cell=cell)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "run.json", {
        "artifact_class": "portable_user_run", "mode": args.mode, "cell": cell,
        "config_sha256": sha256(ROOT / "configs" / "study.json"),
        "train_payload_sha256": sha256(args.train_payload), "val_payload_sha256": sha256(args.val_payload),
        "original_roster_matches": roster_match, "torch": str(torch.__version__),
        "numpy": np.__version__, "device_type": device.type,
        "checkpoint_sha256": sha256(args.checkpoint) if args.checkpoint else None,
        "scope": "retrospective development prediction; no claim of exact original-run reproduction",
    })
    write_json(args.output / "partition_proof.json", partitions.proof)
    if args.mode == "train":
        checkpoint_path = train(model, partitions, config, args.output, cell, device)
        print("Saved checkpoint SHA-256:", sha256(checkpoint_path))
    else:
        write_json(args.output / "metrics.json", score(model, partitions, config, device))


if __name__ == "__main__":
    main()
