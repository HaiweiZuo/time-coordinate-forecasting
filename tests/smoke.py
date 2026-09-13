"""Synthetic forward checks only: no fitting, data downloads, or study scores."""

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from crossed_model import build_crossed_model
from diagnostic_metrics import self_test as score_self_test
from geometry_screen.data import GeometrySpec, encode_time_labels
from geometry_screen.models import build_model
from training import configure_determinism, load_checkpoint, make_partitions


def synthetic_inputs(dataset_config, arm_config, channels):
    spec = GeometrySpec.from_mapping(dataset_config["strata"][-1])
    e_time, y_time = encode_time_labels(spec, arm_config["embedding"])
    r_time, _ = encode_time_labels(spec, arm_config["raster"])
    x = torch.randn(1, spec.history_points, channels)
    x_mask = torch.ones_like(x, dtype=torch.bool)
    x_mask[:, 1::3, 0] = False
    x = torch.where(x_mask, x, 0.0)
    marks = torch.stack((e_time, r_time), dim=-1).unsqueeze(0)
    y_mask = torch.ones(1, spec.horizon_points, channels, dtype=torch.bool)
    return x, marks, x_mask, y_time.unsqueeze(0), y_mask


@torch.inference_mode()
def check_model(model, inputs, channels):
    x, marks, mask, query, y_mask = inputs
    model.eval()
    loss = model.training_loss(torch.zeros_like(y_mask, dtype=torch.float32), y_mask, x, marks, mask, query,
                               generator=torch.Generator().manual_seed(11))
    assert torch.isfinite(loss).all()
    samples = model.sample(x, marks, mask, query, y_mask=y_mask, channels=channels,
                           n_samples=2, ddim_steps=2, sample_chunk=2,
                           generator=torch.Generator().manual_seed(12), x0_clip=5.0)
    assert samples.shape == (1, 2, query.shape[1], channels)
    assert torch.isfinite(samples).all()
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset", choices=("weather_jena2020", "ushcn"))
    parser.add_argument("--family", choices=("conditional_diffusion", "direct_gaussian"))
    parser.add_argument("--arm", choices=("LL", "LS", "SL", "SS"))
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    if args.checkpoint and not all((args.dataset, args.family, args.arm)):
        parser.error("checkpoint verification requires dataset, family, and arm")
    config = json.loads((ROOT / "configs" / "study.json").read_text(encoding="utf-8"))
    configure_determinism(2024)
    score_self_test()
    if args.checkpoint:
        channels = 21 if args.dataset == "weather_jena2020" else 5
        model = build_crossed_model(channels, args.family, config["model"])
        checkpoint = load_checkpoint(model, args.checkpoint, expected_cell={
            "dataset": args.dataset, "family": args.family, "arm": args.arm, "seed": args.seed})
        check_model(model, synthetic_inputs(config["datasets"][args.dataset], config["coordinate_arms"][args.arm], channels), channels)
        print(json.dumps({"check": "released_checkpoint_synthetic_forward", "epoch": checkpoint.get("epoch"),
                          "dataset": args.dataset, "family": args.family, "arm": args.arm,
                          "torch": str(torch.__version__), "numpy": np.__version__}))
        return
    small = dict(config["model"], dim=16, grid_size=8, layers=1, heads=4, ff_dim=32, dropout=0.0)
    for dataset, dataset_config in config["datasets"].items():
        for family in ("conditional_diffusion", "direct_gaussian"):
            original = build_model(2, family, small).eval()
            model = build_crossed_model(2, family, small).eval()
            model.load_state_dict(original.state_dict(), strict=True)
            for arm, modes in config["coordinate_arms"].items():
                inputs = synthetic_inputs(dataset_config, modes, 2)
                crossed_samples = check_model(model, inputs, 2)
                if arm in {"LL", "SS"}:
                    x, marks, mask, query, y_mask = inputs
                    reference = original.sample(x, marks[..., 0], mask, query, y_mask=y_mask, channels=2,
                                                n_samples=2, ddim_steps=2, sample_chunk=2,
                                                generator=torch.Generator().manual_seed(12), x0_clip=5.0)
                    torch.testing.assert_close(crossed_samples, reference, rtol=0, atol=0)
            with tempfile.TemporaryDirectory(prefix="coordinate-smoke-") as directory:
                path = Path(directory) / "synthetic.pt"
                cell = {"dataset": dataset, "family": family, "arm": "SS", "seed": 2024}
                torch.save({"model_state": model.state_dict(), "cell": cell}, path)
                load_checkpoint(model, path, expected_cell=cell)
                try:
                    load_checkpoint(model, path, expected_cell=dict(cell, arm="LL"))
                except ValueError:
                    pass
                else:
                    raise AssertionError("mismatched checkpoint coordinates were accepted")
        # Synthetic records only; reduced caps exercise E/R pairing and the data
        # contract without changing or reading the study's original inputs.
        synthetic_config = copy.deepcopy(config)
        for policy in synthetic_config["datasets"][dataset]["partitions"].values():
            policy.update(cap_per_stratum=2, minimum_per_stratum=1)
        length = 16000 if dataset == "weather_jena2020" else 2400
        times = torch.arange(length, dtype=torch.float32)
        values = torch.stack((times.sin(), times.cos()), dim=-1)
        mask = torch.ones_like(values, dtype=torch.bool)
        train_record = ("synthetic-training", times, values, mask)
        val_record = ("synthetic-validation", times, values, mask)
        train_payload = train_record if dataset == "weather_jena2020" else [train_record]
        val_payload = val_record if dataset == "weather_jena2020" else [val_record]
        parts = make_partitions(train_payload, val_payload, dataset, synthetic_config, "SL")
        assert parts.proof["fit_early_source_block_overlap"] == 0
        assert parts.fit[0]["x_mark"].shape[-1] == 2
    print(json.dumps({"check": "synthetic_forward_and_pairing_pass", "cells": 16,
                      "matched_arm_bitwise_equal": True, "torch": str(torch.__version__), "numpy": np.__version__}))


if __name__ == "__main__":
    main()
