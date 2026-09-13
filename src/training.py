"""Portable extraction of the study's single-cell fitting helpers.

The tensor operations and optimization rule are unchanged. Process supervision,
hardware-identity binding, and private artifact registries are not included.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from geometry_screen.data import PartitionBundle, WindowDataset, build_partitions, collate_windows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def configure_determinism(seed: int) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 before torch import")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(1)


def make_partitions(train_payload, val_payload, dataset, config, arm) -> PartitionBundle:
    """Cross E/R labels only; retain paired values, masks, anchors, and query E."""
    modes = config["coordinate_arms"][arm]
    bundles = {
        mode: build_partitions(train_payload, val_payload, dataset, config["datasets"][dataset], mode, config["analysis_id"])
        for mode in sorted(set(modes.values()))
    }
    parts = {}
    for role in ("fit", "early_stop", "dev_score"):
        left = getattr(bundles[modes["embedding"]], role)
        right = getattr(bundles[modes["raster"]], role)
        if len(left) != len(right):
            raise ValueError("unpaired coordinate roster")
        samples = []
        for e, r in zip(left.samples, right.samples):
            for key in ("sample_id", "anchor_digest", "source_block_digest", "stratum", "source_bounds"):
                if e[key] != r[key]:
                    raise ValueError("unpaired coordinate sample")
            for key in ("x", "y", "x_mask", "y_mask"):
                if not torch.equal(e[key], r[key]):
                    raise ValueError("data or mask changed by coordinate factor")
            samples.append(dict(e, x_mark=torch.stack((e["x_mark"], r["x_mark"]), dim=-1)))
        parts[role] = WindowDataset(samples, left.expected_strata)
    reference = bundles[modes["embedding"]]
    return PartitionBundle(parts["fit"], parts["early_stop"], parts["dev_score"], reference.proof, reference.channels)


def _amp_context(device: torch.device, enabled: bool):
    return torch.autocast("cuda", dtype=torch.bfloat16) if enabled and device.type == "cuda" else nullcontext()


def _move(batch: Mapping[str, object], device: torch.device) -> dict[str, object]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed))


def _finite_tensor(value: torch.Tensor, label: str) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"nonfinite {label}")


def _model_loss(model, batch, generator, reduction="mean"):
    loss = model.training_loss(
        batch["y"], batch["y_mask"], batch["x"], batch["x_mark"], batch["x_mask"],
        batch["y_mark"], generator=generator, reduction=reduction,
    )
    _finite_tensor(loss, "loss")
    return loss


def _optimizer(model, training):
    return torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]),
        betas=tuple(float(value) for value in training["betas"]),
        weight_decay=float(training["weight_decay"]),
    )


def _train_step(model, batch, optimizer, generator, device, use_amp, gradient_clip):
    optimizer.zero_grad(set_to_none=True)
    with _amp_context(device, use_amp):
        loss = _model_loss(model, batch, generator)
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            _finite_tensor(parameter.grad, "gradient")
    norm = clip_grad_norm_(model.parameters(), float(gradient_clip))
    _finite_tensor(torch.as_tensor(norm), "gradient norm")
    optimizer.step()
    for parameter in model.parameters():
        _finite_tensor(parameter, "updated parameter")
    return float(loss.detach())


@torch.inference_mode()
def _early_loss(model, loader, device, use_amp, seed):
    model.eval()
    generator = _generator(device, seed)
    numerator, denominator = 0.0, 0
    for raw in loader:
        batch = _move(raw, device)
        with _amp_context(device, use_amp):
            losses = _model_loss(model, batch, generator, "none")
        numerator += float(losses.sum())
        denominator += int(losses.numel())
    value = numerator / denominator if denominator else float("nan")
    if not math.isfinite(value):
        raise FloatingPointError("nonfinite/empty early-stop loss")
    return value


def _loader(dataset, batch_size: int, *, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
        collate_fn=collate_windows, num_workers=0,
    )


def train(model, partitions, config, output: Path, cell, device):
    settings = config["training"]
    fit = _loader(partitions.fit, int(settings["batch_size"]), shuffle=True, seed=cell["seed"])
    early = _loader(partitions.early_stop, int(settings["batch_size"]), shuffle=False, seed=cell["seed"])
    optimizer = _optimizer(model, settings)
    generator = _generator(device, cell["seed"])
    best, best_epoch, stale = float("inf"), -1, 0
    history = []
    for epoch in range(int(settings["max_epochs"])):
        model.train()
        total, seen = 0.0, 0
        for raw in fit:
            batch = _move(raw, device)
            loss = _train_step(model, batch, optimizer, generator, device, True, float(settings["gradient_clip"]))
            size = int(batch["y"].shape[0])
            total += loss * size
            seen += size
        early_loss = _early_loss(model, early, device, True, int(settings["validation_noise_seed"]))
        row = {"epoch": epoch, "fit_loss": total / seen, "early_stop_loss": early_loss}
        history.append(row)
        write_json(output / "history.json", history)
        if early_loss < best:
            best, best_epoch, stale = early_loss, epoch, 0
            temporary = output / "checkpoint.pt.tmp"
            torch.save({"schema_version": 1, "cell": cell, "epoch": epoch, "early_stop_loss": early_loss,
                        "model_state": model.state_dict(), "artifact_class": "portable_user_run"}, temporary)
            temporary.replace(output / "checkpoint.pt")
        else:
            stale += 1
        print(json.dumps({**row, "best_epoch": best_epoch}), flush=True)
        if stale >= int(settings["patience"]):
            break
    return output / "checkpoint.pt"


def load_checkpoint(model, path: Path, expected_cell=None):
    """Accept released or original model_state containers; no executable pickle."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state"), dict):
        raise ValueError("checkpoint must contain a model_state mapping")
    if expected_cell is not None:
        actual_cell = checkpoint.get("cell")
        if actual_cell is None:
            parts = str(checkpoint.get("cell_id", "")).split("__")
            if len(parts) != 4 or not parts[3].startswith("s") or not parts[3][1:].isdigit():
                raise ValueError("checkpoint has no recognized cell identity")
            actual_cell = {"dataset": parts[0], "family": parts[2],
                           "arm": {"legacy": "LL", "geometry": "SS"}.get(parts[1], parts[1]),
                           "seed": int(parts[3][1:])}
        if actual_cell != expected_cell:
            raise ValueError("checkpoint dataset/family/arm/seed does not match the requested cell")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return checkpoint
