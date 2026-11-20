#!/usr/bin/env python
"""Run the 40 fits of the second protocol and their scoring calls on one machine.

Orchestration only: each fit and each scoring call is a separate process of
confirmation2/fit.py or confirmation2/score.py. The rerun rule, the event log
and the memory floor are the functions of confirmation/orchestrate.py.
Order (section 3): LL and SL for all seeds, then LS and SS.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

from confirmation.common import ROOT, cell_name, require
from confirmation.orchestrate import LOCK, attempt, log_event
from confirmation2.common import ARMS, DATASET, FAMILY, PARENT_DATASET, ROSTER, SEEDS, require_protocol


def ordered_cells() -> list[tuple[str, int]]:
    cells = [(arm, seed) for tier in (("LL", "SL"), ("LS", "SS")) for seed in SEEDS for arm in tier]
    require(len(cells) == len(set(cells)) == len(ARMS) * len(SEEDS) == 40, "cell matrix is not the registered 40")
    return cells


def worker(root: Path, queue: list, gpu: str, args) -> None:
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1",
                       CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    python = sys.executable
    while True:
        with LOCK:
            if not queue or (root / "STOP").exists():
                return
            arm, seed = queue.pop(0)
        name = cell_name(DATASET, FAMILY, arm, seed)
        cell_arguments = ["--arm", arm, "--seed", str(seed), "--device", "cuda:0"]
        fit_output = root / "fits" / name
        fitted = attempt(root, "fit", name, fit_output, fit_output / "terminal.json", [
            python, "-m", "confirmation2.fit", *cell_arguments,
            "--train-payload", str(args.original_data / PARENT_DATASET / "train.pt"),
            "--val-payload", str(args.original_data / PARENT_DATASET / "val.pt"), "--output", str(fit_output),
        ], environment, args.min_available_gb)
        if not fitted:
            continue
        score_output = root / "scores" / name / ROSTER
        score_output.parent.mkdir(parents=True, exist_ok=True)
        attempt(root, "score", f"{name}__{ROSTER}", score_output, score_output / "summary.json", [
            python, "-m", "confirmation2.score", *cell_arguments,
            "--prepared", str(args.prepared), "--checkpoint", str(fit_output / "checkpoint.pt"), "--output", str(score_output),
        ], environment, args.min_available_gb)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="run directory (created if missing)")
    parser.add_argument("--original-data", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--gpu", required=True, help="GPU UUID or index")
    parser.add_argument("--concurrency", required=True, type=int, help="processes on the GPU")
    parser.add_argument("--min-available-gb", type=float, default=3.0)
    args = parser.parse_args()
    require_protocol()
    root = args.root
    for directory in (root, root / "fits", root / "scores", root / "logs"):
        directory.mkdir(parents=True, exist_ok=True)
    queue = ordered_cells()
    log_event(root, event="orchestrator_start", cells=len(queue), gpus=[args.gpu], concurrency=args.concurrency, pid=os.getpid())
    threads = [threading.Thread(target=worker, args=(root, queue, args.gpu, args), daemon=True) for _ in range(args.concurrency)]
    for thread in threads:
        thread.start()
        time.sleep(5)
    for thread in threads:
        thread.join()
    log_event(root, event="orchestrator_end", stopped=(root / "STOP").exists())


if __name__ == "__main__":
    main()
