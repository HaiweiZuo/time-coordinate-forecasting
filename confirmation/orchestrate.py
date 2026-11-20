#!/usr/bin/env python
"""Run the 160 confirmation fits and their scoring calls on one machine.

Orchestration only: each fit and each scoring call is a separate process of
confirmation/fit.py or confirmation/score.py. Cells are started in the order
of protocol section 3. A fit is rerun only after a process failure, with the
same seed, and every rerun is logged. Completed cells are never repeated.

The runner waits whenever free host memory is below a floor, so that it does
not put other workloads on a shared machine at risk.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from confirmation.common import (
    ARMS, CONFIRMATION_SEEDS, DATASETS, FAMILIES, ROOT, ROSTERS, CONFIG, cell_name, require, require_protocol,
)

LOCK = threading.Lock()
MAX_ATTEMPTS = 3


def ordered_cells() -> list[tuple[str, str, str, int]]:
    tiers = [
        [("conditional_diffusion", arm) for arm in ("LL", "SL")],
        [("conditional_diffusion", "SS")],
        [("direct_gaussian", arm) for arm in ("LL", "SL")],
        [("conditional_diffusion", "LS"), ("direct_gaussian", "LS"), ("direct_gaussian", "SS")],
    ]
    cells = [(dataset, family, arm, seed)
             for tier in tiers for seed in CONFIRMATION_SEEDS for dataset in DATASETS for family, arm in tier]
    require(len(cells) == len(set(cells)) == len(DATASETS) * len(FAMILIES) * len(ARMS) * len(CONFIRMATION_SEEDS), "cell matrix is not the registered 160")
    return cells


def available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1048576
    return float("inf")


def log_event(root: Path, **event) -> None:
    event["time_unix"] = time.time()
    with LOCK, (root / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def run_process(command, log_path: Path, environment) -> int:
    with log_path.open("ab") as log:
        return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment, cwd=str(ROOT)).returncode


def wait_for_memory(root: Path, floor: float) -> None:
    while available_gb() < floor and not (root / "STOP").exists():
        time.sleep(20)


def attempt(root: Path, kind: str, name: str, output: Path, done: Path, command, environment, floor: float) -> bool:
    """Run one fit or scoring call; move a failed directory aside and rerun the same cell."""
    if done.exists():
        return True
    for number in range(1, MAX_ATTEMPTS + 1):
        if (root / "STOP").exists():
            return False
        if output.exists():
            aside = root / "failed" / f"{kind}__{name}__{int(time.time())}"
            aside.parent.mkdir(exist_ok=True)
            output.rename(aside)
            log_event(root, event="rerun", kind=kind, cell=name, moved_to=aside.name, reason="incomplete output of a failed process")
        wait_for_memory(root, floor)
        started = time.time()
        code = run_process(command, root / "logs" / f"{kind}__{name}.log", environment)
        log_event(root, event="process_end", kind=kind, cell=name, attempt=number, returncode=code, seconds=time.time() - started)
        if code == 0 and done.exists():
            return True
    log_event(root, event="gave_up", kind=kind, cell=name)
    return False


def worker(root: Path, queue: list, gpu: str, args) -> None:
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1",
                       CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    python = sys.executable
    while True:
        with LOCK:
            if not queue or (root / "STOP").exists():
                return
            dataset, family, arm, seed = queue.pop(0)
        name = cell_name(dataset, family, arm, seed)
        parent = CONFIG["datasets"][dataset]["parent_dataset"]
        cell_arguments = ["--dataset", dataset, "--family", family, "--arm", arm, "--seed", str(seed), "--device", "cuda:0"]
        fit_output = root / "fits" / name
        fitted = attempt(root, "fit", name, fit_output, fit_output / "terminal.json", [
            python, "-m", "confirmation.fit", *cell_arguments,
            "--train-payload", str(args.original_data / parent / "train.pt"),
            "--val-payload", str(args.original_data / parent / "val.pt"), "--output", str(fit_output),
        ], environment, args.min_available_gb)
        if not fitted:
            continue
        for roster in ROSTERS[dataset]:
            score_output = root / "scores" / name / roster
            score_output.parent.mkdir(parents=True, exist_ok=True)
            attempt(root, "score", f"{name}__{roster}", score_output, score_output / "summary.json", [
                python, "-m", "confirmation.score", *cell_arguments, "--roster", roster,
                "--prepared", str(args.prepared), "--checkpoint", str(fit_output / "checkpoint.pt"), "--output", str(score_output),
            ], environment, args.min_available_gb)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="run directory (created if missing)")
    parser.add_argument("--original-data", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--gpu", required=True, action="append", help="GPU UUID or index; repeat for several")
    parser.add_argument("--concurrency", required=True, type=int, help="processes per GPU")
    parser.add_argument("--min-available-gb", type=float, default=3.0)
    parser.add_argument("--limit", type=int, help="start at most this many cells (for a staged launch)")
    args = parser.parse_args()
    require_protocol()
    root = args.root
    for directory in (root, root / "fits", root / "scores", root / "logs"):
        directory.mkdir(parents=True, exist_ok=True)
    queue = ordered_cells()
    if args.limit:
        queue = queue[:args.limit]
    log_event(root, event="orchestrator_start", cells=len(queue), gpus=args.gpu, concurrency=args.concurrency, pid=os.getpid())
    threads = [threading.Thread(target=worker, args=(root, queue, gpu, args), daemon=True)
               for gpu in args.gpu for _ in range(args.concurrency)]
    for thread in threads:
        thread.start()
        time.sleep(5)
    for thread in threads:
        thread.join()
    log_event(root, event="orchestrator_end", stopped=(root / "STOP").exists())


if __name__ == "__main__":
    main()
