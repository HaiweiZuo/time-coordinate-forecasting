"""Shared constants and helpers for the confirmation run.

New file added after release v1.0.1 for CONFIRMATION_PROTOCOL.md. It only
reads the unchanged study configuration; model, loss, training and scoring
modules under src/ are imported as they are and never edited.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent.parent
for entry in (ROOT / "src", ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

PROTOCOL_SHA256 = "d13cc6b35f126a16d919215a0929a8d1b1da0526550b182251bb17d399cf5243"
CONFIG = json.loads((ROOT / "configs" / "study.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((ROOT / "configs" / "data_roster_manifest.json").read_text(encoding="utf-8"))

CONFIRMATION_SEEDS = tuple(range(2027, 2037))
EXPLORATORY_SEEDS = tuple(CONFIG["seeds"])
ARMS = ("LL", "SL", "LS", "SS")
FAMILIES = ("conditional_diffusion", "direct_gaussian")
DATASETS = ("weather_jena2020", "ushcn")
ROSTERS = {"ushcn": ("ushcn_test",), "weather_jena2020": ("weather2023", "weather2020_test")}
ROSTER_FILES = {"ushcn_test": "ushcn_test.pt", "weather2023": "weather2023.pt", "weather2020_test": "weather2020_test.pt"}
ROSTER_CAPS = {"ushcn_test": 128, "weather2020_test": 100}
TEMPORAL_YEAR = 2023
TEMPORAL_RECORD_ID = f"weather:temporal{TEMPORAL_YEAR}"
WEATHER_TEST_RECORD_ID = "weather:test"
WEATHER_SCALER_SHA256 = "4c70adb42907df22a705ef3b7685ee7eeb2d75e6d9acb32c0e6cc2a0733cc238"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def cell_name(dataset: str, family: str, arm: str, seed: int) -> str:
    return f"{dataset}__{arm}__{family}__s{seed}"


def require_protocol() -> None:
    """Refuse to run if the committed protocol file is not the frozen one."""
    require(sha256(ROOT / "CONFIRMATION_PROTOCOL.md") == PROTOCOL_SHA256, "CONFIRMATION_PROTOCOL.md differs from the frozen file")
