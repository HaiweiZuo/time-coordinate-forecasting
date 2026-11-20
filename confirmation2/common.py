"""Constants of the second confirmation protocol (CONFIRMATION_PROTOCOL_2.md).

New files only. The released study code under src/ and the modules of the
first confirmation run under confirmation/ are imported as they are and never
edited.
"""

from __future__ import annotations

from confirmation.common import ROOT, require, sha256

PROTOCOL_FILE = "CONFIRMATION_PROTOCOL_2.md"
PROTOCOL_SHA256 = "b833989d69d97390c8f7eebb079b2dd60aaeef4aed021826ba56819dfedfcdbb"

SEEDS = tuple(range(2037, 2047))
ARMS = ("LL", "SL", "LS", "SS")
DATASET = "weather_jena2020"
PARENT_DATASET = "weather"
FAMILY = "direct_gaussian"
ROSTER = "weather2024"
YEAR = 2024
RECORD_ID = f"weather:temporal{YEAR}"
ARCHIVE = f"mpi_roof_{YEAR}.zip"
PAYLOAD_FILE = f"{ROSTER}.pt"
METADATA_FILE = f"{ROSTER}_metadata.json"


def require_protocol() -> None:
    """Refuse to run if the committed second protocol is not the frozen file."""
    require(sha256(ROOT / PROTOCOL_FILE) == PROTOCOL_SHA256, f"{PROTOCOL_FILE} differs from the frozen file")


def build_roster(payload, metadata, arm: str):
    """Weather 2024 roster: the 2022 temporal rule, as used for Weather 2023."""
    from confirmation.rosters import _temporal

    return _temporal(payload, metadata, arm, record_id=RECORD_ID)
