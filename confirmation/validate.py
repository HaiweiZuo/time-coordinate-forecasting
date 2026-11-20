#!/usr/bin/env python
"""Pipeline health check on data already used by the study (DEVIATIONS.md, N1.10).

No confirmation record and no confirmation seed is touched.

1. The Weather 2022 temporal roster is rebuilt with the confirmation code and
   must reproduce the archived roster and source-block digests of the 2022
   analysis. This checks archive reading, scaling and window construction.
2. A released checkpoint is scored on that roster with the confirmation
   scoring loop and compared with its archived 2022 scores. This checks the
   scoring path in the present software and hardware environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from confirmation.common import ARMS, CONFIG, FAMILIES, read_json, require, write_json

import numpy as np
import torch

from confirmation.prepare_data import prepare_weather_temporal
from confirmation.rosters import _temporal, paired_across_arms
from confirmation.score import score_windows, stratum_summary
from crossed_model import build_crossed_model
from training import configure_determinism, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives-2022", type=Path, nargs=2, required=True)
    parser.add_argument("--reference-2020", type=Path, required=True)
    parser.add_argument("--original-data", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True, help="results/temporal_24_aggregates.json")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(np.__version__ == CONFIG["evaluation"]["numpy_version"], "study aggregation requires NumPy 1.26.4")
    archived = read_json(args.archived)
    payload, metadata = prepare_weather_temporal(list(args.archives_2022), args.reference_2020, args.original_data, year=2022)
    bundles = {arm: _temporal(payload, metadata, arm, record_id="weather:temporal2022") for arm in ARMS}
    paired_across_arms(bundles)
    proof, expected = bundles["LL"].proof, archived["partition_proof"]
    roster = {
        "anchor_digest_matches": proof["anonymous_roster_digest"]["score"] == expected["anonymous_roster_digest"]["dev_score"],
        "block_digest_matches": proof["source_block_digest"]["score"] == expected["source_block_digest"]["dev_score"],
        "counts_match": proof["partition_counts"]["score"] == expected["partition_counts"]["dev_score"],
        "lattice_rows_match": metadata["lattice_rows"] == archived["metadata"]["lattice_rows"],
        "absent_rows_match": metadata["absent_timestamp_rows"] == archived["metadata"]["absent_timestamp_rows"],
    }
    report = {"roster_2022": roster}
    require(all(roster.values()), f"2022 roster differs from the archived analysis: {roster}")
    if args.checkpoint is not None:
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        configure_determinism(args.seed)
        cell = {"dataset": "weather_jena2020", "family": args.family, "arm": args.arm, "seed": args.seed}
        model = build_crossed_model(21, args.family, CONFIG["model"]).to(device)
        load_checkpoint(model, args.checkpoint, expected_cell=cell)
        rows, official = score_windows(model, bundles[args.arm], args.family, device, None)
        summary = stratum_summary(rows)["equal_stratum"]
        name = f"weather2022__{args.arm}__{args.family}__s{args.seed}"
        reference = next(entry for entry in archived["cells"] if entry["cell_id"] == name)["summary"]["equal_stratum"]
        report["scoring_2022"] = {
            "cell": name, "empirical_crps": summary["crps_empirical"], "archived_empirical_crps": reference["empirical_crps"],
            "fair_crps": summary["crps_fair"], "archived_fair_crps": reference["fair_crps"],
            "empirical_relative_difference": summary["crps_empirical"] / reference["empirical_crps"] - 1,
            "torch": str(torch.__version__), "device": torch.cuda.get_device_name(device),
        }
    write_json(args.output, report)
    print(__import__("json").dumps(report, indent=2))


if __name__ == "__main__":
    main()
