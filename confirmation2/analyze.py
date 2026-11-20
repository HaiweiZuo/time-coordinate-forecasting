#!/usr/bin/env python
"""Analysis of the second protocol (section 6) from the exported delivery files.

Every statistic is computed by confirmation.analyze.analyse and
confirmation.analyze.arm_table as committed for the first protocol: paired
relative effect per seed, exact one-sided sign test, and the two block
bootstrap statistics (2,000 resamples, seed 20261120). This file only sets
the registered constants of the second protocol (seeds, roster, family,
labels) in that module and writes the decision for H3.

H3: direct Gaussian forecaster, Weather 2024, SL versus LL. Confirmed if
T_s = 1 - CRPS_SL,s / CRPS_LL,s is positive in at least 9 of the 10 seeds.
Every other contrast is descriptive.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import confirmation.analyze as first

from confirmation2.common import ARMS, DATASET, FAMILY, ROSTER, SEEDS

H3 = (ROSTER, FAMILY, "SL", "LL")
RULE = "T_s > 0 in at least 9 of 10 seeds"


def configure() -> None:
    """Replace the registered constants of the first protocol with those of the second."""
    first.SEEDS, first.ARMS = SEEDS, ARMS
    first.ROSTERS, first.FAMILIES = ((DATASET, ROSTER),), (FAMILY,)
    first.LABELS = {H3: "H3 (confirmatory)"}


def analyse(scores, blocks, sums, metric: str) -> list[dict]:
    configure()
    results = first.analyse(scores, blocks, sums, metric)
    for entry in results:
        if entry["label"].startswith("H3") and metric == "empirical":
            complete = entry["n"] == len(SEEDS)
            entry["decision"] = ("confirmed" if entry["positive"] >= 9 else "not confirmed") if complete else "undetermined: seeds missing"
            entry["rule"] = RULE
    return results


def markdown(results, arms, missing) -> str:
    lines = ["# Second confirmation analysis (CONFIRMATION_PROTOCOL_2.md, section 6)", "",
             "Paired relative effect T = 1 - CRPS_A / CRPS_B per seed, in percent; positive means A is better.",
             "Only H3 is confirmatory. Intervals are 95% block-bootstrap percentile intervals; they do not include",
             "variation between training seeds and are not part of the decision rule.", ""]
    if missing:
        lines += [f"**Incomplete run: {len(missing)} missing entries. Statistics below use the available seeds only.**", ""]
    for metric in ("empirical", "fair"):
        lines += [f"## {'Primary metric: empirical CRPS' if metric == 'empirical' else 'Sensitivity: fair CRPS'}", "",
                  "| Roster | Family | Contrast | Label | Positive seeds | Median | Min | Max | Sign test p | Bootstrap 95% (registered) | Bootstrap 95% (pooled) |",
                  "|---|---|---|---|---:|---:|---:|---:|---:|---|---|"]
        for r in (x for x in results if x["metric"] == metric):
            interval = lambda v: f"{100 * v['interval95'][0]:+.1f} to {100 * v['interval95'][1]:+.1f}"
            lines.append(f"| {r['roster']} | {r['family']} | {r['contrast']} | {r['label']} | {r['positive']}/{r['n']} | {100 * r['median']:+.1f} | "
                         f"{100 * r['min']:+.1f} | {100 * r['max']:+.1f} | {r['sign_test_p_one_sided']:.4f} | {interval(r['bootstrap_registered'])} | {interval(r['bootstrap_pooled'])} |")
        lines.append("")
    decision = next((r for r in results if "decision" in r), None)
    if decision:
        lines += [f"**H3 decision by the registered rule: {decision['decision']}** ({decision['positive']} of {decision['n']} seeds positive; "
                  f"per-seed effects in percent, seeds {decision['seeds'][0]} to {decision['seeds'][-1]}: "
                  f"{', '.join(f'{100 * e:+.1f}' for e in decision['effects'])}).", ""]
    lines += ["## Equal-stratum empirical CRPS by arm, with reference baselines", "",
              "| Roster | Family | Arm | Seeds | Median | Min | Max | Climatology | Persistence |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in arms:
        lines.append(f"| {r['roster']} | {r['family']} | {r['arm']} | {r['n']} | {r['median']:.4f} | {r['min']:.4f} | {r['max']:.4f} | "
                     f"{r.get('baseline_climatology', float('nan')):.4f} | {r.get('baseline_persistence', float('nan')):.4f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", required=True, type=Path)
    args = parser.parse_args()
    scores = first.read_csv(args.delivery / "confirmation_scores.csv")
    blocks = first.read_csv(args.delivery / "confirmation_block_scores.csv")
    sums = first.read_csv(args.delivery / "confirmation_block_channel_sums.csv.gz")
    baselines = first.read_csv(args.delivery / "baseline_scores.csv")
    record_path = args.delivery / "RUN_RECORD.json"
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {}
    results = analyse(scores, blocks, sums, "empirical") + analyse(scores, blocks, sums, "fair")
    arms = first.arm_table(scores, baselines)
    with (args.delivery / "confirmation_analysis.json").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"contrasts": results, "arms": arms}, indent=2) + "\n")
    with (args.delivery / "CONFIRMATION_ANALYSIS.md").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(markdown(results, arms, record.get("missing", [])))
    print("analysis written:", len(results), "contrast rows")


if __name__ == "__main__":
    main()
