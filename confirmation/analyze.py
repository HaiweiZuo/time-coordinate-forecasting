#!/usr/bin/env python
"""Analysis of protocol section 6 from the exported delivery files.

Reads confirmation_scores.csv, confirmation_block_scores.csv,
confirmation_block_channel_sums.csv.gz and baseline_scores.csv. No model,
checkpoint or observation is loaded.

For a contrast "A versus B" and seed s the paired relative effect is
T_s = 1 - CRPS_A,s / CRPS_B,s on the equal-stratum empirical CRPS (positive
means A is better). The confirmatory rule applies to H2 only (DEVIATIONS.md,
D3): confirmed if T_s > 0 in at least 9 of the 10 seeds. Every other contrast
gets the same statistics and is labelled descriptive.

Block bootstrap (2,000 resamples, seed 20261120, the same resampled blocks
for every contrast of a roster). Registered statistic: block scores are
averaged over seeds, averaged over the resampled blocks within each stratum,
averaged over the four strata, and the relative effect is recomputed.
Sensitivity statistic: the registered metric is re-aggregated exactly from
per-channel sums of the resampled blocks (channel-macro within stratum, then
equal strata). Both 95% percentile intervals are reported; neither is part of
the decision rule.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
from math import comb
from pathlib import Path

import numpy as np

SEEDS = tuple(range(2027, 2037))
ARMS = ("LL", "SL", "LS", "SS")
CONTRASTS = (("SL", "LL"), ("LS", "LL"), ("SS", "LL"), ("SS", "SL"), ("SS", "LS"))
ROSTERS = (("ushcn", "ushcn_test"), ("weather_jena2020", "weather2023"), ("weather_jena2020", "weather2020_test"))
FAMILIES = ("conditional_diffusion", "direct_gaussian")
BOOTSTRAP_SEED, RESAMPLES = 20261120, 2000
LABELS = {
    ("weather2023", "conditional_diffusion", "SL", "LL"): "H2 (confirmatory)",
    ("ushcn_test", "conditional_diffusion", "SL", "LL"): "H1 (secondary since D3, descriptive)",
    ("weather2023", "direct_gaussian", "SL", "LL"): "S1 (descriptive)",
    ("weather2023", "conditional_diffusion", "SS", "SL"): "S2 (descriptive)",
}


def read_csv(path: Path) -> list[dict]:
    raw = gzip.open(path, "rb").read() if path.suffix == ".gz" else path.read_bytes()
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))


def sign_test_p(positive: int, total: int) -> float:
    """Exact one-sided probability of at least `positive` successes under no effect."""
    return sum(comb(total, k) for k in range(positive, total + 1)) / 2 ** total


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def analyse(scores, blocks, sums, metric: str) -> list[dict]:
    column = {"empirical": "crps_empirical", "fair": "crps_fair"}[metric]
    sum_column = {"empirical": "empirical_sum", "fair": "fair_sum"}[metric]
    primary = {(r["roster"], r["family"], r["arm"], int(r["seed"])): float(r[column]) for r in scores if r["stratum"] == "ALL"}
    results = []
    for _, roster in ROSTERS:
        roster_blocks = [r for r in blocks if r["roster"] == roster]
        names = sorted({r["source_block"] for r in roster_blocks})
        strata = sorted({r["stratum"] for r in roster_blocks})
        position = {name: i for i, name in enumerate(names)}
        stratum_index = {name: i for i, name in enumerate(strata)}
        roster_sums = [r for r in sums if r["roster"] == roster]
        channels = sorted({int(r["channel"]) for r in roster_sums})
        channel_index = {c: i for i, c in enumerate(channels)}
        draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(names), size=(RESAMPLES, len(names))) if names else None
        for family in FAMILIES:
            mean_block, mean_sum, counts, available = {}, {}, None, {}
            for arm in ARMS:
                seeds = [s for s in SEEDS if (roster, family, arm, s) in primary]
                available[arm] = seeds
                if not seeds:
                    continue
                table = np.zeros((len(seeds), len(strata), len(names)))
                filled = np.zeros_like(table, dtype=bool)
                for r in roster_blocks:
                    if r["family"] == family and r["arm"] == arm and int(r["seed"]) in seeds:
                        i = (seeds.index(int(r["seed"])), stratum_index[r["stratum"]], position[r["source_block"]])
                        assert not filled[i], f"duplicate block score for {roster} {family} {arm}"
                        table[i], filled[i] = float(r[column]), True
                assert filled.all(), f"incomplete block scores for {roster} {family} {arm}"
                mean_block[arm] = table.mean(0)
                total = np.zeros((len(strata), len(names), len(channels)))
                count = np.zeros_like(total)
                for r in roster_sums:
                    if r["family"] == family and r["arm"] == arm and int(r["seed"]) in seeds:
                        i = (stratum_index[r["stratum"]], position[r["source_block"]], channel_index[int(r["channel"])])
                        total[i] += float(r[sum_column]) / len(seeds)
                        if int(r["seed"]) == seeds[0]:
                            count[i] = int(r["count"])
                mean_sum[arm], counts = total, count

            def registered(arm, index):
                return mean_block[arm][:, index].mean(1).mean()

            def pooled(arm, index):
                per_stratum = []
                for s in range(len(strata)):
                    numerator, denominator = mean_sum[arm][s, index].sum(0), counts[s, index].sum(0)
                    active = denominator > 0
                    per_stratum.append(np.mean(numerator[active] / denominator[active]))
                return float(np.mean(per_stratum))

            for a, b in CONTRASTS:
                seeds = [s for s in available.get(a, []) if s in available.get(b, [])]
                if not seeds:
                    continue
                effects = [1 - primary[(roster, family, a, s)] / primary[(roster, family, b, s)] for s in seeds]
                positive = sum(effect > 0 for effect in effects)
                label = LABELS.get((roster, family, a, b), "descriptive")
                everything = np.arange(len(names))
                entry = {
                    "roster": roster, "family": family, "contrast": f"{a} vs {b}", "label": label, "metric": metric,
                    "seeds": seeds, "effects": effects, "positive": positive, "n": len(seeds),
                    "median": float(np.median(effects)), "min": float(min(effects)), "max": float(max(effects)),
                    "sign_test_p_one_sided": sign_test_p(positive, len(seeds)),
                    "bootstrap_registered": {
                        "point": 1 - registered(a, everything) / registered(b, everything),
                        "interval95": percentile_interval(np.asarray([1 - registered(a, i) / registered(b, i) for i in draws])),
                    },
                    "bootstrap_pooled": {
                        "point": 1 - pooled(a, everything) / pooled(b, everything),
                        "interval95": percentile_interval(np.asarray([1 - pooled(a, i) / pooled(b, i) for i in draws])),
                    },
                    "blocks": len(names),
                }
                if label.startswith("H2") and metric == "empirical":
                    complete = len(seeds) == len(SEEDS)
                    entry["decision"] = ("confirmed" if positive >= 9 else "not confirmed") if complete else "undetermined: seeds missing"
                    entry["rule"] = "T_s > 0 in at least 9 of 10 seeds"
                results.append(entry)
    return results


def arm_table(scores, baselines) -> list[dict]:
    rows = []
    for _, roster in ROSTERS:
        reference = {r["baseline"]: float(r["crps_empirical"]) for r in baselines if r["roster"] == roster and r["stratum"] == "ALL"}
        for family in FAMILIES:
            for arm in ARMS:
                values = [float(r["crps_empirical"]) for r in scores
                          if r["roster"] == roster and r["family"] == family and r["arm"] == arm and r["stratum"] == "ALL"]
                if values:
                    rows.append({"roster": roster, "family": family, "arm": arm, "n": len(values), "median": float(np.median(values)),
                                 "min": float(min(values)), "max": float(max(values)), **{f"baseline_{k}": v for k, v in reference.items()}})
    return rows


def markdown(results, arms, missing) -> str:
    lines = ["# Confirmation analysis (protocol section 6)", "",
             "Paired relative effect T = 1 - CRPS_A / CRPS_B per seed, in percent; positive means A is better.",
             "Only H2 is confirmatory (DEVIATIONS.md, D3). Intervals are 95% block-bootstrap percentile intervals.", ""]
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
        lines += [f"**H2 decision by the registered rule: {decision['decision']}** ({decision['positive']} of {decision['n']} seeds positive; "
                  f"per-seed effects in percent: {', '.join(f'{100 * e:+.1f}' for e in decision['effects'])}).", ""]
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
    scores = read_csv(args.delivery / "confirmation_scores.csv")
    blocks = read_csv(args.delivery / "confirmation_block_scores.csv")
    sums = read_csv(args.delivery / "confirmation_block_channel_sums.csv.gz")
    baselines = read_csv(args.delivery / "baseline_scores.csv")
    record = json.loads((args.delivery / "RUN_RECORD.json").read_text(encoding="utf-8")) if (args.delivery / "RUN_RECORD.json").exists() else {}
    results = analyse(scores, blocks, sums, "empirical") + analyse(scores, blocks, sums, "fair")
    arms = arm_table(scores, baselines)
    (args.delivery / "confirmation_analysis.json").write_text(json.dumps({"contrasts": results, "arms": arms}, indent=2) + "\n", encoding="utf-8")
    (args.delivery / "CONFIRMATION_ANALYSIS.md").write_text(markdown(results, arms, record.get("missing", [])), encoding="utf-8")
    print("analysis written:", len(results), "contrast rows")


if __name__ == "__main__":
    main()
