#!/usr/bin/env python
"""Synthetic checks of the confirmation code. No study data, no GPU, no training."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from confirmation.common import CONFIG, TEMPORAL_RECORD_ID, WEATHER_TEST_RECORD_ID

import numpy as np
import torch

from confirmation import baselines, export, rosters
from confirmation.orchestrate import ordered_cells
from confirmation.score import stratum_summary
from diagnostic_metrics import score_terms
from geometry_screen import data
from geometry_screen.metrics import EqualStratumMetrics
from training import make_partitions


def synthetic_weather(record_id: str, rows: int, origin: int = 0, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(rows, 21, generator=generator)
    mask = torch.rand(rows, 21, generator=generator) > 0.05
    return record_id, torch.arange(origin, origin + rows, dtype=torch.float32), torch.where(mask, values, 0.0), mask


def synthetic_ushcn(count: int, seed: int = 0, prefix: str = "S"):
    generator = torch.Generator().manual_seed(seed)
    records = []
    for index in range(count):
        length = 150
        times = torch.cumsum(torch.rand(length, generator=generator) * 1.2 + 0.2, 0).to(torch.float32)
        values = torch.randn(length, 5, generator=generator)
        mask = torch.rand(length, 5, generator=generator) > 0.4
        mask[:, 0] = True
        records.append((f"{prefix}{index:04d}", times, torch.where(mask, values, 0.0), mask))
    return records


def check_rosters() -> None:
    # Capped Weather roster: identical to the development rule applied to the same record.
    payload = synthetic_weather(WEATHER_TEST_RECORD_ID, 792 * 9 + 300, origin=42162)
    bundles = {arm: rosters.build_roster("weather2020_test", payload, arm) for arm in CONFIG["coordinate_arms"]}
    rosters.paired_across_arms(bundles)
    proof = bundles["LL"].proof
    assert proof["realized_anchor_count"] == 100 <= proof["qualifying_anchor_count"]
    assert set(proof["partition_counts"]["score"].values()) == {100}
    settings = CONFIG["datasets"]["weather_jena2020"]
    specs = data._validate_registered_strata(settings["strata"], "weather_jena2020")
    groups = data._anchor_groups(data.validate_payload(payload, "weather_jena2020"), dataset="weather_jena2020", specs=specs,
                                 mode="legacy", block_span=792, missing_rate=0.5, mask_seed=99173)
    assert proof["anonymous_roster_digest"]["score"] == data._roster_digest(sorted(groups)[:100])
    batch = data.collate_windows(bundles["SL"].dev_score.samples[:4])
    assert batch["x_mark"].shape == (4, 72, 2) and batch["y_mark"].shape == (4, 18)
    for left, right in zip(bundles["LL"].dev_score.samples, bundles["SL"].dev_score.samples):
        assert torch.equal(left["x_mark"][:, 1], right["x_mark"][:, 1]) and not torch.equal(left["y_mark"], right["y_mark"])
    for left, right in zip(bundles["LL"].dev_score.samples, bundles["LS"].dev_score.samples):
        assert torch.equal(left["x_mark"][:, 0], right["x_mark"][:, 0]) and torch.equal(left["y_mark"], right["y_mark"])
    # The development builder gives the same windows when the same record is passed as validation data.
    train = synthetic_weather("weather:train", 792 * 60, seed=1)
    reference = make_partitions(train, payload, "weather_jena2020", CONFIG, "SL").dev_score
    for left, right in zip(reference.samples, bundles["SL"].dev_score.samples):
        assert left["sample_id"] == right["sample_id"]
        for key in ("x", "y", "x_mask", "y_mask", "x_mark", "y_mark"):
            assert torch.equal(left[key], right[key])
    # USHCN roster without a minimum: fewer anchors than the cap are all used.
    few = rosters.build_roster("ushcn_test", synthetic_ushcn(6, seed=3), "SS")
    assert 0 < few.proof["realized_anchor_count"] == few.proof["qualifying_anchor_count"] < 128
    many = rosters.build_roster("ushcn_test", synthetic_ushcn(80, seed=4), "LL")
    assert many.proof["realized_anchor_count"] == 128 < many.proof["qualifying_anchor_count"]
    # Temporal roster: one block-origin anchor, all four schedules or none, no replacement.
    length = 792 * 3 + 100
    values = torch.arange(length * 21, dtype=torch.float32).reshape(length, 21) / 100
    mask = torch.ones_like(values, dtype=torch.bool)
    mask[792 + 90:792 + 108] = False
    temporal = TEMPORAL_RECORD_ID, torch.arange(length, dtype=torch.float32), values, mask
    metadata = {"lattice_rows": length, "first_timestamp": "2023-01-01T00:00:00"}
    arms = {arm: rosters.build_roster("weather2023", temporal, arm, metadata) for arm in CONFIG["coordinate_arms"]}
    rosters.paired_across_arms(arms)
    assert len(arms["LL"].dev_score) == 8 and arms["LL"].proof["excluded_block_count"] == 2
    assert arms["LL"].proof["source_block_count"]["score"] == 2
    print("PASS rosters: cap and digest order, no minimum, E/R crossing, equality with the development builder, temporal all-or-none")


def check_scores() -> None:
    generator = np.random.default_rng(7)
    strata = ("a", "b", "c", "d")
    rows, accumulator = [], EqualStratumMetrics(strata, 100, 0.95)
    for index, name in enumerate(strata * 6):
        samples = generator.normal(size=(1, 100, 6, 5))
        target = generator.normal(size=(1, 6, 5))
        mask = generator.random((1, 6, 5)) > 0.3
        mask[0, :2] = True
        accumulator.update(name, samples, target, mask)
        terms = score_terms(samples, target, mask)
        rows.append({"stratum": name, "source_block_digest": f"{index % 5:064d}", **{key: value[0] for key, value in terms.items()}})
    official = accumulator.finalize()
    summary = stratum_summary(rows)
    assert abs(official["primary_equal_stratum"]["marginal_crps"] - summary["equal_stratum"]["crps_empirical"]) < 1e-12
    for name in strata:
        assert abs(official["strata"][name]["marginal_crps"] - summary["strata"][name]["crps_empirical"]) < 1e-12
        assert summary["strata"][name]["crps_fair"] < summary["strata"][name]["crps_empirical"]
    block_scores, block_sums = export.block_tables(rows)
    for name in strata:
        selected = [row for row in block_sums if row["stratum"] == name]
        channels = sorted({row["channel"] for row in selected})
        pooled = np.mean([sum(r["empirical_sum"] for r in selected if r["channel"] == c) / sum(r["count"] for r in selected if r["channel"] == c) for c in channels])
        assert abs(pooled - summary["strata"][name]["crps_empirical"]) < 1e-12
    assert sum(row["n_scored"] for row in block_scores) == summary["equal_stratum"]["n_scored"]
    print("PASS scores: export equals the unchanged study aggregate; block sums re-aggregate exactly")


def check_baselines() -> None:
    train = synthetic_ushcn(30, seed=5, prefix="T")
    first = baselines.climatology_ensemble(train, "ushcn")
    assert first.shape == (100, 5) and np.array_equal(first, baselines.climatology_ensemble(train, "ushcn"))
    windows = rosters.build_roster("ushcn_test", synthetic_ushcn(40, seed=6), "LL").dev_score
    climatology = baselines.climatology_rows(windows, first)
    persistence = baselines.persistence_rows(windows)
    assert len(climatology) == len(persistence) == len(windows)
    sample, row = windows.samples[0], persistence[0]
    history, observed = sample["x"].numpy(), sample["x_mask"].numpy()
    forecast = np.asarray([history[np.flatnonzero(observed[:, c])[-1], c] if observed[:, c].any() else 0.0 for c in range(5)])
    expected = (np.abs(sample["y"].numpy() - forecast) * sample["y_mask"].numpy()).sum(0)
    assert np.allclose(row["empirical_crps_channel_sum"], expected)
    brute = np.abs(first[:, None, :] - sample["y"].numpy()[None]).mean(0) - 0.5 * np.abs(first[:, None] - first[None]).mean((0, 1))
    assert np.allclose(climatology[0]["empirical_crps_channel_sum"], (brute * sample["y_mask"].numpy()).sum(0))
    stratum_summary(climatology), stratum_summary(persistence)
    print("PASS baselines: fixed climatology draws, last-observation persistence, brute-force CRPS identities")


def check_matrix() -> None:
    cells = ordered_cells()
    assert len(cells) == 160 and len(set(cells)) == 160
    assert all(family == "conditional_diffusion" and arm in ("LL", "SL") for _, family, arm, _ in cells[:40])
    assert all(family == "conditional_diffusion" and arm == "SS" for _, family, arm, _ in cells[40:60])
    assert all(family == "direct_gaussian" and arm in ("LL", "SL") for _, family, arm, _ in cells[60:100])
    assert {seed for *_, seed in cells} == set(range(2027, 2037))
    print("PASS matrix: 160 registered cells in the protocol order")


if __name__ == "__main__":
    check_rosters()
    check_scores()
    check_baselines()
    check_matrix()
