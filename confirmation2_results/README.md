# Second confirmation run, 21 November 2026

Results of the experiment registered in `CONFIRMATION_PROTOCOL_2.md` (commit
`2558c34`, SHA-256
`b833989d69d97390c8f7eebb079b2dd60aaeef4aed021826ba56819dfedfcdbb`). The
protocol was written after the results of the first protocol had been
published; see its section 1. Notes fixed before the 2024 data were opened are
in `DEVIATIONS.md` (N5 and the pre-training report).

40 fits (direct Gaussian forecaster, Weather, four arms, seeds 2037-2046) and
40 scoring calls on Weather 2024 completed. Every checkpoint was scored once.
There was no rerun and no missing cell.

## Outcome under the registered rule

H3 is the only hypothesis of this protocol: direct Gaussian forecaster,
Weather 2024, SL versus LL. It is confirmed if the paired relative effect
T_s = 1 - CRPS_SL,s / CRPS_LL,s is positive in at least 9 of the 10 seeds.

**H3 is confirmed.** The effect is positive in 10 of 10 seeds: median +10.2%,
minimum +5.2%, maximum +17.1%, exact one-sided sign test p = 0.001. Per seed
2037 to 2046, in percent: +11.2, +8.6, +7.6, +7.8, +14.6, +17.1, +5.2, +13.7,
+10.6, +9.8.

This does not change the outcome of the first protocol: H2 (conditional
diffusion, Weather 2023, SL versus LL) was not confirmed; see
`confirmation_results/`.

Descriptive contrasts on the same fits (full table in
`CONFIRMATION_ANALYSIS.md`): LS versus LL positive in 8 of 10 seeds (median
+9.3%); SS versus LL in 10 of 10 (median +10.0%); SS versus SL in 5 of 10
(median -1.0%); SS versus LS in 7 of 10 (median +2.3%).

Reference baselines on the same roster: persistence 0.3183, climatology
0.5125. Arm medians: LL 0.3255, SL 0.2862, LS 0.2993, SS 0.2924. Every fit of
every arm is better than climatology. Against persistence: SL and SS are
better in 10 of 10 seeds, LS in 9 of 10, LL in 3 of 10.

All 40 fits stopped by patience. The selected epochs lie between 3 and 12
(epochs run: 19 to 28), as recorded per fit in `RUN_RECORD.json`.

The block-bootstrap intervals describe resampling of evaluation blocks with
the ten fits of each arm held fixed. They do not include variation between
training seeds and are not part of the decision rule.

## Files

| File | Content |
|---|---|
| `confirmation_scores.csv` | Protocol section 8, file 1. One row per cell and stratum, plus the equal-stratum primary score (`stratum` = `ALL`). |
| `confirmation_block_scores.csv` | Protocol section 8, file 2. The same scores per source-time block. |
| `baseline_scores.csv` | Protocol section 8, file 3. Climatology and persistence per stratum. |
| `RUN_RECORD.json` | Protocol section 8, file 4. Code commit, protocol hash, hardware, software versions, start and end times, reruns, selected epoch and checkpoint hash of every fit. Host name and GPU serial identifiers are not recorded. |
| `confirmation_block_channel_sums.csv.gz` | Scored count and CRPS sums per block, stratum and channel, for exact re-aggregation. |
| `baseline_block_scores.csv` | Baseline scores per source-time block. |
| `CONFIRMATION_ANALYSIS.md`, `confirmation_analysis.json` | Output of `confirmation2/analyze.py`, committed before the 2024 data were opened. |
| `SHA256SUMS` | SHA-256 of the files above. |

Checkpoints, prediction samples and observation payloads are not part of this
repository.

## Reproducing the analysis

From the repository root:

```sh
python -m confirmation2.analyze --delivery confirmation2_results
```

The command reads the CSV files and rewrites the two analysis files. It loads
no model and no observation. The output is byte-identical under NumPy 1.26.4
and NumPy 2.2.6.
