# Confirmation run, 20 November 2026

Results of the experiment registered in `CONFIRMATION_PROTOCOL.md` (commit
`bfbd0a6`, SHA-256
`d13cc6b35f126a16d919215a0929a8d1b1da0526550b182251bb17d399cf5243`).
Deviations and the notes fixed before the results existed are in
`DEVIATIONS.md`.

160 fits (seeds 2027-2036, four arms, two datasets, two model families) and
240 scoring calls completed. Every checkpoint was scored once per roster.
There was no rerun and no missing cell.

## Outcome under the registered rule

H2 is the only confirmatory hypothesis (`DEVIATIONS.md`, D3): Weather 2023,
conditional diffusion, SL versus LL. It is confirmed if the paired relative
effect T_s = 1 - CRPS_SL,s / CRPS_LL,s is positive in at least 9 of the 10
seeds.

**H2 is not confirmed.** The effect is positive in 7 of 10 seeds: median
+5.5%, minimum -15.9%, maximum +22.0%, exact one-sided sign test p = 0.172.
Per seed 2027 to 2036, in percent: -11.6, +12.4, +3.9, +3.9, +22.0, -15.9,
-1.7, +20.0, +7.2, +9.5.

Every other contrast is descriptive and is listed in full in
`CONFIRMATION_ANALYSIS.md`.

- H1 (USHCN test role, conditional diffusion, SL versus LL) was made a
  secondary result before any score existed, because that data role had been
  used in an earlier project phase (`DEVIATIONS.md`, D1 and D3). Positive in
  10 of 10 seeds, median +25.4%.
- S1 (Weather 2023, direct Gaussian, SL versus LL): positive in 10 of 10
  seeds, median +9.6%.
- S2 (Weather 2023, conditional diffusion, SS versus SL): positive in 3 of 10
  seeds, median -4.7%.

Reference baselines. On both Weather rosters the median score of every
conditional-diffusion arm is worse than persistence. On the USHCN roster the
LL and LS conditional-diffusion arms and all four direct-Gaussian arms are
worse than climatology.

The block-bootstrap intervals describe resampling of evaluation blocks with
the ten fits of each arm held fixed. They do not include variation between
training seeds and are not part of the decision rule.

## Files

| File | Content |
|---|---|
| `confirmation_scores.csv` | Protocol section 8, file 1. One row per cell, roster and stratum, plus the equal-stratum primary score (`stratum` = `ALL`). |
| `confirmation_block_scores.csv` | Protocol section 8, file 2. The same scores per source-time block. |
| `baseline_scores.csv` | Protocol section 8, file 3. Climatology and persistence per roster and stratum. |
| `RUN_RECORD.json` | Protocol section 8, file 4. Code commit, protocol hash, hardware, software versions, start and end times, reruns, selected epoch and checkpoint hash of every fit. |
| `confirmation_block_channel_sums.csv.gz` | Scored count and CRPS sums per block, stratum and channel, for exact re-aggregation. |
| `baseline_block_scores.csv` | Baseline scores per source-time block. |
| `CONFIRMATION_ANALYSIS.md`, `confirmation_analysis.json` | Output of `confirmation/analyze.py`, committed before any score was read. |
| `SHA256SUMS` | SHA-256 of the files above. |

Checkpoints, prediction samples and observation payloads are not part of this
repository.

## Reproducing the analysis

From the repository root:

```sh
python -m confirmation.analyze --delivery confirmation_results
```

The command reads the CSV files and rewrites the two analysis files. It loads
no model and no observation. The output is numerically identical under NumPy
1.26.4 and NumPy 2.2.6.
