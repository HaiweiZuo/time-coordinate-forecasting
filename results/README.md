# Numerical results

This directory contains the recorded results and training histories for the coordinate comparison.

The original experiment contains 24 fitted cells. The component experiment adds 24 fits and replays the original checkpoints. The temporal experiment evaluates 24 fixed Weather models on 2022 observations; it does not add training runs.

## Files

- `original_24_aggregates.json` contains the original cell and stratum scores, training histories, partition summaries and source hashes.
- `crossed_48_aggregates.json` contains the four-arm results, component contrasts and checkpoint identities.
- `temporal_24_aggregates.json` contains overall, block-level and month-level results for the fixed-model evaluation.
- `new_24_training_histories.json` contains every recorded epoch from the additional fits.
- `support_mask_training_summary.json` contains support counts, scoring denominators and finite-ensemble sensitivity summaries.
- `original_scientific_contract.json` records the scientific configuration used for the original experiment.
- `training_*.png`, `training_*.svg`, `captions.json` and `original_training_history_table.md` show the original training histories without smoothing.

## Verification

Run the standard-library check from the repository root.

```sh
python results/verify_numerical_release_v2.py --package results
```

The check covers file hashes, component contrasts, seed summaries, stopping histories, selected epochs and update totals. It does not repeat model fitting or score individual prediction trajectories. Checkpoints are distributed separately through the repository release.

Arm letters are ordered by embedding and raster coordinates. L denotes separate-axis coordinates; S denotes shared-scale coordinates. The original archived LL/SS scores remain the primary corners in the component comparisons. Replay and alternative-score diagnostics are retained separately.

Positive relative improvement favors shared-scale coordinates. Component score differences and interaction values use a different sign convention, with their definitions retained in the result fields. An interaction sign alone does not establish predictive improvement.

Training seeds share observations. Blocks and months provide descriptive summaries, not additional independent replications. Query times and availability masks were extracted retrospectively. The 2022 records come from the same Weather station, and their anchor roster differs from the original evaluation.
