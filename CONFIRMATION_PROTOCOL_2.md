# Second confirmation protocol: shared time embedding (SL) versus separate axes (LL), direct Gaussian forecaster, Weather 2024

Status: DRAFT until committed. Commit this file to the project repository and
record its SHA-256 **before** the 2024 archive is downloaded or opened and
before any fit of this protocol starts. After that, do not edit it. Log every
deviation in `DEVIATIONS.md` with the time, the reason, and what was done
instead.

Arm names follow the repository: first letter = time-embedding coordinates,
second letter = interpolation/support coordinates; L = separate, S = shared.

This protocol was written on 21 November 2026, after every result of the
first protocol (`CONFIRMATION_PROTOCOL.md`) had been published in
`confirmation_results/` (commit `b34bf47`).


## 1. What is already known, and why it is only exploratory

The first protocol had one confirmatory hypothesis, H2: conditional diffusion,
Weather 2023, SL versus LL. H2 was not confirmed (positive in 7 of 10 seeds;
the rule requires 9). That result stands and is not revisited here.

In the same run the secondary contrast S1, direct Gaussian forecaster, SL
versus LL, was positive in 10 of 10 seeds on Weather 2023 (median +9.6%,
minimum +5.8%, maximum +19.7%) and in 9 of 10 seeds on the Weather 2020 test
role (median +10.5%). On Weather 2023 the direct Gaussian arms scored better
than both reference baselines (arm medians 0.273 to 0.301; persistence 0.328;
climatology 0.532).

S1 was registered as descriptive, and this protocol was written after its
result was seen. It is therefore a hypothesis, not a confirmed result. This
protocol tests it on a year of data and on training seeds that played no part
in forming it.


## 2. Hypothesis

Primary (confirmatory), one hypothesis:

- **H3.** Direct Gaussian forecaster, Weather 2024 records:
  CRPS(SL) < CRPS(LL).

Secondary (same statistics, labelled descriptive, no confirmatory claim):

- S6. LS versus LL, SS versus LL, SS versus SL and SS versus LS on the same
  fits and the same roster.

No other model family, dataset or roster is part of this protocol.


## 3. New fits

- Seeds: **2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046** (ten
  new seeds). Seeds 2024-2036 are not part of the test.
- Arms: LL, SL, LS, SS. Dataset: `weather_jena2020`. Family:
  `direct_gaussian`. Total: 40 fits.
- Code: model, loss, training and scoring code of release v1.0.1, unchanged.
  `configs/study.json` is used unchanged, with the same fitting and
  early-stopping rosters (the 2020 training and validation slices) as in the
  original study and in the first protocol.
- A fit is rerun only if the process crashes; rerun the same seed and log it.
  No checkpoint is reselected, extended or replaced.
- New code may be added for data preparation, orchestration, score export and
  analysis only, in new files under `confirmation2/`. The files of the first
  confirmation run under `confirmation/` are not edited.
- Order: LL and SL for all seeds first, then LS and SS.
- Machine and software: the machine and environment of the first run, as
  recorded in `confirmation_results/RUN_RECORD.json` (one NVIDIA RTX PRO 6000
  Blackwell, PyTorch 2.8.0+cu128, NumPy 1.26.4). All 40 fits run on that one
  machine. If this is not possible, log it as a deviation before the first
  fit.


## 4. Confirmatory data

No 2024 record may be loaded, plotted or summarised before this file is
frozen. Prepare the records by script and do not inspect the values.

- **Weather 2024.** The official roof-station archive for 2024 from the same
  MPI-BGC source. The source publishes 2024 as one full-year archive,
  `mpi_roof_2024.zip` (2,583,427 bytes by header request; not downloaded or
  opened when this file was written). Prepared exactly as Weather 2023 in the
  first protocol: 2020 feature order, unchanged 2020 training scaler, one
  anchor per 792-index block, all four schedules, no exclusion other than
  eligibility, no replacement anchor.
- Preconditions: (a) the project records show no earlier use of 2024 data;
  (b) all 21 numeric headers match the 2020 archives; (c) every timestamp
  lies between 2024-01-01 00:00 and 2025-01-01 00:00. If one fails, use the
  2025 archive (`mpi_roof_2025.zip`) under the same rules and log it. If 2025
  fails as well, stop.
- The reader for a single full-year archive is new code. Before it is applied
  to 2024 it must reproduce the Weather 2023 payload of the first run bit for
  bit from the two 2023 archives merged into one file. The 2023 records have
  already been scored, so this check uses no new information.

Before any fit starts, report three things and stop if one fails: the commit
and SHA-256 of this file; whether the 2024 archive passes preconditions (a),
(b) and (c); and the realized anchor and block counts of the roster (counts
only, no values).


## 5. Scoring

- One scoring call per checkpoint on the Weather 2024 roster. No rescoring
  after results are seen.
- The 40 checkpoints of this protocol are scored on no other roster. No
  checkpoint from an earlier run is scored on 2024 data.
- 100 draws from the fitted Gaussian marginals, sampling seed 271828, the
  same evaluation batch size and precision as in the first protocol.
- Primary metric: empirical marginal CRPS, channel-macro within stratum, then
  equal weight over the four strata (unchanged).
- Sensitivity: fair CRPS with the same aggregation.


## 6. Analysis

For each seed s, compute the paired relative effect

    T_s = 1 - CRPS_SL,s / CRPS_LL,s        (positive = SL better)

- **Decision rule.** H3 is confirmed if T_s > 0 in at least 9 of the 10
  seeds. Under no effect this happens with probability 11/1024 = 0.0107
  (exact one-sided sign test). H3 is the only hypothesis tested.
- Always report the median, minimum and maximum of T_s and the ten values.
- **Data uncertainty.** Paired block bootstrap over source-time blocks, as in
  the first protocol and its note N3 in `DEVIATIONS.md`: 2,000 resamples,
  bootstrap seed 20261120; the registered statistic (relative effect of the
  seed-averaged block scores) and the pooled sensitivity statistic; 95%
  percentile intervals. The intervals are reported next to the decision.
  They do not include variation between training seeds and are not part of
  the decision rule.
- Secondary contrasts use the same statistics and are labelled descriptive.
- The statistics are computed by `confirmation/analyze.py` as committed for
  the first protocol. The analysis code for this protocol is committed before
  any score is read.

Reference baselines, no training, same roster, masks and aggregation, with
the definitions of the first protocol:

- Climatology: for each channel, 100 draws (seed 271828) from the
  training-split standardized observations of that channel.
- Persistence: the last observed value of each channel in the history window
  as a point forecast; if a channel has no observation in the window, use 0.

The baselines are reported next to the arm scores. They are not part of the
decision rule. The paper states for each arm whether its median score is
better or worse than each baseline.


## 7. Reporting commitments

- The outcome is reported whichever way it falls. If H3 is not confirmed, the
  abstract says so.
- The outcome of H3 does not change the report of the first protocol: the
  abstract states that H2 was not confirmed.
- Results of seeds 2024-2026 and of the first protocol are shown separately
  and are not pooled into the test of H3.
- No threshold, exclusion or metric is added after results are seen.
- This is the last confirmation attempt for the SL-versus-LL contrast in this
  study. If H3 is not confirmed, no further round is registered.


## 8. Files to deliver

In `confirmation2_results/`:

1. `confirmation_scores.csv` with columns: dataset, roster, family, arm, seed,
   stratum, crps_empirical, crps_fair, n_scored, selected_epoch, stop_reason.
2. `confirmation_block_scores.csv`: the same scores per source-time block.
3. `baseline_scores.csv`: climatology and persistence per stratum.
4. `RUN_RECORD.json`: code commit, SHA-256 of this file, hardware, software
   versions, start and end times, and the list of any reruns.
