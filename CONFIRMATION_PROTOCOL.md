# Confirmation protocol: shared time embedding (SL) versus separate axes (LL)

Status: DRAFT until committed. Commit this file to the project repository and
record its SHA-256 **before** any confirmatory training or scoring starts.
After that, do not edit it. Log every deviation in `DEVIATIONS.md` with the
time, the reason, and what was done instead.

Arm names follow the repository: first letter = time-embedding coordinates,
second letter = interpolation/support coordinates; L = separate, S = shared.


## 1. What is already known, and why it is only exploratory

The 48 existing fits (seeds 2024, 2025, 2026) show that SL has lower CRPS than
LL for conditional diffusion in every seed, on the USHCN and Weather 2020
development rosters and on Weather 2022 with the models fixed. That contrast
was chosen after the first results had been seen, so it is a hypothesis, not a
confirmed result. This protocol tests it on data and seeds that played no part
in forming it.


## 2. Hypotheses

Primary (confirmatory), conditional diffusion only:

- **H1.** On the USHCN test role, CRPS(SL) < CRPS(LL).
- **H2.** On Weather 2023 records, CRPS(SL) < CRPS(LL).

Secondary (reported with the same statistics, labelled descriptive, no
confirmatory claim):

- S1. Direct Gaussian on Weather: SL versus LL.
- S2. Conditional diffusion on Weather: SS versus SL (does the combined change
  lose the gain?).
- S3. LS versus LL, both families, both datasets.
- S4. Direct Gaussian on USHCN: all arms. Known issue, recorded here in
  advance: under the frozen recipe these fits reach their early-stopping
  minimum in epochs 1-2, so they are reported but support no claim.
- S5. All contrasts on the Weather 2020 test role.


## 3. New fits

- Seeds: **2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036**
  (ten new seeds). Seeds 2024-2026 are not part of the confirmatory test.
- Arms: LL, SL, LS, SS. Datasets: `weather_jena2020`, `ushcn`.
  Families: `conditional_diffusion`, `direct_gaussian`. Total: 160 fits.
- Code: the commit that produced release v1.0.1. `configs/study.json` is used
  unchanged: AdamW, learning rate 1e-4, betas (0.9, 0.99), weight decay 1e-4,
  gradient clip 1, batch size 32, at most 200 epochs, patience 15,
  early-stopping noise seed 314159, the same fitting and early-stopping
  rosters, the same architecture.
- A fit is rerun only if the process crashes; rerun the same seed and log it.
  No checkpoint is reselected, extended or replaced.
- New code may be added for data preparation, orchestration and score export
  only, in new files. Model, loss, training and scoring code are not edited.
  Record the diff against the v1.0.1 commit in `RUN_RECORD.json`.
- If compute has to be cut, run in this order: diffusion LL and SL (both
  datasets), then diffusion SS, then Gaussian LL and SL, then the rest.

Estimated cost from the recorded cell times of the original study: about
17 GPU-hours for all 160 fits, plus scoring.


## 4. Confirmatory data

None of these records may be loaded, plotted or summarised before this file is
frozen. Prepare them by script and do not inspect the values.

- **USHCN test role.** The inherited records that are in neither the training
  role (668) nor the validation role (222). Build the roster with the same
  rule as the development roster: common anchor stride 6, an anchor qualifies
  only if all four strata can be built, shared digest ordering, cap 128.
  Use every qualifying anchor up to the cap and record the realized count.
  The development minimum of 96 is not enforced here: the roster is fixed by
  the data and must not be adjusted after looking at it.
- **Weather 2023.** Official roof-station archives for 2023 from the same
  MPI-BGC source, prepared exactly as the frozen 2022 temporal protocol: 2020
  feature order, unchanged 2020 training scaler, one anchor per 792-index
  block, all four schedules, no exclusion other than eligibility.
  Preconditions: (a) the project records show no earlier use of 2023 data;
  (b) all 21 numeric headers match the 2020 archives. If either fails, use
  2024 instead and log it.
- **Weather 2020 test role** (secondary, S5). Same roster rule as the 2020
  development roster (stride 36, cap 100), applied to the unused remainder of
  the 2020 lattice. Use every qualifying anchor up to the cap and record the
  realized count; no minimum is enforced.

Before any training starts, report three things and stop only if one fails:
the commit and SHA-256 of this file; whether the 2023 archives pass
preconditions (a) and (b); and the realized anchor and block counts of the
three rosters (counts only, no values).


## 5. Scoring

- One scoring call per checkpoint per roster. No rescoring after results are
  seen.
- 100 trajectories, sampling seed 271828, 50-step deterministic DDIM, chunks
  of 10, clean-target clip at +/-5; Gaussian draws from fitted marginals.
- Primary metric: empirical marginal CRPS, channel-macro within stratum, then
  equal weight over the four strata (unchanged from the original study).
- Sensitivity: fair CRPS with the same aggregation.


## 6. Analysis

For each hypothesis and each seed s, compute the paired relative effect

    T_s = 1 - CRPS_SL,s / CRPS_LL,s        (positive = SL better)

- **Decision rule.** A hypothesis is confirmed if T_s > 0 in at least 9 of the
  10 seeds. Under no effect this happens with probability 11/1024 = 0.0107
  (exact one-sided sign test), below 0.025, so the two hypotheses together
  stay under 0.05.
- Always report the median, minimum and maximum of T_s and the ten values.
- **Data uncertainty.** Paired block bootstrap over source-time blocks: 2,000
  resamples, bootstrap seed 20261120; in each resample recompute the relative
  effect of the seed-averaged block scores; report the 95% percentile
  interval. This interval is reported next to the decision; it is not part of
  the decision rule.
- Secondary contrasts use the same statistics and are labelled descriptive.

Reference baselines, no training, same rosters, masks and aggregation:

- Climatology: for each channel, 100 draws (seed 271828) from the
  training-split standardized observations of that channel.
- Persistence: the last observed value of each channel in the history window
  as a point forecast (CRPS equals absolute error); if a channel has no
  observation in the window, use 0, the training mean on the standardized
  scale.


## 7. Reporting commitments

- Every outcome is reported whichever way it falls. If H1 or H2 is not
  confirmed, the abstract says so.
- Results for seeds 2024-2026 are shown separately and are not pooled into
  the confirmatory test.
- No threshold, exclusion or metric is added after results are seen.
- The earlier ten-condition adequacy rule and its failed outcome stay in the
  paper as the record of the first study.


## 8. Files to deliver

1. `confirmation_scores.csv` with columns: dataset, roster, family, arm, seed,
   stratum, crps_empirical, crps_fair, n_scored, selected_epoch, stop_reason.
2. `confirmation_block_scores.csv`: the same scores per source-time block
   (needed for the bootstrap).
3. `baseline_scores.csv`: climatology and persistence per roster and stratum.
4. `RUN_RECORD.json`: code commit, SHA-256 of this file, hardware, start and
   end times, and the list of any reruns.


## Part B. Source verification (no training)

The paper will state where the separate-axis construction occurs in public
code. Record the following verbatim, with commands and printed output.

Repository: `Ladbaby/PyOmniTS`, commit `50a8b59bf922` (2026-10-11).

- **B1. Fallback path.** Instantiate one affected model (for example
  HyperIMTS or tPatchGNN) and call `forward` with `x_mark=None` and
  `y_mark=None`. Print the marks the model builds. Expected from the source
  (`models/HyperIMTS.py` lines 70-78): `arange(L)/L` for the history and
  `arange(P)/P` for the query. Then list every file that contains this
  fallback, and state for each shipped dataset loader whether it supplies
  marks, i.e. whether the fallback is ever reached in a shipped run.
- **B2. Calendar path.** Run the shipped loader for a regular dataset with
  the settings of `scripts/HyperIMTS/Weather.sh` (defaults `--embed timeF`,
  `--freq h`). For one batch print `x_mark[0, :20, 0]` and `y_mark[0, :20, 0]`
  at the point where the model slices `x_mark[:, :, :1]`. Expected from the
  source: hour of day scaled as `hour/23 - 0.5`, so six consecutive
  10-minute samples share one value and the value resets at midnight. Report
  what is actually printed, including if this expectation is wrong.
- **B3. Origin of the study's own interface.** Identify the file and commit
  from which the study's `legacy` marks (`arange(H)/H`, `arange(P)/P` in
  `src/geometry_screen/data.py`, function `encode_time_labels`) were derived,
  and quote the original lines.
- **B4. Wider scan.** For each repository below, record the commit, the lines
  where history and query times are scaled, and whether one shared scale is
  used: t-PatchGNN, latent_ode, mTAN, neural-flows-experiments,
  Continuous-Recurrent-Units, ProFITi, PhysioPro (ContiFormer), KAFNet, plus
  the three already audited (CSDI, GraFITi, APN). Report repositories that do
  not show the construction as plainly as those that do.
