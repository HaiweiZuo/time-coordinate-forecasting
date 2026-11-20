# Deviations and pre-run notes for CONFIRMATION_PROTOCOL.md

Protocol file SHA-256 `d13cc6b35f126a16d919215a0929a8d1b1da0526550b182251bb17d399cf5243`,
committed unchanged in `bfbd0a67f2db1b53c176ad3c2454b6d5b6d66b6d` on 2026-11-20.
Entries are appended in time order (UTC+8). The protocol file itself is not edited.

## 2026-11-20, before any confirmation record was loaded and before any fit

### D1. Two of the three confirmation data sources were opened in an earlier project phase

A search of the project's text records on 2026-11-20 found the following.

- **USHCN test role (224 records).** The same entity split (seed 20260801,
  668/222/224) was frozen in an earlier phase of this project. An earlier
  manuscript reported results on these records for a different model version
  and different comparisons, and states that the model was revised after test
  behaviour had been seen. The project's rescue rules of August 2026 treat
  every previously opened test split as historical motivation only.
- **Weather 2020 test role (last 10,542 lattice rows).** The same rows were
  the frozen test rows of that earlier phase, and results on them were
  reported in the same earlier manuscript.
- **Weather 2023.** No download, preparation, scoring or inspection was
  found. The only earlier contact is an HTTP header request to the two archive
  URLs in September 2026 (size and date, no content). Precondition (a) of
  section 4 is met within the limit stated below.

No coordinate-arm contrast (LL, SL, LS, SS) was computed on any of the three
sources, and none of them was used when the SL-versus-LL contrast was chosen;
that choice used the development rosters only.

Consequence. Section 1 says the test uses data that played no part in forming
the hypothesis. This holds for the contrast, but the USHCN test role and the
Weather 2020 test role are not untouched holdouts: the model family itself was
shaped in a phase that had seen them. The runs are executed exactly as
registered. Any report must state this history next to H1 and S5. H2 (Weather
2023) is the only primary hypothesis evaluated on records with no recorded
earlier use. Seeds 2027-2036 are new for every hypothesis. Decision rule,
rosters, metrics and analysis are unchanged.

Limit of the search: text records in the project folders on the authors'
workstation. Other machines and other people were not covered.

### N1. Implementation choices fixed before any confirmation record was loaded

These are not deviations. They settle points the protocol leaves open.

1. Entry point. `run_experiment.py` accepts seeds 2024-2026 only, so new
   seeds use `confirmation/fit.py`. It calls the unchanged `make_partitions`,
   `build_crossed_model` and `train` and always requires the original fitting
   and early-stopping roster digests and counts.
2. Record identifiers. Weather 2020 test rows use `weather:test` and Weather
   2023 uses `weather:temporal2023`, following the naming of the earlier
   splits. Identifiers enter the anchor digests and the fixed history masks.
3. Rosters. `ushcn_test` and `weather2020_test` call the unchanged
   `_anchor_groups` and `_select` with caps 128 and 100 and no minimum.
   `weather2023` repeats the 2022 temporal rule.
4. Scaling. The original training scaler files are used. The preparation code
   first regenerates the development payloads and requires them to equal the
   original payloads.
5. Scoring. The sampling loop repeats `run_experiment.score`. Per-window sums
   come from the unchanged `diagnostic_metrics.score_terms`; fair CRPS divides
   the pairwise term by S(S-1) instead of S squared. The exported empirical
   CRPS must equal the unchanged `EqualStratumMetrics` aggregate.
6. Block scores. A block score is the channel-macro CRPS within one stratum
   and one source-time block. Per-channel sums and counts are exported too.
7. Climatology. For each channel, 100 values are drawn once with
   `numpy.random.default_rng(271828)`, channels in order, with replacement,
   from the observed standardized training-role values. The same 100 values
   are the ensemble for every target of that channel.
8. Persistence. The history is the window as presented to the model (for
   Weather, after the fixed 50% history deletion).
9. Reruns. A fit or scoring process that exits with an error is rerun with
   the same seed, at most twice, and logged. A completed fit is never repeated
   and a scored checkpoint is never rescored on the same roster.
10. Health check. Before the confirmation fits, one cell with an original
    seed may be fitted and scored on development data to check the software
    environment. It is not part of any analysis.

## 2026-11-20, after data preparation and before the first confirmation fit

### Pre-training report (section 4)

1. Protocol file: commit `bfbd0a67f2db1b53c176ad3c2454b6d5b6d66b6d`, SHA-256
   `d13cc6b35f126a16d919215a0929a8d1b1da0526550b182251bb17d399cf5243`. The branch
   with the protocol, this record and the run code was published before the
   first confirmation fit (GitHub branch creation 2026-11-20T07:18:09Z).
2. Weather 2023 preconditions: (a) no earlier use in the project records, see
   D1; (b) all 21 numeric headers of both 2023 archives equal the 2020
   archives. Both pass, so 2023 is used.
3. Realized rosters (counts only): USHCN test role 94 anchors (94 qualifying,
   cap 128) in 80 source blocks, 376 windows; Weather 2023 67 anchors in 67
   blocks, none excluded, 268 windows; Weather 2020 test role 100 anchors (223
   qualifying, cap 100) in 14 blocks, 400 windows.

The regenerated training and validation payloads of both datasets equal the
original study payloads exactly, and the regenerated scalers equal the
original scalers.

### D2. Software and hardware differ from the original 48 fits

The original fits used NVIDIA RTX 4090 D cards with Python 3.12.6, PyTorch
2.4.1+cu124 and NumPy 1.26.4. The confirmation run uses one NVIDIA RTX PRO
6000 Blackwell card with Python 3.12.3, PyTorch 2.8.0+cu128 and NumPy 1.26.4,
because PyTorch 2.4.1 builds do not support this card. Two processes share
the card. All 160 confirmation fits and all scoring calls use this one
environment, so every paired contrast is within one environment. Results for
seeds 2024-2026 come from the earlier environment and are reported
separately, as registered.

Health check on previously used data (N1, item 10), run before the first
confirmation fit:

- The Weather 2022 roster rebuilt with the confirmation code reproduces the
  archived anchor digest, source-block digest, counts and lattice size.
- The released checkpoint for Weather, SL, conditional diffusion, seed 2024,
  rescored on Weather 2022: empirical CRPS 0.326359 against 0.327440 archived
  (-0.33%), fair CRPS 0.321672 against 0.322752.
- A new fit of the same cell with seed 2024 on development data: fit loss in
  epochs 0-2 was 0.858199, 0.404313, 0.309031 against 0.857971, 0.405246,
  0.309724 archived.

The two environments therefore agree closely but not bit for bit.

### N2. Source location of the 2023 archives

The official download page now links the archives under
`https://www.bgc-jena.mpg.de/wetter/data/`. The files were taken from there on
2026-11-20: `mpi_roof_2023a.zip`, 1,279,553 bytes, SHA-256
`83fc9b310df75b43248f80e602e2edc6876bc197b9f4fcae16fbe72cceff1518`;
`mpi_roof_2023b.zip`, 1,303,267 bytes, SHA-256
`316705ccf21a5daffdedbd477c273ce4f0b93404937da466d3e9443c3fafcaad`. Both sizes
equal the header request of September 2026.

## 2026-11-20 15:53 (UTC+8), fits running, no confirmation score yet

### D3. H1 is reported as a secondary result; H2 is the only confirmatory hypothesis

Because of D1, the authors decided to change the status of H1 (USHCN test
role) from primary to secondary. The decision was taken while the first two
of the 160 fits were still training. No fit had finished, no checkpoint had
been scored on any confirmation roster, and no confirmation score existed.

- H2 (Weather 2023, conditional diffusion, CRPS of SL below CRPS of LL) is
  the single confirmatory hypothesis.
- H1 is computed with the same statistics as registered and reported with
  the secondary results, labelled descriptive, together with the history in
  D1. No confirmatory claim is made from it.
- The decision rule for H2 is unchanged: confirmed if the paired relative
  effect is positive in at least 9 of the 10 seeds. The threshold is not
  relaxed although only one primary hypothesis remains.
- All 160 fits, all rosters, metrics, baselines and reporting commitments
  stay as registered. USHCN results are reported in full whichever way they
  fall.
