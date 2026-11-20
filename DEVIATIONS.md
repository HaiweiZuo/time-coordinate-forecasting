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
