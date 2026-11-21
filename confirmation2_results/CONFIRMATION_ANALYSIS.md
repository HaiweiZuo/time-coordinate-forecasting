# Second confirmation analysis (CONFIRMATION_PROTOCOL_2.md, section 6)

Paired relative effect T = 1 - CRPS_A / CRPS_B per seed, in percent; positive means A is better.
Only H3 is confirmatory. Intervals are 95% block-bootstrap percentile intervals; they do not include
variation between training seeds and are not part of the decision rule.

## Primary metric: empirical CRPS

| Roster | Family | Contrast | Label | Positive seeds | Median | Min | Max | Sign test p | Bootstrap 95% (registered) | Bootstrap 95% (pooled) |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| weather2024 | direct_gaussian | SL vs LL | H3 (confirmatory) | 10/10 | +10.2 | +5.2 | +17.1 | 0.0010 | +7.4 to +14.3 | +7.4 to +14.3 |
| weather2024 | direct_gaussian | LS vs LL | descriptive | 8/10 | +9.3 | -3.0 | +16.1 | 0.0547 | +4.6 to +10.8 | +4.6 to +10.8 |
| weather2024 | direct_gaussian | SS vs LL | descriptive | 10/10 | +10.0 | +2.7 | +17.5 | 0.0010 | +5.9 to +12.7 | +5.9 to +12.7 |
| weather2024 | direct_gaussian | SS vs SL | descriptive | 5/10 | -1.0 | -7.9 | +3.4 | 0.6230 | -3.3 to +0.2 | -3.3 to +0.2 |
| weather2024 | direct_gaussian | SS vs LS | descriptive | 7/10 | +2.3 | -4.7 | +6.7 | 0.1719 | +0.6 to +3.2 | +0.6 to +3.2 |

## Sensitivity: fair CRPS

| Roster | Family | Contrast | Label | Positive seeds | Median | Min | Max | Sign test p | Bootstrap 95% (registered) | Bootstrap 95% (pooled) |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| weather2024 | direct_gaussian | SL vs LL | H3 (confirmatory) | 10/10 | +10.3 | +5.2 | +17.1 | 0.0010 | +7.4 to +14.3 | +7.4 to +14.3 |
| weather2024 | direct_gaussian | LS vs LL | descriptive | 8/10 | +9.3 | -3.0 | +16.1 | 0.0547 | +4.6 to +10.8 | +4.6 to +10.8 |
| weather2024 | direct_gaussian | SS vs LL | descriptive | 10/10 | +9.9 | +2.7 | +17.5 | 0.0010 | +5.8 to +12.7 | +5.8 to +12.7 |
| weather2024 | direct_gaussian | SS vs SL | descriptive | 5/10 | -1.0 | -7.9 | +3.3 | 0.6230 | -3.3 to +0.2 | -3.3 to +0.2 |
| weather2024 | direct_gaussian | SS vs LS | descriptive | 7/10 | +2.3 | -4.6 | +6.7 | 0.1719 | +0.5 to +3.1 | +0.5 to +3.2 |

**H3 decision by the registered rule: confirmed** (10 of 10 seeds positive; per-seed effects in percent, seeds 2037 to 2046: +11.2, +8.6, +7.6, +7.8, +14.6, +17.1, +5.2, +13.7, +10.6, +9.8).

## Equal-stratum empirical CRPS by arm, with reference baselines

| Roster | Family | Arm | Seeds | Median | Min | Max | Climatology | Persistence |
|---|---|---|---:|---:|---:|---:|---:|---:|
| weather2024 | direct_gaussian | LL | 10 | 0.3255 | 0.2993 | 0.3429 | 0.5125 | 0.3183 |
| weather2024 | direct_gaussian | SL | 10 | 0.2862 | 0.2704 | 0.3049 | 0.5125 | 0.3183 |
| weather2024 | direct_gaussian | LS | 10 | 0.2993 | 0.2801 | 0.3189 | 0.5125 | 0.3183 |
| weather2024 | direct_gaussian | SS | 10 | 0.2924 | 0.2806 | 0.3116 | 0.5125 | 0.3183 |
