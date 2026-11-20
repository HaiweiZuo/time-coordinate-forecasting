# Confirmation analysis (protocol section 6)

Paired relative effect T = 1 - CRPS_A / CRPS_B per seed, in percent; positive means A is better.
Only H2 is confirmatory (DEVIATIONS.md, D3). Intervals are 95% block-bootstrap percentile intervals.

## Primary metric: empirical CRPS

| Roster | Family | Contrast | Label | Positive seeds | Median | Min | Max | Sign test p | Bootstrap 95% (registered) | Bootstrap 95% (pooled) |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| ushcn_test | conditional_diffusion | SL vs LL | H1 (secondary since D3, descriptive) | 10/10 | +25.4 | +13.9 | +31.0 | 0.0010 | +21.3 to +27.1 | +21.3 to +26.6 |
| ushcn_test | conditional_diffusion | LS vs LL | descriptive | 7/10 | +1.5 | -6.5 | +6.8 | 0.1719 | -0.5 to +2.3 | -0.3 to +2.2 |
| ushcn_test | conditional_diffusion | SS vs LL | descriptive | 10/10 | +28.3 | +24.6 | +34.6 | 0.0010 | +26.0 to +31.9 | +26.3 to +31.7 |
| ushcn_test | conditional_diffusion | SS vs SL | descriptive | 7/10 | +3.5 | -4.8 | +24.1 | 0.1719 | +4.9 to +8.1 | +5.3 to +8.2 |
| ushcn_test | conditional_diffusion | SS vs LS | descriptive | 10/10 | +29.8 | +19.7 | +32.8 | 0.0010 | +25.5 to +31.1 | +25.6 to +31.0 |
| ushcn_test | direct_gaussian | SL vs LL | descriptive | 6/10 | +0.6 | -6.7 | +3.7 | 0.3770 | -0.8 to +0.2 | -0.7 to +0.4 |
| ushcn_test | direct_gaussian | LS vs LL | descriptive | 3/10 | -0.6 | -3.0 | +2.7 | 0.9453 | -1.0 to -0.0 | -0.8 to +0.1 |
| ushcn_test | direct_gaussian | SS vs LL | descriptive | 2/10 | -2.3 | -7.4 | +5.3 | 0.9893 | -3.0 to -1.5 | -2.8 to -1.4 |
| ushcn_test | direct_gaussian | SS vs SL | descriptive | 2/10 | -1.7 | -8.5 | +4.4 | 0.9893 | -2.3 to -1.5 | -2.3 to -1.6 |
| ushcn_test | direct_gaussian | SS vs LS | descriptive | 4/10 | -0.6 | -10.4 | +5.7 | 0.8281 | -2.4 to -1.1 | -2.4 to -1.1 |
| weather2023 | conditional_diffusion | SL vs LL | H2 (confirmatory) | 7/10 | +5.5 | -15.9 | +22.0 | 0.1719 | +3.9 to +7.0 | +3.9 to +7.0 |
| weather2023 | conditional_diffusion | LS vs LL | descriptive | 7/10 | +5.7 | -6.9 | +21.6 | 0.1719 | +6.3 to +9.4 | +6.3 to +9.4 |
| weather2023 | conditional_diffusion | SS vs LL | descriptive | 4/10 | -2.3 | -16.9 | +6.4 | 0.8281 | -4.4 to -0.7 | -4.4 to -0.7 |
| weather2023 | conditional_diffusion | SS vs SL | S2 (descriptive) | 3/10 | -4.7 | -26.6 | +6.1 | 0.9453 | -10.2 to -6.9 | -10.2 to -6.9 |
| weather2023 | conditional_diffusion | SS vs LS | descriptive | 2/10 | -12.8 | -45.7 | +10.4 | 0.9893 | -12.6 to -10.0 | -12.6 to -10.0 |
| weather2023 | direct_gaussian | SL vs LL | S1 (descriptive) | 10/10 | +9.6 | +5.8 | +19.7 | 0.0010 | +8.8 to +13.2 | +8.8 to +13.2 |
| weather2023 | direct_gaussian | LS vs LL | descriptive | 6/10 | +1.3 | -4.0 | +11.2 | 0.3770 | -0.2 to +4.7 | -0.2 to +4.7 |
| weather2023 | direct_gaussian | SS vs LL | descriptive | 10/10 | +4.6 | +0.2 | +15.6 | 0.0010 | +3.0 to +7.8 | +3.0 to +7.8 |
| weather2023 | direct_gaussian | SS vs SL | descriptive | 0/10 | -4.9 | -12.9 | -1.7 | 1.0000 | -8.3 to -4.0 | -8.3 to -4.0 |
| weather2023 | direct_gaussian | SS vs LS | descriptive | 9/10 | +1.8 | -0.3 | +7.8 | 0.0107 | +2.0 to +4.4 | +2.0 to +4.4 |
| weather2020_test | conditional_diffusion | SL vs LL | descriptive | 7/10 | +9.9 | -39.3 | +25.3 | 0.1719 | +3.3 to +6.4 | +3.2 to +6.3 |
| weather2020_test | conditional_diffusion | LS vs LL | descriptive | 8/10 | +9.6 | -9.4 | +30.5 | 0.0547 | +10.0 to +12.7 | +10.3 to +12.8 |
| weather2020_test | conditional_diffusion | SS vs LL | descriptive | 5/10 | -1.4 | -14.1 | +9.9 | 0.6230 | -2.7 to +0.7 | -2.4 to +1.0 |
| weather2020_test | conditional_diffusion | SS vs SL | descriptive | 3/10 | -4.1 | -27.1 | +23.7 | 0.9453 | -8.3 to -4.2 | -8.1 to -3.6 |
| weather2020_test | conditional_diffusion | SS vs LS | descriptive | 2/10 | -13.7 | -58.0 | +17.6 | 0.9893 | -15.2 to -12.8 | -15.0 to -12.7 |
| weather2020_test | direct_gaussian | SL vs LL | descriptive | 9/10 | +10.5 | -6.8 | +21.0 | 0.0107 | +7.6 to +13.5 | +7.9 to +13.5 |
| weather2020_test | direct_gaussian | LS vs LL | descriptive | 6/10 | +4.7 | -15.3 | +15.2 | 0.3770 | -1.6 to +3.7 | -0.2 to +4.0 |
| weather2020_test | direct_gaussian | SS vs LL | descriptive | 8/10 | +7.4 | -4.3 | +18.9 | 0.0547 | +2.7 to +9.4 | +4.0 to +9.8 |
| weather2020_test | direct_gaussian | SS vs SL | descriptive | 1/10 | -3.5 | -14.2 | +8.5 | 0.9990 | -6.8 to -2.6 | -6.2 to -1.9 |
| weather2020_test | direct_gaussian | SS vs LS | descriptive | 6/10 | +3.1 | -4.1 | +16.3 | 0.3770 | +3.3 to +6.7 | +3.4 to +6.6 |

## Sensitivity: fair CRPS

| Roster | Family | Contrast | Label | Positive seeds | Median | Min | Max | Sign test p | Bootstrap 95% (registered) | Bootstrap 95% (pooled) |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| ushcn_test | conditional_diffusion | SL vs LL | H1 (secondary since D3, descriptive) | 10/10 | +25.6 | +13.8 | +31.0 | 0.0010 | +21.4 to +27.2 | +21.4 to +26.7 |
| ushcn_test | conditional_diffusion | LS vs LL | descriptive | 7/10 | +1.8 | -6.1 | +6.8 | 0.1719 | -0.3 to +2.5 | -0.1 to +2.4 |
| ushcn_test | conditional_diffusion | SS vs LL | descriptive | 10/10 | +28.2 | +24.6 | +34.4 | 0.0010 | +26.0 to +32.0 | +26.3 to +31.8 |
| ushcn_test | conditional_diffusion | SS vs SL | descriptive | 7/10 | +3.4 | -4.5 | +23.9 | 0.1719 | +4.7 to +7.9 | +5.1 to +8.0 |
| ushcn_test | conditional_diffusion | SS vs LS | descriptive | 10/10 | +29.8 | +19.6 | +32.4 | 0.0010 | +25.3 to +31.1 | +25.4 to +30.9 |
| ushcn_test | direct_gaussian | SL vs LL | descriptive | 6/10 | +0.6 | -6.5 | +3.6 | 0.3770 | -0.8 to +0.2 | -0.6 to +0.4 |
| ushcn_test | direct_gaussian | LS vs LL | descriptive | 3/10 | -0.6 | -3.0 | +2.7 | 0.9453 | -1.0 to -0.0 | -0.8 to +0.1 |
| ushcn_test | direct_gaussian | SS vs LL | descriptive | 2/10 | -2.2 | -7.3 | +5.2 | 0.9893 | -3.0 to -1.4 | -2.7 to -1.4 |
| ushcn_test | direct_gaussian | SS vs SL | descriptive | 2/10 | -1.8 | -8.4 | +4.3 | 0.9893 | -2.3 to -1.5 | -2.3 to -1.6 |
| ushcn_test | direct_gaussian | SS vs LS | descriptive | 4/10 | -0.5 | -10.2 | +5.6 | 0.8281 | -2.4 to -1.0 | -2.4 to -1.1 |
| weather2023 | conditional_diffusion | SL vs LL | H2 (confirmatory) | 7/10 | +5.4 | -15.9 | +22.2 | 0.1719 | +3.8 to +7.0 | +3.8 to +7.0 |
| weather2023 | conditional_diffusion | LS vs LL | descriptive | 7/10 | +6.0 | -6.8 | +21.8 | 0.1719 | +6.4 to +9.5 | +6.4 to +9.5 |
| weather2023 | conditional_diffusion | SS vs LL | descriptive | 4/10 | -2.3 | -16.8 | +6.3 | 0.8281 | -4.4 to -0.6 | -4.4 to -0.6 |
| weather2023 | conditional_diffusion | SS vs SL | S2 (descriptive) | 3/10 | -4.5 | -26.7 | +6.3 | 0.9453 | -10.1 to -6.7 | -10.1 to -6.7 |
| weather2023 | conditional_diffusion | SS vs LS | descriptive | 2/10 | -13.0 | -45.7 | +10.5 | 0.9893 | -12.7 to -10.0 | -12.7 to -10.0 |
| weather2023 | direct_gaussian | SL vs LL | S1 (descriptive) | 10/10 | +9.6 | +5.8 | +19.7 | 0.0010 | +8.7 to +13.2 | +8.7 to +13.2 |
| weather2023 | direct_gaussian | LS vs LL | descriptive | 6/10 | +1.4 | -3.9 | +11.1 | 0.3770 | -0.3 to +4.7 | -0.3 to +4.7 |
| weather2023 | direct_gaussian | SS vs LL | descriptive | 10/10 | +4.6 | +0.1 | +15.5 | 0.0010 | +2.9 to +7.8 | +2.9 to +7.8 |
| weather2023 | direct_gaussian | SS vs SL | descriptive | 0/10 | -5.0 | -12.9 | -1.7 | 1.0000 | -8.3 to -4.0 | -8.3 to -4.0 |
| weather2023 | direct_gaussian | SS vs LS | descriptive | 9/10 | +1.8 | -0.4 | +7.8 | 0.0107 | +2.0 to +4.4 | +2.0 to +4.4 |
| weather2020_test | conditional_diffusion | SL vs LL | descriptive | 7/10 | +9.8 | -39.7 | +25.4 | 0.1719 | +3.2 to +6.3 | +3.1 to +6.2 |
| weather2020_test | conditional_diffusion | LS vs LL | descriptive | 7/10 | +9.8 | -9.4 | +30.6 | 0.1719 | +10.1 to +12.8 | +10.4 to +12.9 |
| weather2020_test | conditional_diffusion | SS vs LL | descriptive | 5/10 | -1.3 | -13.9 | +10.1 | 0.6230 | -2.6 to +0.8 | -2.3 to +1.1 |
| weather2020_test | conditional_diffusion | SS vs SL | descriptive | 3/10 | -3.9 | -27.1 | +24.1 | 0.9453 | -8.1 to -3.9 | -7.9 to -3.4 |
| weather2020_test | conditional_diffusion | SS vs LS | descriptive | 2/10 | -13.8 | -58.0 | +17.8 | 0.9893 | -15.3 to -12.9 | -15.1 to -12.7 |
| weather2020_test | direct_gaussian | SL vs LL | descriptive | 9/10 | +10.5 | -7.0 | +21.0 | 0.0107 | +7.5 to +13.5 | +7.9 to +13.4 |
| weather2020_test | direct_gaussian | LS vs LL | descriptive | 6/10 | +4.7 | -15.5 | +15.2 | 0.3770 | -1.7 to +3.7 | -0.2 to +4.0 |
| weather2020_test | direct_gaussian | SS vs LL | descriptive | 8/10 | +7.4 | -4.4 | +18.8 | 0.0547 | +2.6 to +9.4 | +3.9 to +9.8 |
| weather2020_test | direct_gaussian | SS vs SL | descriptive | 1/10 | -3.5 | -14.1 | +8.6 | 0.9990 | -6.8 to -2.6 | -6.2 to -1.9 |
| weather2020_test | direct_gaussian | SS vs LS | descriptive | 6/10 | +3.1 | -4.3 | +16.3 | 0.3770 | +3.3 to +6.8 | +3.4 to +6.7 |

**H2 decision by the registered rule: not confirmed** (7 of 10 seeds positive; per-seed effects in percent: -11.6, +12.4, +3.9, +3.9, +22.0, -15.9, -1.7, +20.0, +7.2, +9.5).

## Equal-stratum empirical CRPS by arm, with reference baselines

| Roster | Family | Arm | Seeds | Median | Min | Max | Climatology | Persistence |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ushcn_test | conditional_diffusion | LL | 10 | 0.4141 | 0.3881 | 0.4333 | 0.3186 | 0.5934 |
| ushcn_test | conditional_diffusion | SL | 10 | 0.3049 | 0.2853 | 0.3720 | 0.3186 | 0.5934 |
| ushcn_test | conditional_diffusion | LS | 10 | 0.4078 | 0.3839 | 0.4390 | 0.3186 | 0.5934 |
| ushcn_test | conditional_diffusion | SS | 10 | 0.2939 | 0.2724 | 0.3139 | 0.3186 | 0.5934 |
| ushcn_test | direct_gaussian | LL | 10 | 0.3912 | 0.3766 | 0.4129 | 0.3186 | 0.5934 |
| ushcn_test | direct_gaussian | SL | 10 | 0.3971 | 0.3735 | 0.4080 | 0.3186 | 0.5934 |
| ushcn_test | direct_gaussian | LS | 10 | 0.3956 | 0.3769 | 0.4138 | 0.3186 | 0.5934 |
| ushcn_test | direct_gaussian | SS | 10 | 0.4022 | 0.3887 | 0.4160 | 0.3186 | 0.5934 |
| weather2023 | conditional_diffusion | LL | 10 | 0.3926 | 0.3543 | 0.4443 | 0.5320 | 0.3278 |
| weather2023 | conditional_diffusion | SL | 10 | 0.3709 | 0.3270 | 0.4531 | 0.5320 | 0.3278 |
| weather2023 | conditional_diffusion | LS | 10 | 0.3558 | 0.3152 | 0.4534 | 0.5320 | 0.3278 |
| weather2023 | conditional_diffusion | SS | 10 | 0.3999 | 0.3672 | 0.4592 | 0.5320 | 0.3278 |
| weather2023 | direct_gaussian | LL | 10 | 0.3013 | 0.2932 | 0.3311 | 0.5320 | 0.3278 |
| weather2023 | direct_gaussian | SL | 10 | 0.2734 | 0.2642 | 0.2812 | 0.5320 | 0.3278 |
| weather2023 | direct_gaussian | LS | 10 | 0.2984 | 0.2844 | 0.3101 | 0.5320 | 0.3278 |
| weather2023 | direct_gaussian | SS | 10 | 0.2884 | 0.2795 | 0.2983 | 0.5320 | 0.3278 |
| weather2020_test | conditional_diffusion | LL | 10 | 0.3523 | 0.2803 | 0.4299 | 0.4182 | 0.2308 |
| weather2020_test | conditional_diffusion | SL | 10 | 0.3277 | 0.2917 | 0.4783 | 0.4182 | 0.2308 |
| weather2020_test | conditional_diffusion | LS | 10 | 0.3044 | 0.2596 | 0.4344 | 0.4182 | 0.2308 |
| weather2020_test | conditional_diffusion | SS | 10 | 0.3620 | 0.3197 | 0.4153 | 0.4182 | 0.2308 |
| weather2020_test | direct_gaussian | LL | 10 | 0.2674 | 0.2380 | 0.2881 | 0.4182 | 0.2308 |
| weather2020_test | direct_gaussian | SL | 10 | 0.2328 | 0.2206 | 0.2543 | 0.4182 | 0.2308 |
| weather2020_test | direct_gaussian | LS | 10 | 0.2555 | 0.2334 | 0.2847 | 0.4182 | 0.2308 |
| weather2020_test | direct_gaussian | SS | 10 | 0.2466 | 0.2327 | 0.2582 | 0.4182 | 0.2308 |
