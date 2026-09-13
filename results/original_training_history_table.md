# Original R5 training histories

All 24 cells; no retraining or checkpoint replacement. Best epoch is reconstructed from the original strict-minimum rule; checkpoint internal metadata was not loaded here. Epochs below are 1-based.

| Cell | Completed epochs | Recorded best epoch | Best early loss | Last early loss | Optimizer steps | Stop |
|---|---:|---:|---:|---:|---:|---|
| ushcn__geometry__conditional_diffusion__s2024 | 59 | 44 | 0.27907906 | 0.29564927 | 1770 | early_stopping_patience |
| ushcn__legacy__conditional_diffusion__s2024 | 48 | 33 | 0.31371416 | 0.33492764 | 1440 | early_stopping_patience |
| ushcn__geometry__conditional_diffusion__s2025 | 68 | 53 | 0.27622591 | 0.3122362 | 2040 | early_stopping_patience |
| ushcn__legacy__conditional_diffusion__s2025 | 65 | 50 | 0.31001967 | 0.36177453 | 1950 | early_stopping_patience |
| ushcn__geometry__conditional_diffusion__s2026 | 60 | 45 | 0.28394456 | 0.33433081 | 1800 | early_stopping_patience |
| ushcn__legacy__conditional_diffusion__s2026 | 49 | 34 | 0.30872213 | 0.33131841 | 1470 | early_stopping_patience |
| ushcn__geometry__direct_gaussian__s2024 | 17 | 2 | 2.2614091 | 38.708893 | 510 | early_stopping_patience |
| ushcn__legacy__direct_gaussian__s2024 | 17 | 2 | 2.0490192 | 80.768865 | 510 | early_stopping_patience |
| ushcn__geometry__direct_gaussian__s2025 | 17 | 2 | 3.2817129 | 55.898582 | 510 | early_stopping_patience |
| ushcn__legacy__direct_gaussian__s2025 | 16 | 1 | 2.0035183 | 58.380009 | 480 | early_stopping_patience |
| ushcn__geometry__direct_gaussian__s2026 | 16 | 1 | 2.3512854 | 32.064367 | 480 | early_stopping_patience |
| ushcn__legacy__direct_gaussian__s2026 | 16 | 1 | 2.6036882 | 71.301319 | 480 | early_stopping_patience |
| weather_jena2020__geometry__conditional_diffusion__s2024 | 139 | 124 | 0.099983969 | 0.10153201 | 8896 | early_stopping_patience |
| weather_jena2020__legacy__conditional_diffusion__s2024 | 169 | 154 | 0.09377831 | 0.093845607 | 10816 | early_stopping_patience |
| weather_jena2020__geometry__conditional_diffusion__s2025 | 123 | 108 | 0.10129372 | 0.10206078 | 7872 | early_stopping_patience |
| weather_jena2020__legacy__conditional_diffusion__s2025 | 156 | 141 | 0.094558456 | 0.095200505 | 9984 | early_stopping_patience |
| weather_jena2020__geometry__conditional_diffusion__s2026 | 157 | 142 | 0.09944291 | 0.10079835 | 10048 | early_stopping_patience |
| weather_jena2020__legacy__conditional_diffusion__s2026 | 200 | 197 | 0.093041405 | 0.093185982 | 12800 | epoch_cap |
| weather_jena2020__geometry__direct_gaussian__s2024 | 21 | 6 | 0.58656884 | 2.1365444 | 1344 | early_stopping_patience |
| weather_jena2020__legacy__direct_gaussian__s2024 | 21 | 6 | 0.65290609 | 1.8514951 | 1344 | early_stopping_patience |
| weather_jena2020__geometry__direct_gaussian__s2025 | 21 | 6 | 0.60549312 | 1.2513214 | 1344 | early_stopping_patience |
| weather_jena2020__legacy__direct_gaussian__s2025 | 22 | 7 | 0.67052073 | 1.2168061 | 1408 | early_stopping_patience |
| weather_jena2020__geometry__direct_gaussian__s2026 | 21 | 6 | 0.52909449 | 1.6310661 | 1344 | early_stopping_patience |
| weather_jena2020__legacy__direct_gaussian__s2026 | 20 | 5 | 0.64124809 | 1.5325021 | 1280 | early_stopping_patience |
