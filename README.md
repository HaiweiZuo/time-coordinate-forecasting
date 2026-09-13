# Time-coordinate forecasting

Research materials for *Time-Coordinate Collisions and Model-Dependent Effects in Probabilistic Forecasting*.

The study compares separate and shared time coordinates in conditional diffusion and direct Gaussian prediction. The repository contains portable training and preprocessing code, archived numerical results, and a registry of 48 retained checkpoints. Checkpoint archives are distributed through [release v1.0.0](https://github.com/HaiweiZuo/time-coordinate-forecasting/releases/tag/v1.0.0), outside Git history.

## Coordinate arms

The first letter specifies time-embedding coordinates. The second specifies history interpolation and support coordinates.

| Arm | Time embedding | Interpolation and support |
| --- | --- | --- |
| LL | Separate | Separate |
| LS | Separate | Shared |
| SL | Shared | Separate |
| SS | Shared | Shared |

The original comparison contains 24 LL/SS fits. The crossed study adds 24 LS/SL fits across two datasets, two model families, and seeds 2024, 2025, and 2026. The Weather 2022 analysis reuses 24 retained Weather checkpoints without fitting or checkpoint selection.

## Installation

The recorded training environment used Python 3.12.6, NumPy 1.26.4, and PyTorch 2.4.1 with CUDA 12.4. Use an isolated Python environment and install the dependencies.

```sh
python -m pip install -r requirements.txt
```

For CUDA, select the matching PyTorch wheel using the [official previous-version instructions](https://pytorch.org/get-started/previous-versions/#v241). The command-line interface defaults to CPU. CUDA execution uses bfloat16 and requires a compatible device.

## Source data

Observations are obtained separately from their upstream providers. [DATA.md](DATA.md) records the archives, processed-data source, checksums, access terms, and preparation rules. No observation payloads are included in this repository or its checkpoint archives.

After obtaining the exact source files, prepare the development inputs.

```sh
python prepare_inputs.py weather2020 --weather-a data/mpi_roof_2020a.zip --weather-b data/mpi_roof_2020b.zip --out data/prepared
python prepare_inputs.py ushcn --ushcn data/ushcn.pt --out data/prepared
```

Each command produces training and validation payloads, a training-only scaler, and a preparation receipt. Existing output directories are refused. The utility does not export the reserved remainder of either dataset.

## Checkpoints

Download both archives from the release page. With the GitHub CLI, the equivalent command is

```sh
gh release download v1.0.0 --repo HaiweiZuo/time-coordinate-forecasting --pattern '*-checkpoints.zip' --dir downloads
python verify_artifacts.py --archives downloads
```

After verification, extract both archives into the repository root. Each contains 24 files beneath `checkpoints/`. The [checkpoint registry](checkpoint_index.json) records their identities, hashes, selected epochs, and early-stop losses.

Public filenames use LL/LS/SL/SS. Original checkpoint bytes remain unchanged, including stored `legacy` and `geometry` identifiers for LL and SS. The loader validates dataset, model family, coordinate arm, and seed before loading the state dictionary.

## Training and scoring

Train a single crossed cell with explicitly prepared inputs.

```sh
python run_experiment.py --mode train --dataset weather_jena2020 --family conditional_diffusion --arm SL --seed 2024 --train-payload data/prepared/weather/train.pt --val-payload data/prepared/weather/val.pt --output runs/weather_SL_diffusion_2024 --device cuda:0 --verify-study-roster
```

Score the matching retained checkpoint in a separate invocation.

```sh
python run_experiment.py --mode score --dataset weather_jena2020 --family conditional_diffusion --arm SL --seed 2024 --train-payload data/prepared/weather/train.pt --val-payload data/prepared/weather/val.pt --checkpoint checkpoints/weather_jena2020__SL__conditional_diffusion__s2024.pt --output runs/weather_SL_diffusion_2024_score --device cuda:0 --verify-study-roster
```

Supported dataset names are `weather_jena2020` and `ushcn`. Model families are `conditional_diffusion` and `direct_gaussian`. Use a new output directory for each invocation. Training does not automatically score the development partition, resume an existing run, or launch other cells.

The roster option requires the original partition counts and digests. A matching roster does not establish byte-identical input values or an identical numerical run.

## Verification

The archived-result checks operate on published summaries and histories, without loading observations or running models.

```sh
python results/verify_numerical_release_v2.py --package results
python -B -m unittest discover -s tests -p test_preparation.py -v
python -B tests/smoke.py
```

The preparation tests use synthetic fixtures. Model smoke tests cover all 16 dataset/family/arm combinations and matched-coordinate equivalence. Four retained checkpoints were additionally checked using strict state loading and synthetic forward passes. Local compatibility checks used Python 3.11.9, PyTorch 2.8.0 CPU, and NumPy 2.2.6; they do not reproduce the recorded GPU experiments.

## Scope and interpretation

This release preserves the model, partitioning, and metric implementations while adapting the execution entry points for local use. It is not the historical server launcher. Preprocessing preserves the inspected source transformations, but regenerated serialization can change file hashes. End-to-end regeneration of the original numerical results has not been verified with this portable package.

Prediction uses retrospective query times and realized target masks. It does not establish performance when future observation availability is unknown. Weather temporal results concern a later period at the same station. The specialized Weather 2022 inference-roster workflow is not included in the portable training interface; its retained summaries and verification scripts are included under `results/`.

No general reuse license is assigned in this release. Third-party data and software remain subject to their respective terms. See [NOTICE.md](NOTICE.md).

