# Data sources and preparation

## Weather

The observations come from the rooftop station operated by the Max Planck Institute for Biogeochemistry in Jena. The [official archive](https://www.bgc-jena.mpg.de/wetter/weather_data.html) distributes the data under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit the institute and retain the source and license links when using those observations.

The training study uses [2020a](https://www.bgc-jena.mpg.de/wetter/mpi_roof_2020a.zip) and [2020b](https://www.bgc-jena.mpg.de/wetter/mpi_roof_2020b.zip). The fixed-checkpoint temporal analysis uses [2022a](https://www.bgc-jena.mpg.de/wetter/mpi_roof_2022a.zip) and [2022b](https://www.bgc-jena.mpg.de/wetter/mpi_roof_2022b.zip). These archives are linked rather than redistributed.

The public preparation command covers the 2020 inputs. It retains the original column order, averages valid duplicate timestamps, handles the missing sentinel, and constructs a ten-minute index lattice. Training uses the first 70 percent of rows; validation uses the following 10 percent. The remainder is not exported. Scaling uses finite observed training values only.

## USHCN

The study uses the five-channel processed benchmark distributed by [t-PatchGNN](https://github.com/usail-hkust/t-PatchGNN/tree/00c94e7bbaf21c71b03ed84ff690ae59e37129e5/data/ushcn), pinned to commit `00c94e7bbaf21c71b03ed84ff690ae59e37129e5`. Obtain [processed/ushcn.pt](https://github.com/usail-hkust/t-PatchGNN/blob/00c94e7bbaf21c71b03ed84ff690ae59e37129e5/data/ushcn/processed/ushcn.pt) from that distribution. The upstream [loader](https://github.com/usail-hkust/t-PatchGNN/blob/00c94e7bbaf21c71b03ed84ff690ae59e37129e5/lib/ushcn.py) documents its record representation.

This file is an already-processed derivative, not untouched NOAA observations. The public preparation utility does not reconstruct it from the benchmark CSV. The complete physical station/calendar mapping and upstream transformation history have not been established for this release.

Permission to redistribute this derivative has not been verified. Readers are therefore directed to the upstream distribution and its applicable terms; the data are not bundled here. This is a licensing-verification limitation, not a claim of privacy restrictions or a prohibition on access.

The source contains a legacy pickle representation. The utility verifies its complete pinned SHA-256 before deserializing the same open stream. A matching hash establishes file identity, not a general guarantee that pickle is safe. Use only the trusted upstream file; no unchecked-pickle option is provided.

Preparation string-sorts source identifiers and applies the original NumPy permutation with seed 20260801. The first 60 percent of records form training; the following 20 percent form validation. The shuffled order is retained. Only observed training values determine scaling.

## Input identities

| Source file | SHA-256 |
| --- | --- |
| mpi_roof_2020a.zip | `6a0be0552a2a26a04c3b41a5281d4a55ae251d4c65ce630971a8662f43b31f9f` |
| mpi_roof_2020b.zip | `c37eca910ee2803b17b8c587f3194bc7625abf994576c50f32538dd7cb65ac77` |
| processed/ushcn.pt | `06049d363c8bf7eabfe207ef76d73417a0e9b97467e1f060517f94f7eedf4d9a` |

The machine-readable [source specification](preprocessing/source_spec.json) also records preparation rules and source-code identities. Unknown or changed inputs are rejected before preparation.

## Reproduction limits

Prepared tensors are cloned before saving so unselected backing storage is not retained. Record identifiers are represented by standard Python integers or strings. Container representation and library versions can change serialized bytes even when numerical contents agree.

Synthetic tests verified the extracted transformations against the inspected preparation code. The release was not validated by regenerating real-data payloads or retraining models. Users should check source hashes, preparation receipts, original roster digests, and their numerical outputs separately. The archived checkpoints and aggregate results provide fixed comparison artifacts.

