# Source and third-party notices

The model, partitioning, and scoring modules are retained from the study implementation. The public training entry point and preparation interface are portable adaptations. Original checkpoint bytes are preserved; no new training results are represented as archived study outputs.

This repository does not vendor PyTorch, NumPy, their distributions, or third-party observation files. Those projects and data providers retain their respective rights and license terms.

Weather observations are supplied by the Max Planck Institute for Biogeochemistry, Jena, through its [weather archive](https://www.bgc-jena.mpg.de/wetter/weather_data.html) under CC BY 4.0. USHCN benchmark files are referenced through the [t-PatchGNN repository](https://github.com/usail-hkust/t-PatchGNN/tree/00c94e7bbaf21c71b03ed84ff690ae59e37129e5). Neither observation source is redistributed here.

The project code, published checkpoint weights, and numerical materials are licensed under the MIT License in LICENSE. This grant includes the unchanged checkpoint files distributed with release v1.0.0 and identified in checkpoint_index.json. Third-party observations, dependencies, and any separately attributed third-party material retain their own terms and are not relicensed by this notice.
