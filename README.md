# Diffusion consistency

Code and public evidence for **Residual Penalties Versus Algebraic Completion In Conditional Diffusion**, accepted at the 2nd Workshop on Principles of Generative Modeling, NeurIPS 2026.

The study compares ways to impose algebraic constraints on conditional diffusion predictions of magnetic statistics during an hour of withheld measurements. It distinguishes mathematical consistency from predictive accuracy. The [paper](paper/workshops/prigm/main.pdf) describes the methods, results and limitations; its [TeX source](paper/workshops/prigm/main.tex) is included.

The [release guide](docs/prigm_release.md) covers installation, CPU tests, evidence verification, data reconstruction and paper builds. The public evidence includes TEST observations, numerical results, toy arrays, selected Gaussian predictions and inputs for the figures. Neural checkpoints, complete neural prediction sets, TRAIN/VAL arrays and raw OMNI files are not included, so the available checks reproduce only part of the experiments. The original three training seeds and earlier exposure to TEST limit the conclusions as stated in the paper.

For raw-data reconstruction, use the [modified Level-3 OMNI one-minute ASCII product](https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/modified/hro_modified_format.txt) from the [NASA data directory](https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/modified/). Compare downloaded yearly files with the checksums of the original inputs before reconstruction; upstream revisions can change the resulting arrays.

`RELEASE_MANIFEST.json` records the file hashes and sizes in this export. It establishes copy identity; test results and paper compilation require separate checks. The [accepted-source record](reproducibility/accepted_source/README.md) and [documentation-source record](reproducibility/documentation_source/README.md) explain the historical source copies needed to verify the evidence.

Repository code uses the [MIT license](LICENSE). The NeurIPS style keeps its original attribution. NASA OMNI provenance and acknowledgment instructions remain in the linked source documentation.
