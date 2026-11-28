# Diffusion constraints for solar wind gap filling

Code and results for *Residual Penalties Versus Algebraic Completion In Conditional Diffusion*, by Kevin Lee, NeurIPS 2026 Workshop on Principles of Generative Modeling (PriGM).

Can a diffusion model reconstruct a missing hour of solar wind measurements from the 90 minutes on either side? For each five-minute bin, the models reconstruct three mean field components, the mean and variance of field magnitude, and the vector variance. The paper compares models that predict all six with G5, which predicts the first five and calculates vector variance from them. Use the saved observations and scores for the 480 test windows to reproduce Table 1. You can redraw all nine figures on a CPU.

## Start with Table 1

Use Python 3.10 or later. From a terminal on Linux or macOS, clone the repository and install the packages in a virtual environment.

```bash
git clone https://github.com/magnaprog/diffusion-constraints.git
cd diffusion-constraints
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python scripts/reproduce_table1.py
```

Table 1 compares eight procedures. For G5, the energy score on the five directly predicted quantities is 1.785 for raw forecasts and 1.784 for clipped forecasts. Table 1 averages window scores within each test month and gives the months equal weight. Lower scores are better. Clipping restricts each five-minute prediction to moments that five field vectors could produce. Each diffusion fit uses the training step with the lowest raw validation score for the raw column and the step with the lowest clipped validation score for the clipped column.

## Reproduce a result

Run the commands from the repository root. Create `runs/` with `mkdir -p runs`. Each figure command needs a fresh output directory and writes a PDF, a PNG, and a JSON record of the inputs for each figure.

| Result | Command |
| --- | --- |
| Figure 4, the two-state hidden Markov experiment | `python scripts/reproduce_figures.py --figure 4 --output runs/figure4` |
| All nine figures | `python scripts/reproduce_figures.py --output runs/figures` |
| Repeat checkpoint selection under year resampling | `python scripts/reselect_checkpoints.py --output runs/reselection.json` |
| G5 follow-up on the 144 windows from 2025 | `python -m experiments.g5_auxiliary.replay` |

For Figure 3, including a run of all nine figures, install TeX Live with `pdftex` and the Times fonts. The command `kpsewhich utmr8a.pfb` should print the path to the table font. The reproduction guide also gives the command to recalculate the hidden Markov path diagnostic and explains the inputs for each analysis. [Reproduction guide](docs/reproduction.md)

The separate 2025 experiment compares two models that both predict five quantities. One also uses the error in the derived vector variance during training. The replay scores the saved forecasts for each fit at the epoch with its lowest validation score and at epoch 31. [Study design and replay instructions](docs/g5_auxiliary/README.md) · [2025 results](docs/g5_auxiliary/RESULTS.md)

## Repository layout

| Path | What to find there |
| --- | --- |
| [scripts/](scripts/) | Commands for Table 1, the figures, and checkpoint selection |
| [src/prigm/](src/prigm/) | Data preparation, magnetic constraints, diffusion model, samplers, and scoring rules |
| [src/experiments/](src/experiments/) | Training and evaluation for the solar wind and simulated experiments |
| [src/analysis/](src/analysis/) | Score aggregation, Gaussian regression, interval calibration, and clipping analyses |
| [src/scripts/](src/scripts/) | Original checkpoint calculation used by the reselection command |
| [figures/](figures/) | Plotting code and the input records for each figure |
| [evidence/](evidence/) | Study inputs and saved results, including test observations, scores, case forecasts, and the 2025 forecast archive |
| [docs/](docs/) | Reproduction instructions, data availability, and the 2025 study design and results |
| [tests/](tests/) | Tests for the constraints, denoiser, scoring rules, and hidden Markov calculations |

## Read the saved scores

In `evidence/real/scores.json`, `data["D_raw"]` contains the energy scores for unpenalized diffusion (D). Its three rows correspond to training seeds 7, 13, and 42. Its 480 columns correspond to the test windows in `ids`, and `months` gives the month for each window. The other diffusion methods use the same layout. I-S penalizes the instantaneous residual, O-S penalizes the residual of the averaging identity, and G5 calculates vector variance from that identity. [Window scores](evidence/real/scores.json)

For a diffusion fit, `_corrected` means that the forecasts were clipped and the training step was chosen by the clipped validation score. To measure the effect of clipping at a fixed step, compare `D_raw` with `D_corrected_at_raw_checkpoint`. Use `row_ids` to identify the fit for each row. [Score aggregation code](scripts/reproduce_table1.py)

## Data and checks

The paper uses modified one-minute OMNI solar wind data from 2005 through 2024, provided by J. H. King, N. E. Papitashvili, and NASA’s Space Physics Data Facility. The checkout includes test observations and training normalization. Retraining requires reconstructing the original training and validation arrays. The data inventory identifies the yearly input files by checksum. [OMNI dataset](https://doi.org/10.48322/mj0k-fq60) · [Input inventory](evidence/data/inventory.json) · [Data availability](docs/reproduction.md)

To run the method tests, install the test dependencies in the same environment. On Linux, the last command checks the distributed files against their SHA-256 digests.

```bash
python -m pip install -e '.[test]'
python -m pytest -q tests
sha256sum --check --quiet SHA256SUMS
```

## Citation

```bibtex
@inproceedings{lee2026diffusion,
  author    = {Lee, Kevin},
  title     = {Residual Penalties Versus Algebraic Completion In Conditional Diffusion},
  booktitle = {40th Conference on Neural Information Processing Systems ({NeurIPS}),
               2nd Workshop on Principles of Generative Modeling ({PriGM})},
  year      = {2026}
}
```

## License

The code is distributed under the [MIT license](LICENSE).
