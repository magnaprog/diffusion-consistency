# Reproduce the paper’s results

You can reproduce Table 1, redraw all nine figures, and repeat checkpoint selection on a CPU using the saved forecasts and scores. For the hidden Markov model (HMM), you can also calculate the conditional probabilities of the paths and compare them with the frequencies in Figure 4.

## Install and check the code

Use Python 3.10 or later. Run the commands in this guide from the repository root.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
```

The tests check the magnetic constraints, scoring rules, denoiser, and HMM on small examples. On Linux, run `sha256sum --check SHA256SUMS` to verify the files against their published digests.

## Table 1

```bash
python scripts/reproduce_table1.py
```

The script prints scores, interval coverage, and constraint errors for all eight methods. The energy score compares each ensemble with the withheld measurements across 60 standardized coordinates (five quantities in twelve missing bins) and accounts for the spread among its members. For each fit, the script averages the window scores within each test month, then gives all months equal weight. Each diffusion row averages three training seeds. Each fit uses the checkpoint with the lowest raw validation energy score for the raw columns and the lowest clipped validation energy score for the clipped columns. [Scores and diagnostics](../evidence/real/)

Coverage is the percentage of withheld values inside their marginal 90% prediction intervals, including the endpoints (the second and 38th of 39 ordered draws). The last two columns describe raw samples. One reports the mean absolute equality residual in nT². The other gives the percentage of generated bins where the mean or variance of field magnitude violates a bound by more than the rounding tolerance. For Gaussian regression, the script reads scores and coverage by window and calculates the raw residual from the supplied ensemble. [Gaussian regression records](../evidence/deadline/gaussian/)

## Draw the figures

```bash
python scripts/reproduce_figures.py --output runs/figures
```

Use a new output directory. Each numbered folder, from `figure_01` through `figure_09`, contains a PDF, a PNG, and a JSON manifest identifying the inputs. The renderers check the input hashes. Figure 3 uses the paper’s Nimbus Roman font for its table and needs TeX Live with `pdftex`, `kpsewhich`, and the Times fonts. To check for the font, run `kpsewhich utmr8a.pfb`. The plots use Matplotlib’s DejaVu Serif. [Figure inputs](../figures/inputs.json)

To draw only Figure 4, run `python scripts/reproduce_figures.py --figure 4 --output runs/figure4`. Open `runs/figure4/figure_04/fig_hmm.pdf`. To check the calculation behind the plot, see [Recalculate the HMM diagnostic](#recalculate-the-hmm-diagnostic).

- **Training seeds.** In Figures 1–3, 5, and 6, circles, squares, and upward-pointing triangles identify seeds 7, 13, and 42. Gold bars mark means across seeds in Figures 1, 2, 5, and 6. Figure 2(b) gives the residual for unpenalized diffusion (D). Figure 2(a) uses D as the reference for the penalized fits, so its D row is empty. The experiment manifest records the original filename `fixed-final.json` for the [Figure 2 data](../evidence/deadline/penalty-sensitivity.json).
- **Checkpoint choices.** Figure 3(a) shows raw energy scores for D and penalized diffusion (O-S) at five training steps. The validation columns use 25 and 50 denoiser evaluations per sample. The test column uses 50 on a different set of windows. The orange curves in (b) give O-S minus D at the same step. The test difference in (c) uses each fit at the step with its lowest validation score.
- **Clipping.** In Figure 6(c), the left-pointing triangle shows how much clipping reduces the mean distance to the withheld observation. The right-pointing triangle shows the reduction in half the mean distance between distinct ensemble members.
- **Example forecasts.** In Figures 7–9, D is blue, O-S pink, G5 orange, and Gaussian regression green. Solid colored lines show predictive medians and the colored bands show their 90% prediction intervals. Gray shading marks the missing hour. In Figures 7 and 8, solid black steps in each forecast panel show the five-minute input bins immediately before and after the gap. Figure 9 shows ten minutes of input on each side. Black dashes show the withheld measurements. In Figures 7 and 8, thin dash-dot lines trace three individual samples. [Forecasts for the two illustrated windows](../evidence/imputation/)

## Recalculate the HMM diagnostic

```bash
mkdir -p runs
OPENBLAS_NUM_THREADS=1 python figures/hmm_display_evidence.py \
  --evidence-dir evidence/toy --output runs/path-diagnostic.json
```

The script checks the same 36 cases at four noise levels. Gold diamonds in Figure 4(a) and (b) mark means over those cases. At the highest noise level, each estimate is decoded to the nearer HMM state in each bin. Counting switches along a path includes the transitions at both ends of the gap, for 13 transitions in all.

Figure 4(c) compares the frequencies of switch counts in the decoded paths with their probabilities under the model given the end states. Figure 4(d) asks where the switch occurs among paths with exactly one switch. For Table 2, the script enumerates all 4,096 hidden paths to calculate the reference energy score on the 288 realized HMM targets (4.2207). The direct sampling row uses 39 draws per target from the same conditional law (4.2287). Use a new output filename and compare the resulting JSON with the [saved diagnostic](../evidence/toy/path-diagnostic.json).

## Repeat checkpoint selection

```bash
mkdir -p runs
python scripts/reselect_checkpoints.py --output runs/reselection.json
```

The script samples years with replacement and uses each year the same number of times in validation and testing. In each resample it chooses the step with the lowest validation score for each fit, then calculates the mean test score difference (O-S minus D) across the three seeds. The JSON records the selection counts, score differences, and input path relative to the output directory. Choose a new filename for each run. [Checkpoint scores and choices](../evidence/followup/checkpoints/) · [Original reselection result](../evidence/followup/checkpoint_reselection.json)

## Solar wind data and new experiments

| Available data | Where to find them |
| --- | --- |
| Measurements for the 480 test windows and normalization statistics from training | [Test observations](../evidence/observations/) |
| Interval coverage and fractions below and above the interval for each window | [Calibration records](../evidence/followup/calibration/) |
| Gaussian regression fits, validation choices, and the complete test ensemble at the chosen parameters | [Gaussian regression records](../evidence/deadline/gaussian/) |
| G5 forecasts from the separate 2025 experiment on variance supervision | [Replay instructions](g5_auxiliary/README.md) and [results](g5_auxiliary/RESULTS.md) |

For the paper’s diffusion models, the repository includes scores for all 480 test windows and sample forecasts for the two illustrated windows. Generating the original neural forecasts again would require the original checkpoints. The release omits those weights, the full neural and Gaussian interpolation ensembles, the annual OMNI files, and the original training and validation arrays.

The earlier subgroup analysis ranks windows by the mean change in the field vector between consecutive hidden bins (Euclidean distance), then separately by the mean absolute residual of the hidden measurements. It reports mean scores for each third of either ranking. It also reports the fraction of windows where the energy score averaged across three training seeds beats linear interpolation, overall and for each third of the field-change ranking. This script requires its original training arrays, scores, and ensembles. The public release provides the paper’s scores by window and measurements of the withheld fields. [Earlier subgroup analysis](../src/prigm/run_strata.py) · [Scores by window](../evidence/real/scores.json)

**Recalculate interval coverage from the original ensembles.** The tail calibration script defaults to the study’s original directory layout. If you have that study directory, including the full neural and Gaussian interpolation test ensembles, set `study` to its path and run the following command with a new output directory. The public calibration records above contain the resulting summaries and fractions for each window. [Tail calibration script](../src/analysis/prigm_tail_calibration.py)

```bash
mkdir -p runs
python -m analysis.prigm_tail_calibration --study-root "$study" --evidence-root evidence --output runs/tails
```

**Rebuild the arrays.** Download the annual files `omni_min2005.asc` through `omni_min2024.asc` from NASA’s modified one-minute OMNI solar wind data (Level 3). NASA may revise the files. To identify the versions used in the paper, compare each file’s SHA-256 digest with its entry under `hashes` in the inventory. [NASA data directory](https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/modified/) · [Column definitions](https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/modified/hro_modified_format.txt) · [Study inventory](../evidence/data/inventory.json)

The array builder takes the directory containing the annual files (`--omni-root`) and a new directory for the arrays and normalization statistics (`--output`). The solar wind experiment driver passes a residual penalty coefficient of 0.0005 for I-S and O-S. If you invoke `prigm.run_fit` directly for either method, pass `--c-r 0.0005`. Its default is zero. [Array builder](../src/prigm/prepare_arrays.py) · [Solar wind experiment](../src/experiments/run_prigm_repair.py) · [HMM experiment](../src/experiments/run_prigm_toy_repair.py)

```bash
python -m prigm.prepare_arrays --help
python -m experiments.run_prigm_repair --help
python -m experiments.run_prigm_toy_repair --help
```
