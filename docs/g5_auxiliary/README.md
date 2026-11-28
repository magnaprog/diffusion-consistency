# Supervision through the derived vector variance

The workshop paper compares a diffusion model that predicts six magnetic quantities with G5, which predicts five and calculates vector variance from them. That comparison changes both the network outputs and the training loss. This follow-up keeps the five outputs and asks whether training against the measured vector variance improves forecasts of a missing hour.

G5 and G5aux use the same denoising loss on the five outputs. G5aux adds a term calculated from the network’s first estimate of the clean moments. For each of the twelve missing bins, it derives vector variance from those estimates and compares it with the measured variance. The term averages the squared errors over the bins, divides by the square of the training standard deviation for vector variance, and applies one fifth of the diffusion time weight. Both models receive all six measured channels on either side of the gap.

The measured moments satisfy the completion rule up to rounding. Why can a loss on the derived variance change the fit? Under squared error, the best estimate of each clean quantity at a given noisy input is its conditional mean. The completion rule squares some of its inputs. Applying it to the five conditional means can give a different answer from averaging the possible values of vector variance at that same input. Equation 4 in the paper gives the difference. The extra loss favors estimates whose derived variance is close to that conditional mean of vector variance.

For each of three new seeds, the two models start from the same weights and use the same training windows, diffusion noise, and optimizer configuration for 31 epochs. Of five candidate epochs, the one with the lowest validation energy score supplies the forecast for each fit. This score uses only the five quantities that both networks predict directly. Each validation ensemble contains 39 reconstructions, with 25 denoiser evaluations per reconstruction.

Training and validation use data from earlier years. For the 2025 evaluation, the workshop study’s quality and sampling rules yield twelve four-hour windows per month from the modified OMNI minute records. Both models forecast the same windows from the same initial noise, with 50 denoiser evaluations per reconstruction. The [2025 comparison](RESULTS.md) gives the energy score across all five primary quantities and twelve hidden bins. It also reports the continuous ranked probability score for vector variance, along with the width and coverage of its 90% prediction intervals.

## Recompute the results

From the repository root, install the project and score the [archived forecasts](../../evidence/g5_auxiliary/forecasts_2025.npz) on a CPU.

```bash
python -m pip install -e .
python -m experiments.g5_auxiliary.replay
```

With NumPy already installed, the source checkout also supports `PYTHONPATH=src python -m experiments.g5_auxiliary.replay` from the repository root. The [replay code](../../src/experiments/g5_auxiliary/replay.py) reads the forecasts and the [registered scores](../../evidence/g5_auxiliary/results.json) from `evidence/g5_auxiliary/`.

For each seed, the command compares G5 with and without variance supervision, first at the epochs selected by validation score and then at epoch 31. Add `--json` for the scores, interval coverage and width, and both distance terms in the variance score for each fit. If the records are stored elsewhere, pass `--evidence-dir` followed by the directory containing `results.json` and `forecasts_2025.npz`.

Each forecast array has shape `(144, 39, 5, 12)`. Its axes index windows, ensemble members, standardized quantities, and hidden bins, in that order. The archive also contains the six measured quantities in physical units, the training normalization, and the window identifiers. A suffix of `_final` marks the G5 forecasts at epoch 31. For all three fits with variance supervision, epoch 31 also had the lowest validation score among the candidates. The same array serves both comparisons.

## Repeat training and forecasting

Training and forecasting require PyTorch with CUDA and the original training and validation arrays. [The protocol](../../src/experiments/g5_auxiliary/protocol.json) gives the SHA-256 digests of those arrays. Set `inventory` to the directory containing `train.npz` and `val.npz`, `omni` to the directory containing the [2025 OMNI minute records](https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/modified/omni_min2025.asc), and `study` to the output directory. Run the commands from the repository root. They write the weights, the six checkpoint choices, and the forecasts for each window under `study`.

```bash
mkdir -p "$study"
python -m experiments.g5_auxiliary.holdout "$omni/omni_min2025.asc" "$study/holdout.npz"
for seed in 71 73 79; do
  for arm in G5 G5aux; do
    python -m experiments.g5_auxiliary.train "$arm" "$seed" "$inventory/train.npz" "$study/runs/${arm}_${seed}"
    for epoch in 3 10 17 24 31; do
      python -m experiments.g5_auxiliary.evaluate validate "$study" --val "$inventory/val.npz" --arm "$arm" --seed "$seed" --epoch "$epoch"
    done
  done
done
python -m experiments.g5_auxiliary.evaluate freeze "$study"
for seed in 71 73 79; do
  for arm in G5 G5aux; do
    python -m experiments.g5_auxiliary.evaluate holdout "$study" --val "$inventory/val.npz" --holdout "$study/holdout.npz" --arm "$arm" --seed "$seed"
  done
done
python -m experiments.g5_auxiliary.summarize "$study" "$inventory/val.npz" "$study/results.json"
for seed in 71 73 79; do
  python -m experiments.g5_auxiliary.common_epoch "$study" G5 "$seed" "$inventory/val.npz" "$study/holdout.npz" "$study/final_forecasts"
done
```

Training resumes at the most recently completed epoch. The summary command recomputes the registered scores from the individual reconstructions. The last three inference runs reuse the original 2025 noise sequence and write the G5 forecasts at epoch 31 to `final_forecasts/`.
