# What changes when G5 learns from the measured vector variance?

Both G5 models predict five magnetic quantities and derive vector variance from them. Only G5aux uses the error in that variance during training. The registered comparison uses each fit at the epoch with its lowest validation energy score. On the 2025 windows, adding variance supervision raised the mean energy score for the five directly predicted quantities. For vector variance itself, supervision lowered the continuous ranked probability score at every seed and, after averaging over the three fits, in every month.

| Forecast measure | G5 | G5 with variance supervision | Change |
|---|---:|---:|---:|
| Energy score for the five primary quantities | 2.797 | 2.839 | +0.042 |
| Continuous ranked probability score for vector variance (nT²) | 3.427 | 1.872 | −1.555 |
| Coverage of the 90% interval for vector variance | 75.41% | 74.63% | −0.77 percentage points |
| Mean width of that interval (nT²) | 16.454 | 7.901 | −8.553 |

Lower values indicate better forecasts for both scores. The energy score uses the five quantities standardized on the training data. Each row averages windows within month and then gives equal weight to the twelve months and three fits. Changes use the unrounded values.

With variance supervision, the mean absolute distance between a forecast member and the measured variance fell by 2.90 nT². The forecasts also became more concentrated. Half the mean absolute distance between distinct members fell by 1.35 nT². The variance score subtracts the second distance from the first, so it fell by 1.55 nT². The 90% intervals from both models covered about three quarters of the observations.

Beucler et al. made a similar distinction for atmospheric quantities. They compared calculating some outputs after training with including those calculations in the network, where errors in the derived outputs could affect the fit. [Supplementary Figure 6 and Table VIII](https://arxiv.org/pdf/1909.00912v5)

## Comparison at the same epoch

The lowest validation score occurred at epoch 31 for all three fits with variance supervision. For G5 it occurred earlier, at epochs 24, 24, and 10 for seeds 71, 73, and 79, respectively. After examining the registered results, the study compared the models at epoch 31 to give them equal amounts of training. Only G5 required new forecasts, using the same windows and sampling noise.

Each entry gives the score with variance supervision minus the G5 score for the same seed, or the mean of those differences. Negative values favor supervision. The joint energy score is dimensionless. The variance score is in nT².

| Training seed | Joint, validation choice | Joint, epoch 31 | Variance, validation choice | Variance, epoch 31 |
| --- | ---: | ---: | ---: | ---: |
| 71 | −0.066 | −0.100 | −0.735 | −2.241 |
| 73 | +0.150 | +0.100 | −1.613 | −1.917 |
| 79 | +0.042 | −0.158 | −2.317 | −0.991 |
| Mean | +0.042 | −0.053 | −1.555 | −1.716 |

Variance supervision lowered the variance score at every seed under both choices of epoch. The result for the five primary quantities depended on the comparison. At epoch 31 the mean energy score fell, with improvements at seeds 71 and 79. At the epochs with the best validation scores, only seed 71 improved and the mean score rose.

At epoch 31, the narrower 90% intervals from the model with variance supervision covered 74.6% of the measured variances, compared with 59.0% for G5. The frequency of physically impossible moments also depended on the choice of epoch. With supervision, 49.4% of bins across the forecast members violated at least one moment bound after allowing for rounding. For G5, the fraction was 52.4% at the epochs with the best validation scores and 23.5% at epoch 31.

The [protocol](../../src/experiments/g5_auxiliary/protocol.json) was committed before fitting the models or examining the 2025 outcomes. Each fit trained on the original 122,309 windows for 31 epochs. Scores on the 319 validation windows determined the epoch for each of the six fits before evaluation on the 144 windows from 2025. The comparison at epoch 31 was added after the registered results had been examined. The [numerical results](../../evidence/g5_auxiliary/results.json) give the validation scores, chosen epochs, and differences by seed and month. The [reproduction instructions](README.md) explain how to calculate the scores from the forecasts.
