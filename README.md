# Target-period source assignment

Reference-calibrated source-period inference from mixed biological recordings.

The circadian model combines source-role period evidence with reference-specific cohort calibration and bounded transport of period distributions. Harmonic information computed from observed timestamps determines the calibration weights. The fetal ECG model combines synchronous abdominal sensors through a shared-landmark renewal likelihood.

## Inputs and outputs

Observations use the columns `trace_id`, `run`, `condition`, `x`, and `signal_y`. Times are expressed in the model's physical unit. Keep the complete peer cohort or synchronous sensor group together. Missing observations are represented by their observed coordinates.

Predictions include a normalized period distribution, its expected period, a cohort mean, and individual deviations. Circadian and ECG models use their own source references, period supports, and sensor grouping.

## License

Original software is released under the MIT License. Data and third-party software retain their source licenses and attribution.
