# ReSPI 0.3.1

**ReSPI** stands for **Reference-calibrated Source-specific Period Inference**. It estimates source periods from measured time series and reference-trained source models. Prediction uses observed timestamps, masks, period bounds and measurements. Native mixed reporters use an information-conditioned mixture of reference calibrations; simultaneous abdominal ECG sensors use a shared-landmark renewal model. Both return one normalized predictive period distribution, its expectation, the average source expectation within each run, and the corresponding individual deviation.

## Predict from a fitted model

The installable distribution is `target-period-source-assignment`, and its prediction entry point is `predict_source_assignment`. Install the supplied wheel in an environment with its declared dependencies. From the extracted project root:

    python -m pip install release/target_period_source_assignment-0.3.1-py3-none-any.whl
    python -m predict_source_assignment --input data/sota_dev/recomputed_clean/lemna_task_period_acf_input.parquet --model results/0.3.1/unified_model.npz --output native.csv

Input Parquet columns are trace_id, run, condition, x and signal_y. x contains actual observation times. Retain the observed time coordinates when samples are missing. Keep the complete peer cohort of each run together. condition supplies labels for reported treatment contrasts.

Native output contains estimate_period, experiment_mean, individual_deviation and period_unit, with companion files for experiment means, contrasts and predictive probabilities. The shared-source ECG output uses estimate, group_mean, deviation and time_unit, with a companion source-distribution NPZ. Every reported period and deviation is derived from the final predictive distribution. Hour-specific aliases are supplied for hour-valued models. Biological effect comparisons average experiments within their declared control or pairing context.

For the supplied ECG model:

    python -m predict_source_assignment --input data/external/adfecgdb/inputs.parquet --model results/0.3.1/adfecg_model.npz --output adfecg.csv

Keep the detector .joblib beside its model as distributed. Simultaneous sensors in a run measure the same fetal source and receive the same source-period distribution. Direct fetal waveforms and expert landmarks enter reference training and evaluation; query prediction takes abdominal measurements alone. The supplied native and ECG models use hours and seconds, respectively. To change units, scale timestamps and both period bounds together and declare --time-unit.

## Native source model

The source operator evaluates harmonics and residual covariance on actual timestamps and observation masks. Each fitted reference calibration applies its recorded source operator and measurement coefficients, its training-only mean calibration, and a bounded Wasserstein affine transport of the raw-measurement individual density. This produces a complete period distribution for each reference calibration. Nonnegative interpolation weights depend continuously on the observed harmonic information and sum to one. Observation count and dataset name do not select a prediction head.

The final mixture preserves its support and total probability. Its mean is the official period estimate; the finite-cohort mean averages these estimates. The distributions are calibrated predictive mass. Predictive-interval coverage requires separate calibration.

The supplied model contains 30 reference-exclusion cases: deployment, experiment exclusions, and treatment/control exclusions. Each calibration records its own reference identities, independent units and exclusions. The archive native_reference_calibration_model.npz contains the fitted calibration parameters, source identities and training exclusions used for the benchmark model.

The native benchmark reference bank was trained with foreground periods, optical background-peak periods, and matched views made from the measured optical foreground/background components. The base background head used 506 qualified recordings from 17 runs. The refined background head used 635 distinct recordings from 13 runs across five weighted views; its foreground-period fit used 387 qualified recordings. These view rows are repeated measurements of the same sources. Foreground-label-matched scalar controls use the foreground periods and their group means. Query prediction uses No and peers alone. New cold fits without supplied background references use a No-peer proxy, which is a different supervision recipe from the supplied benchmark bank.

## Shared-source ECG model

Training uses expert fetal QRS landmarks and direct-source reference waveforms from training subjects. A histogram gradient-boosting detector learns from every timestamp using independently standardized sensor amplitudes, actual-time temporal neighborhoods, and derivative/curvature features. A source-level event stream enters a renewal likelihood with training-estimated event timing, clutter and missed-event terms. The learned foreground interval prior enters once. The output is a predictive distribution for the source's mean interbeat interval. Subject exclusion keeps held-out expert annotations out of fitted parameters.

## Fit and reproduce

Replay the supplied models and complete comparisons:

    python experiments/evaluate_posterior_clock_ablations.py --family unified --stage all-evaluate --model results/0.3.1/unified_model.npz --device cpu

This evaluates native exclusions, controlled mixtures, shorter and missing acquisitions, the two 2021 reporter datasets, six SCN strata, coupled clocks and fetal ECG, then runs numerical validation. Results are written to results/0.3.1. See REPRODUCING.md for reference reuse and new-source fitting.

Native comparisons include reference/calibration removals, frozen-parameter structural ablations with training-only mean recalibration, and a matched-budget Density31 fit. ECG comparisons change one component at a time and retain same-detector event-mean, event-median and ridge controls. Tables identify fit policy, reference budget, coverage, convergence and applicability. Historical 0.2.4 predictions keep their original protocol labels.

## Dependencies and licenses

Python 3.11 or newer is required. The wheel declares NumPy, pandas, SciPy, PyTorch, PyArrow, scikit-learn 1.8.0 and joblib 1.5.2; the last two also load the fitted ECG detector. The numerical evaluation scripts use the same declared scientific Python dependencies.

Original project code is licensed under MIT. Third-party software and source data retain their original terms. PhysioNet Abdominal and Direct Fetal ECG Database 1.0.0 uses ODC Attribution 1.0; its source, attribution and checksum records accompany the data. 