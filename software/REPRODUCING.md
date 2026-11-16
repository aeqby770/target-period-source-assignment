# Numerical reproduction 0.3.1

Run commands from the extracted project root. Install the supplied wheel and use its declared dependencies. Set KIMI_DEVICE=cpu for CPU execution or cuda for an available PyTorch CUDA device. Native and ECG models are separate reference-trained models selected by their saved measurement definitions.

## Complete numerical evaluation

Install the supplied wheel in the selected Python environment. Extract the versioned model and data assets into the project root, preserving their data/ and results/ paths. Run:

    python experiments/evaluate_posterior_clock_ablations.py --family unified --stage all-evaluate --model results/0.3.1/unified_model.npz --device cpu

The evaluation uses the distributed reference parameters and all fixed observations. Numerical validation and the experiment tables are written to results/0.3.1.

The fitted deployment case supplies transfer predictions. Native accuracy uses the saved held-out queries. Model comparison_model_paths declare the accompanying native ablation models; the evaluation runner forwards them to the same standard evaluators. Every comparison preserves the original input population and its reference-eligibility denominator.

To reconstruct the ten native ablation models from the distributed formal model and raw No measurements, then evaluate the complete native/external matrix:

    python experiments/evaluate_posterior_clock_ablations.py --family unified --stage models --model results/0.3.1/unified_model.npz --device cuda
    python experiments/evaluate_posterior_clock_ablations.py --family unified --stage all-evaluate --model results/0.3.1/unified_model.npz --device cuda

Use cpu in place of cuda when needed. The models stage fits the matched Density31 comparator and makes the declared frozen-parameter/readout controls. Their fitting policies and active reference budgets are in native_unified_ablation_catalogue.csv. The all-evaluate stage replays fitted models only. For the complete nine-row ECG ablation replay:

    python experiments/evaluate_unified_adfecg.py --model results/0.3.1/adfecg_model.npz --output-dir results/0.3.1

The ECG evaluator follows the five relative comparison models declared in the main model. Detector-changing comparisons use their own fitted detectors; readout comparisons reuse the main model's freshly computed event stream and saved training parameters.

## Reference parameters and new fits

The benchmark native model uses two fitted reference calibrations, then computes query source features from raw measurements with the current actual-time operator. To reconstruct that model from its supplied reference parameter archive:

    python experiments/fit_unified_source.py --dataset native --stage fit --source-stage reuse --calibrations results/0.3.1/native_reference_calibration_model.npz --model results/0.3.1/unified_model.npz --device cpu

The reference archive is self-contained: per-calibration coefficients, source-operator flags, observed-information knots, source identities and exclusion masks are stored together. Reuse validates declared sources and splits. Replaying a fitted reference archive and learning new reference coefficients are distinct operations. The distributed benchmark bank used foreground periods and optical background-peak supervision. Its refined background head also used four matched waveform views derived from measured foreground/background components, with view shares 0.5 for the native view and 0.125 for each matched view. The 2927 source-view rows represent 635 distinct recordings in 13 runs; the foreground regularization denominator remains 387. The base background head used 506 qualified recordings in 17 runs. Both heads exclude the query records and independent units in every held-out case.

The original background-label and view resources are included under data/sota_dev/recomputed_clean/: lemna_task_period_background_density_fit.npz, matched_peer_background/features.npz and matched_peer_background/background_fit.npz. Current model metadata retains their supervision manifests and provenance. Default cold fitting without background labels uses a No-peer proxy. It is a separate reference recipe and does not reproduce the benchmark bank's optical background supervision.

For a new period-reference dataset:

    python experiments/fit_unified_source.py --input INPUT.parquet --references REFERENCES.csv --period-min LOWER --period-max UPPER --time-unit UNIT --output-dir MODEL_DIRECTORY --device cpu

The reference CSV has one row per observed trace and supplies trace_id, reference_period, qualified and an independent split_id. It may include background_period. All observations of a run share one independent unit. --calibration-recipe explicitly selects alternative reference-calibration recipes. The default --source-stage fit learns new coefficients from supplied No/reference data. The automatic recipe uses two calibrations when every requested training context has a nonzero range of observed harmonic information; otherwise it fits one calibration consistently across the contexts. --full-only fits a deployment-only model.

For measured-source landmark training, supply --reference-waveforms, --reference-landmarks and optional --reference-intervals, with source grouping and independent split IDs. EDF+ annotations can instead be supplied through --reference-annotations, --record-metadata and --window-duration. The built-in ECG preparation has these inputs for all five subjects. Training landmarks and direct-source waveforms stay within each training fold; prediction accepts measured abdominal signals and timestamps alone. The distributed adfecg_model.npz and relative detector .joblib files support prediction and held-out replay.

## Fixed evaluations

| Evaluation | Input and references | Source |
| --- | --- | --- |
| Native mixed reporters | data/sota_dev/recomputed_clean/lemna_task_period_acf_input.parquet and lemna_task_period_supervised_context.npz | Watanabe et al. 2023, doi:10.1093/plphys/kiad218; 777 measurements, 506 qualified period references |
| Controlled mixtures | data/sota_dev/final_model_controlled/inputs.parquet and truth_cells.csv | Fixed prepared construction; all 2304 measurements retained in native, shorter and missing acquisitions |
| Direct single reporter | data/sota_dev/recomputed_clean/lemna2021_single_reporter_input.parquet and references CSV | Watanabe et al. 2021, doi:10.1093/pcp/pcab037; Green-pass worksheets; 124 inputs, 114 references |
| Dual reporter transfer | data/external/lemna_watanabe2021/dual_reporter_observations.parquet and author_period_references.csv | Same 2021 study; 185 inputs, 60 reference-eligible recordings, actual observation windows |
| Recorded SCN mixtures | data/external/shan2020_semisynthetic/input_PHASE_lLENGTH.parquet and mixture_records.csv | Shan et al. 2020, doi:10.1016/j.neuron.2020.07.012; all six phase/acquisition strata |
| Coupled clocks | data/external/wang2026_mechanism/inputs.parquet and truth_cells.csv | Wang et al. 2026, doi:10.1038/s41467-025-68218-x; realized finite-window phase-period truths |
| Fetal ECG | data/external/adfecgdb/inputs.parquet, references.csv, direct-source waveforms and EDF+ expert annotations | PhysioNet ADFECGDB 1.0.0, doi:10.13026/C2RP4B; 600 sensor measurements in 150 windows from five subjects |

The two reference-ineligible ECG windows remain in prediction outputs. Mean-RR accuracy first aggregates simultaneous measurements to their shared source window, then averages within subjects and across subjects. There are 148 eligible source windows and 592 corresponding sensor measurements. Change accuracy reports all possible adjacent-window pairs, reference failures and finite-prediction counts.

Classical predictions on unchanged inputs retain their recorded sources. experiments/predict_classical_transfer.py additionally evaluates the full native dual-reporter observations and both changed controlled acquisitions. Its official Java MESA/mFourFit inputs preserve actual times and every observed value. Historical methods lacking a same-input prediction remain explicitly unavailable in those comparison rows.

## Outputs

Scientific tables and fitted models are in results/0.3.1. kernel_validation.json reports moment, probability, complete-cohort batch, time-unit and public-CLI agreement. deployment_validation.json records installed-wheel checks. Models retain parameter provenance, convergence, source identities, reference budgets and readout definitions.

Prepared observations, fixed truths and attribution accompany the numerical data asset. Original upstream archives are available from the sources listed in DATA_SOURCES.md. Component-derived training resources accompany the fitted optical reference bank. Original code uses MIT. ECG data use ODC Attribution 1.0. Other data and third-party predictions retain the terms recorded with their source materials.