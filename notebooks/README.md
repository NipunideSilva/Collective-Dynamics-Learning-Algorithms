# Notebook Guide

The notebooks in this directory orchestrate the numerical experiments, evaluation studies, and publication outputs associated with `ips-learning`.

Reusable simulation, fitting, evaluation, and repository logic belongs in `src/ips_learning/`. The notebooks primarily configure experiments, call the package APIs, inspect saved artifacts, and generate figures and tables.

## Main system workflows

The top-level notebooks implement the primary learning workflow for the core systems.

| Notebook | System | Purpose |
| --- | --- | --- |
| `kuramoto.ipynb` | Kuramoto | First-order synchronization model |
| `phototaxis.ipynb` | Phototaxis / Levi | Second-order alignment model with environmental forcing |
| `spp.ipynb` | Self-propelled particles | Second-order attraction-repulsion model with self-propulsion |
| `spp_swarm.ipynb` | SPP swarm / milling experiment | SPP configuration used for the extended swarm/scaling study |

These notebooks follow the same general sequence:

1. project and system configuration;
2. dataset preparation;
3. feature construction;
4. basis construction and localization;
5. empirical/reference density preparation;
6. fitting;
7. learned-model reconstruction;
8. trajectory rollout.

The top-level notebooks are the natural starting point for understanding how a complete system-specific experiment is assembled.

## `error_analysis/`

This directory contains cross-trial evaluation and robustness workflows.

### `01_one_run_baseline_analysis.ipynb`

Loads one fitted experiment for each primary system and computes the corresponding baseline evaluation metrics.

Systems:

- Phototaxis / Levi;
- SPP;
- Kuramoto.

This notebook assumes the relevant datasets, features, fits, and rollout artifacts have already been generated.

### `02_repeated_trials_N_scaling.ipynb`

Runs or analyzes repeated-trial experiments for the fully nonparametric formulation, including prediction after changing the number of agents.

The notebook contains sections for:

- Phototaxis / Levi;
- SPP;
- Kuramoto.

### `02_repeated_trials_N_scaling_parametric.ipynb`

Analogous repeated-trial and agent-scaling analysis for the semi-parametric environmental-force formulation.

### `03_ML_trial_sweep_analysis.ipynb`

Analyzes dependence of learning accuracy on:

- \(M\): number of independent trajectories;
- \(L\): number of temporal samples.

The study is performed for:

- Phototaxis / Levi;
- SPP;
- Kuramoto.

The combined publication heatmaps derived from these results are produced separately by `outputs/sample_complexity_check.ipynb`.

### `04_noise_sweep_analysis.ipynb`

Runs or analyzes learning under observational noise for:

- Phototaxis / Levi;
- SPP;
- Kuramoto.

The learning errors are evaluated using a fixed empirical measure constructed from the corresponding clean training data so that results across noise levels remain comparable.

### `table_creater.ipynb`

Constructs the main publication comparison tables for:

- Kuramoto;
- Phototaxis / Levi;
- SPP.

The active LaTeX outputs are written under each system/profile:

```text
artifacts/<system>/<profile>/tables/publication/
```

The notebook reads the existing repeated-trial evaluation summaries and formats them for publication. It does not generate the underlying experiments.

## `limitations/`

This directory contains experiments designed to study limitations and sensitivity of the learning procedure.

### `change_of_basis.ipynb`

SPP basis-size sensitivity study.

The notebook:

1. loads a fixed SPP dataset and feature set;
2. uses fixed empirical evaluation grids;
3. varies the interaction and environmental basis dimensions;
4. fits each basis configuration;
5. reloads the resulting fits;
6. compares feature-recovery errors.

Using fixed data and empirical measures isolates the effect of changing the hypothesis-space dimension.

### `ic_range.ipynb`

SPP initial-condition distribution study.

This notebook compares learning under different initial-condition profiles and examines how the region explored by the training trajectories affects recovery of the interaction kernel and environmental force.

Unlike the standard feature-error workflows, this study constructs profile-specific empirical densities because the purpose of the experiment is to compare the regions explored under different initial-condition distributions.

### `parameter_misspecification.ipynb`

SPP parameter-misspecification study.

The notebook investigates how the learned mechanisms respond when the assumed or supplied model parameters differ from those used to generate the data.

## `model_selection/`

These notebooks implement and analyze the candidate-model selection framework.

The system-specific notebooks are:

| Notebook | Benchmark |
| --- | --- |
| `01_sppcs_model_selection.ipynb` | Self-propelled particles with velocity alignment |
| `02_levi_model_selection.ipynb` | Phototaxis / Levi |
| `03_cs_model_selection.ipynb` | Cucker-Smale |
| `04_op_model_selection.ipynb` | Opinion dynamics |

Their workflow is organized into stages:

1. benchmark dataset preparation;
2. candidate-neutral feature and basis construction;
3. empirical train-fit density construction;
4. definition of candidate models;
5. candidate fitting, rollout, and evaluation;
6. loading saved model-selection outputs;
7. candidate diagnostics;
8. model comparison and selection.

The model-selection framework compares candidate mechanisms using reconstruction, residual, validation-trajectory, and complexity information. AIC and BIC are also retained as an additional comparison.

### `05_repeated_trials_model_selection_summary.ipynb`

Aggregates model-selection behavior across repeated trials.

### `06_build_website_assets.ipynb`

Exports the saved model-selection results required by the static model-selection website and verifies the resulting website assets.

The website presents a reduced set of thesis-relevant columns, while the underlying summary artifacts retain the full collection of model-selection diagnostics.

## `outputs/`

These notebooks are primarily for visualization, publication figures, and publication tables. They generally consume artifacts generated by the main system and analysis workflows rather than serving as the first stage of an experiment.

### System visualization notebooks

The standard visualization notebooks are:

```text
kuramoto_visualizations_and_tables.ipynb
levi_visualizations_and_tables.ipynb
spp_visualizations_and_tables.ipynb
spp_swarm_visualizations_and_tables.ipynb
```

They include:

- support and empirical-density diagnostics;
- learned interaction/environmental-force visualizations;
- algorithm-comparison figures;
- true-versus-learned trajectory visualizations.

### Evaluation and publication notebooks

The extended evaluation notebooks are:

```text
kuramoto_EA_vizualizations and tables.ipynb
levi_EA_vizualizations and tables.ipynb
spp_EA_vizualizations and tables.ipynb
```

These include publication-oriented analyses such as:

- one-run evaluation;
- trajectory-error plots;
- scalar metric summaries;
- interaction/environmental-force error figures;
- repeated-trial summaries;
- agent-number scaling;
- \(M,L\) dependence tables;
- LaTeX table generation.

### `sample_complexity_check.ipynb`

Combines the three-system \(M,L\) sweep results into the cross-system training-data-dependence heatmaps.

Active outputs are stored under the study-level namespace:

```text
artifacts/studies/training_data_dependence/figures/
```

This notebook reads existing sweep summaries; it does not rerun the learning experiments.

### `noise_check.ipynb`

Provides additional visual diagnostics for noisy observations, including comparisons of the observed states and derivative/left-hand-side quantities.

## Artifact conventions

### System-specific artifacts

System-specific results use:

```text
artifacts/<system>/<profile>/
```

The profile identifies the complete experimental/data-generation configuration without introducing additional mathematical system names.

Examples include:

```text
artifacts/kuramoto/profile1/
artifacts/levi/levi_velocity_grid/
artifacts/spp/spp_velocity_grid_clumps/
artifacts/spp/spp_swarm_spread/
```

### Cross-system studies

Outputs that summarize multiple mathematical systems use:

```text
artifacts/studies/<study_name>/
```

Examples include:

```text
artifacts/studies/scalability/
artifacts/studies/model_selection/
artifacts/studies/training_data_dependence/
```

A cross-system study should not use a synthetic system name such as `combined`.

### Historical artifacts

Historical artifacts retained for provenance are kept separately under:

```text
artifacts/_legacy/
```

These are not active workflow locations.

## Recommended reading order

For understanding the code and research workflow:

1. start with one top-level system notebook, preferably `kuramoto.ipynb`;
2. inspect the corresponding visualization notebook in `outputs/`;
3. examine `error_analysis/` for repeated-trial, scaling, \(M,L\), and noise studies;
4. inspect `model_selection/` for candidate-model comparison;
5. use `limitations/` for sensitivity and limitation studies.

For reproducing the complete numerical results rather than browsing the notebooks interactively, see:

```text
../docs/REPRODUCING_RESULTS.md
```