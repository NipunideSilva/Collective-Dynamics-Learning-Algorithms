# Reproducing the Numerical Results

This document describes how the numerical experiments in `ips-learning` are organized and how the main result families are reproduced.

The repository separates reusable numerical code from experiment orchestration:

```text
src/ips_learning/   reusable simulation, learning, evaluation, and analysis code
notebooks/          experiment orchestration and publication workflows
artifacts/          generated datasets, fits, evaluations, figures, and tables
```

There is currently no dedicated command-line interface for reproducing the complete manuscript. The primary publication workflows are orchestrated through the notebooks, while the underlying numerical stages are implemented as reusable Python APIs.

## 1. Installation

Python 3.10 or newer is required.

Create an isolated environment and install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install ".[test]"
```

Run the package tests:

```bash
pytest -q
```

For interactive notebook execution, use a Jupyter environment in addition to the package dependencies.

For example:

```bash
python -m pip install jupyterlab
jupyter lab
```

## 2. Reproduction philosophy

The numerical workflow is divided into distinct stages:

```text
data generation
      ↓
feature construction
      ↓
basis construction / localization
      ↓
least-squares fitting
      ↓
learned dynamical model
      ↓
trajectory rollout
      ↓
evaluation
      ↓
repeated-trial / robustness analysis
      ↓
publication figures and tables
```

The code under `src/ips_learning/` implements these stages. The notebooks configure the experiments and connect the stages.

Generated scientific artifacts should be reused when only figures or tables need to be regenerated. It is not necessary to rerun learning experiments merely to regenerate presentation outputs.

## 3. Artifact layout

Several repository classes use stage-specific artifact roots.

Core numerical artifacts are organized under roots such as:

```text
artifacts/datasets/
artifacts/features/
artifacts/fits/
artifacts/rollouts/
```

Within these repositories, artifacts are identified by the canonical mathematical system and, when applicable, the experimental profile.

Conceptually:

```text
<artifact-root>/<system>/<profile>/...
```

For example, dataset artifacts follow the profile-aware repository structure under:

```text
artifacts/datasets/<system>/<profile>/<split>/
```

The same system/profile distinction is used by the feature, fit, and rollout repositories.

Publication figures and tables use system/profile-aware locations such as:

```text
artifacts/<system>/<profile>/figures/<stage>/
artifacts/<system>/<profile>/tables/<stage>/
```

Cross-system results use a study-level namespace:

```text
artifacts/studies/<study_name>/
```

Examples include:

```text
artifacts/studies/scalability/
artifacts/studies/model_selection/
artifacts/studies/training_data_dependence/
```

Historical artifacts retained for provenance are stored under:

```text
artifacts/_legacy/
```

Files under `_legacy` are not active workflow inputs or output targets.

## 4. System names and profiles

A system name identifies the mathematical model.

A profile identifies a complete experimental or data-generation configuration.

The canonical core system names are:

```text
kuramoto
levi
spp
```

The Phototaxis system uses the canonical package system name `levi`.

SPP configurations such as velocity-grid, clumped, swarm, or milling-style experiments are represented by profiles rather than by creating new mathematical system names.

Legacy aliases may still be recognized by the system registry, but new artifacts should use canonical system names.

## 5. Data generation

### 5.1 Generic generators

The reusable first-order generator is:

```python
from ips_learning.experiments.generate_first_order_data import (
    run_one_first_order_dataset,
)
```

The reusable second-order generator is:

```python
from ips_learning.experiments.generate_second_order_data import (
    run_one_second_order_dataset,
)
```

These functions generate trajectory data, construct metadata, and save the resulting dataset through `DatasetRepository`.

A generated raw trajectory dataset contains quantities such as:

```text
t
X
V       # second-order systems
LHS
U0
metadata
```

### 5.2 System-specific profiles

Published experiment configurations are encoded in system-specific profile builders.

Kuramoto:

```python
from ips_learning.experiments.generate_kuramoto_data import (
    build_kuramoto_dataset_profile,
    run_one_kuramoto_dataset,
)
```

Phototaxis / Levi:

```python
from ips_learning.experiments.generate_levi_data import (
    build_levi_dataset_profile,
)
```

SPP:

```python
from ips_learning.experiments.generate_spp_data import (
    build_spp_dataset_profile,
    run_one_spp_dataset,
)
```

Whenever the goal is to reproduce an existing experiment, prefer the corresponding profile configuration rather than manually re-entering numerical parameters.

### 5.3 Primary data-generation notebooks

The main system notebooks are:

```text
notebooks/kuramoto.ipynb
notebooks/phototaxis.ipynb
notebooks/spp.ipynb
notebooks/spp_swarm.ipynb
```

Each begins with system configuration and dataset preparation before proceeding to learning.

## 6. Feature construction

Trajectory data are converted into learning features using:

```python
from ips_learning.features.feature_builder import FeatureBuilder

features = FeatureBuilder().build(dataset)
```

The feature representation supplies the quantities required by the normal-equation assembly, including pairwise distances and relative state information.

Feature artifacts may also be stored and reloaded through the feature repository for larger experiments.

## 7. Basis construction

Hypothesis spaces are configured through basis specifications.

The standard nonparametric formulation uses radial B-splines for interaction kernels and B-spline or tensor-product B-spline representations for environmental forces.

Typical construction begins with:

```python
from ips_learning.bases.specs import (
    BasisConfig,
    RadialBasisSpec,
    EnvironmentBasisSpec,
)

from ips_learning.bases.builders import BasisBuilder
```

A `BasisConfig` determines the basis family, polynomial degree, and number of basis functions.

The basis is localized using the observed training features.

## 8. Fitting

Normal equations are assembled from the features and basis bundle.

The principal fitting components are:

```python
from ips_learning.fitting.assembler import NormalEquationAssembler
from ips_learning.fitting.solver import LinearSystemSolver
```

A typical solver configuration is:

```python
solver = LinearSystemSolver(
    method="ridge",
    ridge_lambda=1e-8,
    svd_tol=1e-12,
)
```

The fitting stage returns a fit result containing learned coefficients, basis information, metadata, and matrix diagnostics.

Diagnostics include quantities such as:

```text
rank
condition number
minimum singular value
maximum singular value
solver method
regularization parameter
SVD tolerance
```

These diagnostics are retained for numerical analysis but are not necessarily included in every publication table.

## 9. Learned-system reconstruction

Learned coefficients are converted back into an executable dynamical system using:

```python
from ips_learning.models.factories import LearnedModelFactory
```

The learned model exposes the recovered interaction kernel and environmental force and can be integrated as a dynamical system.

## 10. Rollout

Learned dynamics are simulated using:

```python
from ips_learning.rollout.simulator import LearnedSystemRollout
```

The main rollout scenarios used in the experiments include:

```text
training-fit reconstruction
full-training prediction
unseen-test prediction
changed-agent-number prediction
extended-horizon prediction
```

Rollouts may be saved through `RolloutRepository` when they are needed by later evaluation or visualization stages.

## 11. Evaluation

The package evaluates learned models at three levels.

### 11.1 Residual error

Residual metrics measure agreement between observed derivatives and derivatives predicted by the learned mechanisms.

Implemented under:

```text
src/ips_learning/evaluation/residuals.py
```

### 11.2 Mechanism recovery

Feature-recovery metrics compare learned interaction kernels and environmental forces with their known true functions in synthetic experiments.

Implemented under:

```text
src/ips_learning/evaluation/function_errors.py
```

These errors are evaluated with empirical weights derived from the training data.

The relevant empirical density is constructed from the same training or train-fit features used for learning.

Reference densities generated from large auxiliary samples are used for visualization and support comparison; they are not the weighting measure used for the reported feature-recovery errors.

### 11.3 Trajectory recovery

Trajectory metrics compare true and learned trajectories and distinguish reconstruction from prediction when a fitting-time boundary is supplied.

Implemented under:

```text
src/ips_learning/evaluation/trajectory_errors.py
```

## 12. One-run baseline evaluation

The notebook

```text
notebooks/error_analysis/01_one_run_baseline_analysis.ipynb
```

loads previously generated datasets, features, fits, and rollouts and evaluates one representative run for:

```text
Phototaxis / Levi
SPP
Kuramoto
```

The reusable implementation is:

```python
from ips_learning.analysis.baseline_evaluation import (
    GeneralRunOneBaseline,
)
```

This is an evaluation workflow and therefore assumes that the underlying learning artifacts have already been generated.

## 13. Repeated trials and agent-number scaling

The primary fully nonparametric workflow is implemented by:

```python
from ips_learning.analysis.scalability_trial_runner import (
    ScalabilityTrialConfig,
    ScalabilityTrialRunner,
)
```

For each trial, the runner:

```text
loads the train-fit dataset
loads the full training dataset
loads the test dataset
loads the changed-N test dataset
builds features
builds the basis
fits the model
constructs learned systems
computes mechanism and residual errors
performs trajectory rollouts
computes trajectory metrics
```

The corresponding notebook is:

```text
notebooks/error_analysis/02_repeated_trials_N_scaling.ipynb
```

The semi-parametric counterpart uses:

```python
from ips_learning.analysis.scalability_trial_runner_parametric import (
    ParametricScalabilityTrialConfig,
    ParametricScalabilityTrialRunner,
)
```

with notebook:

```text
notebooks/error_analysis/02_repeated_trials_N_scaling_parametric.ipynb
```

The standard scaling experiment trains on \(N\) agents and evaluates the learned mechanism on a larger system.

The separate SPP milling/swarm experiment uses its own configuration and extended prediction horizon; it should not be replaced by the standard scaling configuration.

## 14. Dependence on training-data size

The \(M,L\) study varies:

- \(M\): number of independent training trajectories;
- \(L\): number of temporal observations.

The main analysis notebook is:

```text
notebooks/error_analysis/03_ML_trial_sweep_analysis.ipynb
```

The sweep results are stored as evaluation summaries and reused for publication.

The cross-system heatmaps are generated by:

```text
notebooks/outputs/sample_complexity_check.ipynb
```

That notebook reads existing sweep summaries rather than rerunning the learning experiments.

The active combined outputs are stored under:

```text
artifacts/studies/training_data_dependence/
```

## 15. Noise robustness

The reusable noise runner is:

```python
from ips_learning.analysis.noise_sweep_runner import (
    NoiseSweepConfig,
    NoiseSweepRunner,
)
```

A separate runner evaluates filtering approaches:

```python
from ips_learning.analysis.noise_filter_sweep_runner import (
    NoiseFilterSweepConfig,
    NoiseFilterSweepRunner,
)
```

The principal notebook is:

```text
notebooks/error_analysis/04_noise_sweep_analysis.ipynb
```

Additional visualization diagnostics are available in:

```text
notebooks/outputs/noise_check.ipynb
```

When comparing error across noise levels, the empirical feature-error weighting measure is held fixed using the clean training-data features. This prevents changes in the weighting distribution from being confused with changes in recovery accuracy.

## 16. Model selection

Model selection is implemented as a separate reusable subsystem under:

```text
src/ips_learning/model_selection/
```

Candidate models are represented by:

```python
from ips_learning.model_selection.candidates import (
    ModelCandidate,
    get_candidates,
)
```

The primary single-experiment runner is:

```python
from ips_learning.model_selection.runner import ModelSelectionRunner
```

It supports:

```text
candidate fitting
candidate rollout
candidate evaluation
candidate scoring
model selection
selected-model test evaluation
```

Repeated-trial model selection is implemented by:

```python
from ips_learning.model_selection.repeated_trial_runner import (
    ModelSelectionRepeatedTrialRunner,
)
```

Repeated-trial results are summarized using:

```python
from ips_learning.model_selection.repeated_trial_summary import (
    ModelSelectionRepeatedTrialSummaryBuilder,
)
```

The model-selection experiments are orchestrated by:

```text
notebooks/model_selection/01_sppcs_model_selection.ipynb
notebooks/model_selection/02_levi_model_selection.ipynb
notebooks/model_selection/03_cs_model_selection.ipynb
notebooks/model_selection/04_op_model_selection.ipynb
```

Repeated-trial summaries are constructed in:

```text
notebooks/model_selection/05_repeated_trials_model_selection_summary.ipynb
```

The model-selection procedure uses reconstruction, residual, validation-trajectory, and model-complexity information.

AIC and BIC are retained as additional comparison criteria.

## 17. Model-selection website

The static website assets are generated by:

```text
notebooks/model_selection/06_build_website_assets.ipynb
```

The reusable exporter is:

```python
from ips_learning.model_selection.static_website_export import (
    StaticModelSelectionWebsiteExporter,
)
```

The full model-selection summary artifacts retain the available diagnostics.

The website intentionally displays a smaller set of thesis-relevant columns rather than exposing every stored diagnostic in the primary table.

## 18. Limitation and sensitivity studies

### Initial-condition range

```text
notebooks/limitations/ic_range.ipynb
```

This study compares different SPP initial-condition profiles.

Because the initial-condition distribution itself is the object of study, profile-specific empirical densities are constructed and visualized.

### Basis size

```text
notebooks/limitations/change_of_basis.ipynb
```

This study varies basis dimensions while keeping the training data and evaluation measure fixed.

### Parameter misspecification

```text
notebooks/limitations/parameter_misspecification.ipynb
```

This study examines recovery when model parameters are misspecified relative to the data-generating system.

## 19. Publication figures

System-specific visualization notebooks include:

```text
notebooks/outputs/kuramoto_visualizations_and_tables.ipynb
notebooks/outputs/levi_visualizations_and_tables.ipynb
notebooks/outputs/spp_visualizations_and_tables.ipynb
notebooks/outputs/spp_swarm_visualizations_and_tables.ipynb
```

These consume previously generated artifacts to produce:

```text
support and density diagnostics
learned mechanism plots
fully nonparametric / semi-parametric comparisons
trajectory reconstruction and prediction figures
```

Extended evaluation/publication notebooks are:

```text
notebooks/outputs/kuramoto_EA_vizualizations and tables.ipynb
notebooks/outputs/levi_EA_vizualizations and tables.ipynb
notebooks/outputs/spp_EA_vizualizations and tables.ipynb
```

They produce repeated-trial summaries, error plots, scaling figures, and LaTeX tables.

## 20. Publication tables

The principal fully nonparametric versus semi-parametric comparison tables are formatted by:

```text
notebooks/error_analysis/table_creater.ipynb
```

The notebook reads existing repeated-trial results and writes the active LaTeX publication tables under:

```text
artifacts/<system>/<profile>/tables/publication/
```

The current publication tables report scientific performance quantities while retaining numerical-conditioning diagnostics in the underlying evaluation data.

The \(M,L\) longtables are generated from the corresponding repeated-trial sweep summaries in the extended evaluation notebooks.

Data-generation parameter tables, learning-parameter tables, and initial-condition tables in the manuscript/thesis are not generated by `table_creater.ipynb`; they are maintained separately from these result-table workflows.

## 21. Recommended reproduction order

For a complete system experiment, use this dependency order:

```text
1. Generate datasets
2. Construct features
3. Construct and localize bases
4. Fit the model
5. Construct the learned system
6. Generate reconstruction/prediction rollouts
7. Evaluate residual, mechanism, and trajectory errors
8. Run repeated-trial or robustness studies
9. Generate publication figures
10. Generate publication tables
```

For the three principal systems, begin with:

```text
notebooks/kuramoto.ipynb
notebooks/phototaxis.ipynb
notebooks/spp.ipynb
```

Then use the relevant notebooks under:

```text
notebooks/error_analysis/
notebooks/outputs/
```

For model-selection results, use the independent workflow under:

```text
notebooks/model_selection/
```

## 22. Regenerating presentation outputs without rerunning experiments

When validated numerical artifacts already exist, publication figures and tables should be regenerated from those artifacts rather than rerunning the learning experiment unnecessarily.

Examples include:

```text
table_creater.ipynb
sample_complexity_check.ipynb
system visualization notebooks
model-selection website export
```

This keeps scientific computation separate from presentation generation and avoids accidentally changing validated numerical results.

## 23. Reproducibility check

The package has been tested using a clean wheel installation outside the repository.

The fresh-install smoke test successfully executed:

```text
Kuramoto data generation
→ dataset repository reload
→ feature construction
→ basis construction
→ least-squares fitting
→ learned-model reconstruction
→ residual evaluation
→ unseen-data rollout
→ trajectory evaluation
```

This verifies that the installed package can execute a complete learning workflow without importing modules directly from the source checkout.

## 24. Notebook navigation

For a notebook-by-notebook description, see:

```text
notebooks/README.md
```

For installation and a compact package example, see:

```text
README.md
```
