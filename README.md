# IPS Learning

**Simultaneous inference of environmental and interaction forces in collective dynamics**

`ips-learning` is a Python research package for learning governing mechanisms in interacting particle systems from trajectory data. The package is designed for collective systems in which observed dynamics arise from a combination of pairwise interactions and environmental, self-propulsion, or other intra-agent forces.

The framework uses the known structure of collective dynamics to reduce a high-dimensional system-identification problem to the recovery of low-dimensional interaction kernels and environmental forces. It supports fully nonparametric variational learning, semi-parametric environmental-force models, trajectory reconstruction and prediction, error analysis, scalability and robustness studies, and model-selection experiments.

## Mathematical models

### First-order systems

The first-order model has the form

\[
\dot{x}_i
=
f(x_i)
+
\frac{1}{N}
\sum_{j=1}^{N}
\phi(r_{ij})(x_j-x_i),
\qquad
r_{ij}=\|x_j-x_i\|.
\]

Here,

- \(\phi\) is the pairwise interaction kernel;
- \(f\) is an environmental or intra-agent force;
- \(N\) is the number of agents.

The learning problem is to recover \(\phi\) and \(f\) from observed trajectories.

### Second-order systems

For second-order dynamics,

\[
\dot{x}_i=v_i,
\]

\[
\dot{v}_i
=
f(v_i)
+
\frac{1}{N}
\sum_{j=1}^{N}
\phi_E(r_{ij})(x_j-x_i)
+
\frac{1}{N}
\sum_{j=1}^{N}
\phi_A(r_{ij})(v_j-v_i).
\]

The terms represent

- \(\phi_E\): energy-type attraction/repulsion interactions;
- \(\phi_A\): velocity-alignment interactions;
- \(f\): environmental, frictional, or self-propulsion effects.

Candidate models may contain energy interactions, alignment interactions, environmental forces, or combinations of these mechanisms.

## Learning approaches

The package supports two main representations of the environmental force:

- **Fully nonparametric:** both interaction kernels and environmental forces are learned using basis expansions.
- **Semi-parametric:** interaction kernels remain nonparametric while the environmental force is represented using a specified parametric basis.

Radial interaction kernels are represented using B-spline bases. Nonparametric environmental forces use B-spline or tensor-product B-spline bases. Coefficients are estimated through variational least-squares systems with diagnostics for rank, conditioning, and singular values.

The learned mechanisms can then be assembled into dynamical systems for trajectory reconstruction and prediction.

## Systems

### Core system registry

The canonical system factory currently exposes:

| System | Package name | Order | Main mechanism |
| --- | --- | ---: | --- |
| Kuramoto | `kuramoto` | First | Phase synchronization |
| Phototaxis / Levi | `levi` | Second | Velocity alignment with environmental forcing |
| Self-propelled particles | `spp` | Second | Attraction-repulsion with self-propulsion |

SPP experimental configurations are distinguished using profiles rather than separate system names.

Legacy aliases such as `spp_swarm` and `spp_rings` are mapped internally to the canonical `spp` system.

### Model-selection experiments

The model-selection studies additionally include collective-dynamics examples based on:

- Cucker-Smale dynamics;
- self-propelled particles with velocity alignment (SPPCS);
- opinion dynamics;
- Phototaxis.

These experiments are organized separately from the core system registry.

## Installation

Python 3.10 or newer is required.

Clone the repository and create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

To install the test dependencies:

```bash
python -m pip install ".[test]"
```

Run the test suite with:

```bash
pytest -q
```

The package has also been verified by building a wheel, installing it into a clean environment, and running a fresh end-to-end learning workflow without importing source files directly from the repository.

## Minimal example

The following example generates a small Kuramoto dataset and fits both an interaction kernel and an environmental force.

It is intended as a quick package demonstration rather than as a reproduction of the numerical experiments used in the manuscript.

```python
from pathlib import Path
import numpy as np

from ips_learning.experiments.generate_kuramoto_data import (
    run_one_kuramoto_dataset,
)
from ips_learning.data.dataset_repository import DatasetRepository
from ips_learning.features.feature_builder import FeatureBuilder
from ips_learning.bases.specs import (
    BasisConfig,
    RadialBasisSpec,
    EnvironmentBasisSpec,
)
from ips_learning.bases.builders import BasisBuilder
from ips_learning.fitting.assembler import NormalEquationAssembler
from ips_learning.fitting.solver import LinearSystemSolver

artifact_root = Path("artifacts")
dataset_root = artifact_root / "datasets"

run_one_kuramoto_dataset(
    split="train",
    dataset_name="dataset_train_example",
    N=4,
    D=1,
    M=4,
    L=31,
    t0=0.0,
    T=0.3,
    seed=1701,
    ic_generator_name="kuramoto_uniform",
    ic_params={"low": 0.0, "high": 2.0 * np.pi},
    system_params={
        "K": 2.0,
        "eps_r": 1e-12,
        "wrap_angles": True,
    },
    profile_name="profile1",
    root_dir=dataset_root,
)

dataset_repo = DatasetRepository(root_dir=dataset_root)

train = dataset_repo.load(
    system_name="kuramoto",
    profile_name="profile1",
    split="train",
    dataset_name="dataset_train_example",
)

features = FeatureBuilder().build(train)

basis_config = BasisConfig(
    radial=RadialBasisSpec(
        family="bspline",
        degree=1,
        n_basis=4,
    ),
    environment=EnvironmentBasisSpec(
        family="bspline",
        degree=1,
        n_basis_per_dim=4,
    ),
)

basis_bundle = BasisBuilder(basis_config).build(features, None)

solver = LinearSystemSolver(
    method="lstsq",
    ridge_lambda=1e-8,
    svd_tol=1e-12,
)

assembler = NormalEquationAssembler(
    parallel_jobs=1,
    interaction_type="energy",
)

fit_result = assembler.fit(
    features=features,
    basis_bundle=basis_bundle,
    solver=solver,
    system_name="kuramoto",
)

print("rank:", fit_result.diagnostics.rank)
print("condition number:", fit_result.diagnostics.condition_number)
```

The full research workflows additionally include learned-system construction, trajectory rollout, residual evaluation, feature-recovery metrics, repeated trials, and prediction on unseen initial conditions.

## Repository structure

The Python package is organized by scientific stage:

```text
src/ips_learning/
├── analysis/          repeated trials, sweeps, scalability, and robustness
├── bases/             radial and environmental basis construction
├── data/              trajectory simulation and dataset repositories
├── evaluation/        residual, feature, and trajectory error metrics
├── experiments/       system-specific data-generation workflows
├── features/          feature extraction from trajectory data
├── fitting/           normal-equation assembly, solvers, and diagnostics
├── model_selection/   candidate-model fitting and model selection
├── models/            learned dynamical-system representations
├── rollout/           learned-model simulation and rollout repositories
├── statistics/        empirical measures and density construction
├── systems/           mathematical systems and system registry
├── utils/             shared utilities
└── visualization/     plotting and artifact repositories
```

The main separation of responsibilities is:

```text
systems      -> define the true dynamics
data         -> simulate and store trajectories
features     -> construct learning data
bases        -> define hypothesis spaces
fitting      -> estimate model coefficients
models       -> construct learned dynamical systems
rollout      -> simulate learned dynamics
evaluation   -> quantify recovery and prediction error
analysis     -> repeated experiments and robustness studies
visualization-> publication and analysis outputs
```

## Notebooks

Research notebooks are stored under `notebooks/`.

The top-level notebooks contain the primary system workflows:

```text
notebooks/
├── kuramoto.ipynb
├── phototaxis.ipynb
├── spp.ipynb
└── spp_swarm.ipynb
```

Additional notebook groups are organized as:

```text
notebooks/
├── error_analysis/     repeated trials, scaling, noise, and tables
├── limitations/        initial-condition and basis-sensitivity studies
├── model_selection/    candidate-model and website workflows
└── outputs/            publication visualizations and tables
```

Notebooks are primarily used for experiment orchestration, analysis, and publication output. Reusable numerical logic belongs in `src/ips_learning/`.

## Artifact organization

Generated data and results are stored under `artifacts/`.

System-specific artifacts use the canonical system/profile hierarchy:

```text
artifacts/<system>/<profile>/...
```

For example:

```text
artifacts/spp/spp_velocity_grid_clumps/
```

Cross-system studies use a separate study namespace:

```text
artifacts/studies/<study_name>/
```

Examples include:

```text
artifacts/studies/scalability/
artifacts/studies/model_selection/
artifacts/studies/training_data_dependence/
```

Historical outputs retained for provenance are stored separately under:

```text
artifacts/_legacy/
```

## Evaluation

The package provides metrics for three complementary aspects of learned models:

1. **Mechanism recovery** — weighted errors in learned interaction kernels and environmental forces.
2. **Residual accuracy** — agreement between observed derivatives and derivatives predicted by the learned model.
3. **Trajectory accuracy** — reconstruction and prediction errors obtained by simulating the learned dynamical system.

Feature-recovery errors are weighted using empirical measures derived from the training trajectory data so that errors are emphasized in regions explored by the observed dynamics.

## Research workflows

The repository contains workflows for:

- fully nonparametric learning;
- semi-parametric environmental-force learning;
- repeated-trial uncertainty analysis;
- scaling to different numbers of agents;
- dependence on the number of trajectories and temporal samples;
- robustness to observational noise;
- sensitivity to basis size and initial-condition distributions;
- model selection among competing collective-dynamics mechanisms;
- publication figure and table generation.

Detailed reproduction instructions and a notebook navigation guide are maintained separately from the root README.

## Research and Citation

This repository accompanies the research article:

> Nipuni de Silva, Ming Zhong, and James M. Greene,
> **“Simultaneous inference of environmental and interaction forces in collective dynamics.”**
> [arXiv:2608.25181, 2026](https://doi.org/10.48550/arXiv.2608.25181).

The manuscript develops variational learning methods for collective systems containing both interaction forces and environmental or intra-agent forces. The framework includes fully nonparametric and semi-parametric learning formulations, evaluation of mechanistic recovery and trajectory prediction, and a model-selection procedure for distinguishing candidate collective-dynamics mechanisms.

### Software DOI

Version `v0.1.0` is archived on Zenodo:

[https://doi.org/10.5281/zenodo.24538490](https://doi.org/10.5281/zenodo.24538490)

For the software project across all versions, use the Zenodo concept DOI:

[https://doi.org/10.5281/zenodo.24538489](https://doi.org/10.5281/zenodo.24538489)

[![DOI](https://zenodo.org/badge/1469077747.svg)](https://doi.org/10.5281/zenodo.24538489)

Machine-readable citation metadata are provided in [`CITATION.cff`](CITATION.cff).

## Acknowledgments

OpenAI's ChatGPT was used to assist with code review, repository cleanup,
documentation, and software-release preparation. The authors retain
responsibility for all scientific content and software.

## License

This project is released under the MIT License. See `LICENSE`.
