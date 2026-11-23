from dataclasses import dataclass, asdict
from typing import Optional


@dataclass(frozen=True)
class DatasetRef:
    system_name: str
    split: str
    dataset_name: str


@dataclass(frozen=True)
class FitConfig:
    basis_config_name: str
    solver_method: str = "ridge"
    ridge_lambda: float = 1e-6
    svd_tol: float = 1e-12
    parallel_jobs: int = 1


@dataclass(frozen=True)
class RolloutConfig:
    method: str = "Radau"
    rtol: float = 1e-3
    atol: float = 1e-6
    run_reconstruction: bool = True
    run_prediction: bool = True
    prediction_horizon_multiplier: float = 4.0
    prediction_num_points: int = 401


@dataclass(frozen=True)
class EvaluationConfig:
    compute_residual: bool = True
    compute_function_errors: bool = True
    compute_trajectory_errors: bool = True
    save_pointwise_curves: bool = True


@dataclass(frozen=True)
class RunConfig:
    run_name: str
    dataset_ref: DatasetRef
    fit_config: FitConfig
    rollout_config: RolloutConfig
    evaluation_config: EvaluationConfig
    trial_index: int = 0
    notes: Optional[str] = None

    def to_dict(self):
        return asdict(self)