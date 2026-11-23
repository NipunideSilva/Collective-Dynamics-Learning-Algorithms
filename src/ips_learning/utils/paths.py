from pathlib import Path
#It helps every script consistently locate the project root folder and 
# the artifacts/output folder, regardless of where code is executed from.

def project_root() -> Path:
    """
    Resolve the project root assuming this file lives at:
    src/ips_learning/utils/paths.py
    """
    return Path(__file__).resolve().parents[3] #__file__ is current file path, resolve() gives absolute path, parents[3] goes up 3 levels to project root


def artifacts_dir() -> Path:
    path = project_root() / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def datasets_dir() -> Path:
    path = artifacts_dir() / "datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def fits_dir() -> Path:
    path = artifacts_dir() / "fits"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    path = artifacts_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def rollouts_dir() -> Path:
    path = artifacts_dir() / "rollouts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def evaluations_dir() -> Path:
    path = artifacts_dir() / "evaluations"
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir() -> Path:
    path = artifacts_dir() / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def experiments_dir() -> Path:
    path = artifacts_dir() / "experiments"
    path.mkdir(parents=True, exist_ok=True)
    return path