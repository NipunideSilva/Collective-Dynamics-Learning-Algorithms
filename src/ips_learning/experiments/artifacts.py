from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunArtifacts:
    run_dir: Path
    fit_dir: Path
    model_dir: Path
    rollout_dir: Path
    evaluation_dir: Path
    figures_dir: Path