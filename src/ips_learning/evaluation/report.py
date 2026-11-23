#Structured evaluation output object

from dataclasses import dataclass
import numpy as np


@dataclass
class EvaluationReport:
    """
    Structured output of Stage 2.6.

    Attributes
    ----------
    residual_metrics : dict
    function_metrics : dict
    trajectory_metrics : dict
    metadata : dict
    """
    residual_metrics: dict
    function_metrics: dict
    trajectory_metrics: dict
    metadata: dict