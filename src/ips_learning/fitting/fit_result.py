# Defines structured output artifacts.
# one object representing “coefficients + basis + diagnostics.”
from dataclasses import dataclass
import numpy as np


@dataclass
class SolverDiagnostics:
    condition_number: float
    singular_values: np.ndarray
    rank: int
    min_singular: float
    max_singular: float
    method: str
    ridge_lambda: float | None = None
    svd_tol: float | None = None


@dataclass
class FitResult:
    """
    Output of Stage 2.3.

    Attributes
    ----------
    theta : np.ndarray
        Full fitted coefficient array.
    n_radial : int
        Number of interaction-kernel coefficients.

    n_environment : int
        Number of environmental coefficients.

        Second-order:
            coefficients for f(V)

        First-order:
            coefficients for f(X)
    basis_bundle : object
        The Stage 2.2 basis bundle used to build the fit.
    A : np.ndarray
        Normal-equation matrix.

    b : np.ndarray
        TNormal-equation RHS.

    diagnostics : SolverDiagnostics
        Numerical diagnostics for the solve.

    metadata : dict
        Fit metadata.
    """
    theta: np.ndarray
    n_radial: int
    n_environment: int
    
    basis_bundle: object
    diagnostics: SolverDiagnostics
    metadata: dict
    n_energy: int =0
    n_alignment: int =0
    A: np.ndarray | None = None
    b: np.ndarray | None = None


    @property
    def theta_f(self) -> np.ndarray:
        """
        Environmental coefficients.

        Returns an empty array when the candidate
        does not include an environment force.
        """

        include_environment = self.metadata.get(
            "include_environment",
            True,
        )

        if not include_environment or self.n_environment == 0:
            return np.zeros(0, dtype=float)

        if self.has_energy_alignment:
            return self.theta[
                self.n_energy + self.n_alignment:
            ]

        return self.theta[self.n_radial:]
    
    @property
    def interaction_type(self) -> str:
        return self.metadata.get("interaction_type", "energy")
    
    @property
    def include_environment(self) -> bool:
        return bool(
            self.metadata.get(
                "include_environment",
                True,
            )
        )


    @property
    def has_energy_alignment(self) -> bool:
        return self.interaction_type == "energy_alignment"
    
    @property
    def has_environment(self) -> bool:
        return (
            self.include_environment
            and self.n_environment > 0
        )


    @property
    def theta_phi_E(self) -> np.ndarray:
        if self.has_energy_alignment:
            return self.theta[:self.n_energy]
        if self.interaction_type == "energy":
            return self.theta[:self.n_radial]
        return np.zeros(self.n_radial)


    @property
    def theta_phi_A(self) -> np.ndarray:
        if self.has_energy_alignment:
            return self.theta[self.n_energy:self.n_energy + self.n_alignment]
        if self.interaction_type == "alignment":
            return self.theta[:self.n_radial]
        return np.zeros(self.n_radial)


    @property
    def theta_phi(self) -> np.ndarray:
        """
        Backward-compatible single interaction block.
        """
        return self.theta[:self.n_radial]

       
