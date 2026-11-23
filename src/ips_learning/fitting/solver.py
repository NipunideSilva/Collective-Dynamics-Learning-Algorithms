import numpy as np
from ips_learning.fitting.diagnostics import MatrixDiagnostics


class LinearSystemSolver:
    """
    Robust solver for Stage 2.3.

    Supports:
    - least squares
    - ridge on normal equations
    - truncated SVD / pseudoinverse
    """

    def __init__(self, method: str = "lstsq",
                 ridge_lambda: float =  None,#float = 1e-6,
                 svd_tol: float = 1e-12):
        
        self.method = method
        self.ridge_lambda = ridge_lambda
        self.svd_tol = svd_tol
        self._diagnostics = MatrixDiagnostics()

    def solve(self, A: np.ndarray, b: np.ndarray):
        """
        Parameters
        ----------
        A : np.ndarray
            Normal matrix or dense design-derived system matrix.
        b : np.ndarray
            Right-hand side.

        Returns
        -------
        theta : np.ndarray
        diagnostics : SolverDiagnostics
        """
        if self.method == "lstsq":
            theta, *_ = np.linalg.lstsq(A, b, rcond=None)
            diag = self._diagnostics.diagnose(A, method="lstsq")
            return theta, diag

        if self.method == "ridge":
            P = A.shape[1]
            theta = np.linalg.solve(A + self.ridge_lambda * np.eye(P), b)
            diag = self._diagnostics.diagnose(
                A,
                method="ridge",
                ridge_lambda=self.ridge_lambda,
            )
            return theta, diag

        if self.method == "svd":
            U, s, Vt = np.linalg.svd(A, full_matrices=False)
            s_inv = np.array([1 / x if x > self.svd_tol else 0.0 for x in s])
            A_pinv = Vt.T @ np.diag(s_inv) @ U.T
            theta = A_pinv @ b
            diag = self._diagnostics.diagnose(
                A,
                method="svd",
                svd_tol=self.svd_tol,
            )
            return theta, diag

        raise ValueError(f"Unknown solver method: {self.method}")