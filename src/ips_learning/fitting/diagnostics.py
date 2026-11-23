# Check the conditioning of the normal matrix.

import numpy as np
from ips_learning.fitting.fit_result import SolverDiagnostics


class MatrixDiagnostics:
    """
    Numerical diagnostics for the linear system / normal equations.
    """

    def diagnose(self, 
                 A: np.ndarray, 
                 method: str,
                 ridge_lambda: float | None = None,
                 svd_tol: float | None = None) -> SolverDiagnostics:
        
        U, s, Vt = np.linalg.svd(A, full_matrices=False)

        if s[-1] > 0:
            cond = float(s[0] / s[-1])
        else:
            cond = float("inf")

        # rank = int(np.sum(s > 1e-12))
        rank= int(np.sum(s > 0)) #rank of the assembled matrix \(A\),

        return SolverDiagnostics(
            condition_number=cond,
            singular_values=s,
            rank=rank,
            min_singular=float(s[-1]),
            max_singular=float(s[0]),
            method=method,
            ridge_lambda=ridge_lambda, # Relevant only for the ridge solver.
            svd_tol=svd_tol, # Relevant only for the truncated-SVD solver.
        )