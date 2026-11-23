import numpy as np
from scipy import sparse

VALID_INTERACTION_TYPES = {"energy", "alignment", "energy_alignment"}

class DesignBlockBuilder:
    """
    Build trajectory-wise design blocks for first- and second-order systems.

    First-order:
        dot{x}_i = f_env(x_i)
                   + (1/N) sum_j phi(r_ij)(x_j - x_i)

        environmental input Z = X
        target LHS = dot X

    Second-order, energy:
        ddot{x}_i = f_env(v_i)
                    + (1/N) sum_j phi(r_ij)(x_j - x_i)

        environmental input Z = V
        target LHS = ddot X

    Second-order, alignment:
        ddot{x}_i = f_env(v_i)
                    + (1/N) sum_j phi(r_ij)(v_j - v_i)

        environmental input Z = V
        target LHS = ddot X


    Second-order, energy + alignment:
    ddot{x}_i = f_env(v_i)
                + (1/N) sum_j phi_E(r_ij)(x_j - x_i)
                + (1/N) sum_j phi_A(r_ij)(v_j - v_i)
    """

    def __init__(self, 
                 interaction_type: str = "energy",
                  system_order: int | None = None,
                algorithm: str = "nonparametric_f",
                include_environment: bool = True,
                ):
        
        if interaction_type not in VALID_INTERACTION_TYPES:
            raise ValueError(
                f"interaction_type must be one of {sorted(VALID_INTERACTION_TYPES)}"
                f"got {interaction_type!r}."
            )
        
        if system_order is not None and system_order not in (1, 2):
            raise ValueError(
                f"system_order must be 1, 2, or None, got {system_order}."
            )
        
        if system_order == 1 and interaction_type in {"alignment", "energy_alignment"}:
            raise ValueError(
                "Alignment and energy_alignment interactions are only supported "
                "for second-order systems."
            )
        
        if algorithm not in {"nonparametric_f", "parametric_f"}:
            raise ValueError(
                "algorithm must be 'nonparametric_f' or 'parametric_f', "
                f"got {algorithm!r}."
            )

        self.interaction_type = interaction_type
        self.system_order = system_order
        self.algorithm = algorithm
        self.include_environment = bool(include_environment)

    def build_for_one_trajectory(
        self,
        X,
        LHS,
        R,
        R_D,
        basis_bundle,
        V=None,
        VR_D=None,
        system_order: int | None = None,
    ):
        """
        Parameters
        ----------
        X : np.ndarray
            Shape (L, N, D).
        V : np.ndarray or None
            Shape (L, N, D) for second-order systems.
            None for first-order systems.
        LHS : np.ndarray
            Shape (L, N, D).
            First-order:  dot X.
            Second-order: ddot X.
        R : np.ndarray
            Shape (L, N, N, q), usually q=1.
        R_D : np.ndarray
            Shape (L, N, N, D), where R_D[l,i,j] = X[l,j] - X[l,i].
        VR_D : np.ndarray or None
            Shape (L, N, N, D), where VR_D[l,i,j] = V[l,j] - V[l,i].
            Required when interaction_type == "alignment".
        basis_bundle : BasisBundle.

        system_order : int or None
            1 for first-order, 2 for second-order.
            If None, uses self.system_order.

        Returns
        -------
        Psi : np.ndarray
            Dense design matrix of shape (D * L * N, n_total).
        y : np.ndarray
            Target column of shape (D * L * N, 1).
        """
        system_order = self._resolve_system_order(system_order)

        #=================================================================
        if self.interaction_type == "energy":
            psi_radial = self._build_radial_block(
                R=R,
                interaction_direction=R_D,
                radial_basis=basis_bundle.radial_basis,
            )

        elif self.interaction_type == "alignment":
            if system_order != 2:
                raise ValueError("Alignment interaction requires second-order data.")
            if VR_D is None:
                raise ValueError("VR_D is required for alignment interaction.")

            psi_radial = self._build_radial_block(
                R=R,
                interaction_direction=VR_D,
                radial_basis=basis_bundle.radial_basis,
            )

        elif self.interaction_type == "energy_alignment":
            if system_order != 2:
                raise ValueError("energy_alignment requires second-order data.")
            if VR_D is None:
                raise ValueError("VR_D is required for energy_alignment interaction.")

            energy_basis = getattr(basis_bundle, "energy_basis", None)
            alignment_basis = getattr(basis_bundle, "alignment_basis", None)

            if energy_basis is None:
                energy_basis = basis_bundle.radial_basis

            if alignment_basis is None:
                alignment_basis = basis_bundle.radial_basis

            psi_energy = self._build_radial_block(
                R=R,
                interaction_direction=R_D,
                radial_basis=energy_basis,
            )

            psi_alignment = self._build_radial_block(
                R=R,
                interaction_direction=VR_D,
                radial_basis=alignment_basis,
            )

            psi_radial = np.hstack([psi_energy, psi_alignment])

        else:
            raise RuntimeError(f"Unsupported interaction_type={self.interaction_type!r}")


        #================================================================
        if self.include_environment:
            Z = self._select_environment_input(
                X=X,
                V=V,
                system_order=system_order,
            )


            if self.algorithm == "nonparametric_f":
                psi_env = self._build_environment_block(
                    Z=Z,
                    environment_basis=basis_bundle.environment_basis,
                )

            elif self.algorithm == "parametric_f":
                if basis_bundle.parametric_f_basis is None:
                    raise ValueError(
                        "algorithm='parametric_f' requires "
                        "basis_bundle.parametric_f_basis."
                    )

                psi_env = self._build_parametric_environment_block(
                    Z=Z,
                    parametric_f_basis=basis_bundle.parametric_f_basis,
                )

            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm!r}")
            


            if sparse.isspmatrix(psi_env):
                psi_env = psi_env.toarray()

            Psi = np.hstack([psi_radial, psi_env])
        else:
            Psi = psi_radial

        y = LHS.transpose(2, 0, 1).reshape(-1, 1)
        return Psi, y
    
    def _resolve_system_order(self, system_order: int | None) -> int:
        if system_order is None:
            system_order = self.system_order

        if system_order not in (1, 2):
            raise ValueError(
                "system_order must be passed either to the constructor "
                "or to build_for_one_trajectory."
            )

        return system_order

    
    def _select_environment_input(self, X, V, system_order: int):
        if system_order == 1:
            return X

        if system_order == 2:
            if V is None:
                raise ValueError("Second-order design block requires V.")
            return V

        raise RuntimeError(f"Unsupported system_order={system_order}")
    



    def _build_radial_block(self, R, interaction_direction, radial_basis):
        """
        Build interaction-design block.

        Input shapes
        ------------
        R : (L, N, N, 1)

        interaction_direction:
            energy    -> R_D  : (L, N, N, D)
            alignment -> VR_D : (L, N, N, D)

        Output
        ------
        np.ndarray of shape (D * L * N, n_r)
        """
        L, N, _, _ = R.shape
        D = interaction_direction.shape[-1]

        k = radial_basis.evaluate(R)  # (L, N, N, n_r)

        idx = np.arange(N)
        k[:, idx, idx, :] = 0.0

        mult = interaction_direction[..., None] * k[..., None, :]
        psi = (1.0 / N) * np.sum(mult, axis=2)  # (L, N, D, n_r)

        return psi.transpose(2, 0, 1, 3).reshape(D * L * N, -1)

    def _build_environment_block(self, Z, environment_basis):
        """
        Build environment/design block.

        Input
        -----
        Z : (L, N, D)

        Output
        ------
        sparse array of shape:
            (D * L * N, D * prod(n_basis_per_dim))
        """
        if Z is None:
            raise ValueError("Environment input Z cannot be None.")

        if Z.ndim != 3:
            raise ValueError(f"Expected Z shape (L, N, D), got {Z.shape}.")

        L, N, D = Z.shape
        rows = L * N

        Z_flat = Z.reshape(rows, D)
        per_dim = environment_basis.evaluate_per_dim(Z_flat)

        tp = self._rowwise_tensor_product(per_dim)

        tp_sparse = sparse.csr_matrix(tp)

        return sparse.kron(
            sparse.identity(D, format="csr"),
            tp_sparse,
            format="csr",
        )

    @staticmethod
    def _rowwise_tensor_product(basis_list):
        """
        basis_list: list of arrays, each with shape (rows, n_d).
        """
        rows = basis_list[0].shape[0]
        out = basis_list[0]

        for arr in basis_list[1:]:
            temp = (
                out.reshape(rows, out.shape[1], 1)
                * arr.reshape(rows, 1, arr.shape[1])
            )
            out = temp.reshape(rows, -1)

        return out
    


    #-------parametric---------------------------------------
    def _evaluate_parametric_f_basis(
        self,
        Z: np.ndarray,
        parametric_f_basis,
    ) -> np.ndarray:
        """
        Evaluate scalar parametric terms b_k(z).

        Parameters
        ----------
        Z:
            Shape (L, N, D).

        Returns
        -------
        B:
            Shape (L*N, K), with B[:, k] = b_k(z).
        """
        Z = np.asarray(Z, dtype=float)

        if Z.ndim != 3:
            raise ValueError(f"Expected Z shape (L, N, D), got {Z.shape}.")

        L, N, D = Z.shape
        Z_flat = Z.reshape(L * N, D)

        cols = []
        for term in parametric_f_basis.terms:
            values = np.asarray(term.function(Z_flat), dtype=float).reshape(-1)
            if values.shape[0] != Z_flat.shape[0]:
                raise ValueError(
                    f"Parametric f term {term.name!r} returned shape "
                    f"{values.shape}, expected length {Z_flat.shape[0]}."
                )
            cols.append(values)

        return np.column_stack(cols)
    
    def _build_parametric_environment_block(
        self,
        Z: np.ndarray,
        parametric_f_basis,
    ) -> np.ndarray:
        """
        Build componentwise vector-field block from scalar parametric terms.

        If B has shape (Q, K), output has shape (D*Q, D*K).
        """
        Z = np.asarray(Z, dtype=float)
        L, N, D = Z.shape
        Q = L * N

        B = self._evaluate_parametric_f_basis(
            Z=Z,
            parametric_f_basis=parametric_f_basis,
        )

        K = B.shape[1]
        block = np.zeros((D * Q, D * K), dtype=float)

        for d in range(D):
            row_start = d * Q
            row_end = (d + 1) * Q
            col_start = d * K
            col_end = (d + 1) * K
            block[row_start:row_end, col_start:col_end] = B

        return block
