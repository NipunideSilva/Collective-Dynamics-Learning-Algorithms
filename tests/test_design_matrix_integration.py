import numpy as np

from ips_learning.bases.bundle import BasisBundle
from ips_learning.bases.radial import BSplineRadialBasisBuilder
from ips_learning.fitting.design_blocks import DesignBlockBuilder
from ips_learning.fitting.solver import LinearSystemSolver

from ips_learning.bases.environment import BSplineEnvironmentBasisBuilder




def test_radial_design_matrix_recovers_known_constant_kernel():
    """
    Integration test:

        radial basis
        -> design matrix
        -> normal equations
        -> least-squares solver

    We use a constant radial basis psi(r)=1 and choose

        phi(r) = 2.

    For N=2,

        F_i = (1/2) * phi(r_ij) * (x_j - x_i)
            = x_j - x_i.

    Therefore the true basis coefficient is theta = 2.
    """

    # ---------------------------------------------------------
    # Three time samples, two agents, one spatial dimension.
    # Shape: (L, N, D) = (3, 2, 1)
    # ---------------------------------------------------------
    X = np.array([
        [[0.0], [0.5]],
        [[0.2], [0.8]],
        [[-0.1], [0.4]],
    ])

    L, N, D = X.shape

    # ---------------------------------------------------------
    # Pairwise position differences:
    #
    # R_D[l, i, j] = X[l, j] - X[l, i]
    # ---------------------------------------------------------
    R_D = (
        X[:, np.newaxis, :, :]
        - X[:, :, np.newaxis, :]
    )

    R = np.linalg.norm(
        R_D,
        axis=-1,
        keepdims=True,
    )

    # ---------------------------------------------------------
    # Since phi(r)=2 and N=2,
    #
    # F_1 = x_2 - x_1
    # F_2 = x_1 - x_2
    #
    # These values are calculated independently by hand.
    # ---------------------------------------------------------
    LHS = np.array([
        [[0.5], [-0.5]],
        [[0.6], [-0.6]],
        [[0.5], [-0.5]],
    ])

    # ---------------------------------------------------------
    # Degree-0 B-spline with one basis function:
    #
    # psi_1(r) = 1
    #
    # over the radial domain.
    # ---------------------------------------------------------
    radial_basis = BSplineRadialBasisBuilder().build(
        degree=0,
        n_basis=1,
        r_min=0.0,
        r_max=1.0,
    )

    basis_bundle = BasisBundle(
        radial_basis=radial_basis,
    )

    # ---------------------------------------------------------
    # Build interaction-only design matrix.
    # ---------------------------------------------------------
    builder = DesignBlockBuilder(
        interaction_type="energy",
        system_order=1,
        algorithm="nonparametric_f",
        include_environment=False,
    )

    Psi, y = builder.build_for_one_trajectory(
        X=X,
        LHS=LHS,
        R=R,
        R_D=R_D,
        basis_bundle=basis_bundle,
        system_order=1,
    )

    # ---------------------------------------------------------
    # Hand-computed design matrix.
    #
    # Since N=2 and psi(r)=1:
    #
    # Psi_i = (1/2)(x_j - x_i)
    # ---------------------------------------------------------
    expected_Psi = np.array([
        [0.25],
        [-0.25],
        [0.30],
        [-0.30],
        [0.25],
        [-0.25],
    ])

    expected_y = np.array([
        [0.5],
        [-0.5],
        [0.6],
        [-0.6],
        [0.5],
        [-0.5],
    ])

    np.testing.assert_allclose(Psi, expected_Psi)
    np.testing.assert_allclose(y, expected_y)

    # ---------------------------------------------------------
    # Build normal equations:
    #
    # A theta = b
    #
    # A = Psi^T Psi
    # b = Psi^T y
    # ---------------------------------------------------------
    A = Psi.T @ Psi
    b = Psi.T @ y

    solver = LinearSystemSolver(method="lstsq")

    theta_hat, diagnostics = solver.solve(A, b)

    # True phi(r) = 2 * psi_1(r)
    expected_theta = np.array([[2.0]])

    np.testing.assert_allclose(
        theta_hat,
        expected_theta,
        atol=1e-12,
    )

    assert diagnostics.rank == 1

def test_environment_design_matrix_recovers_linear_force():
    """
    First-order system with N=1.

    Since there is only one agent, the interaction term is zero.
    We choose

        f(x) = 3x

    and verify that the environmental basis/design block recovers it.
    """

    # Shape: (L, N, D) = (5, 1, 1)
    X = np.array([
        [[0.00]],
        [[0.25]],
        [[0.50]],
        [[0.75]],
        [[1.00]],
    ])

    # f(x) = 3x
    LHS = 3.0 * X

    L, N, D = X.shape

    # Pairwise quantities. Since N=1, these are all zero.
    R_D = (
        X[:, np.newaxis, :, :]
        - X[:, :, np.newaxis, :]
    )

    R = np.linalg.norm(
        R_D,
        axis=-1,
        keepdims=True,
    )

    # Radial basis is required by DesignBlockBuilder,
    # although its contribution is zero for N=1.
    radial_basis = BSplineRadialBasisBuilder().build(
        degree=0,
        n_basis=1,
        r_min=0.0,
        r_max=1.0,
    )

    # For degree 1 with two basis functions on [0, 1]:
    #
    # B1(x) = 1 - x
    # B2(x) = x
    environment_basis = BSplineEnvironmentBasisBuilder().build(
        degree=1,
        n_basis_per_dim=2,
        domain_min=np.array([0.0]),
        domain_max=np.array([1.0]),
    )

    basis_bundle = BasisBundle(
        radial_basis=radial_basis,
        environment_basis=environment_basis,
    )

    builder = DesignBlockBuilder(
        interaction_type="energy",
        system_order=1,
        algorithm="nonparametric_f",
        include_environment=True,
    )

    Psi, y = builder.build_for_one_trajectory(
        X=X,
        LHS=LHS,
        R=R,
        R_D=R_D,
        basis_bundle=basis_bundle,
    )

    # Column ordering:
    #
    # [radial | environment B1 | environment B2]
    #
    # radial column = 0 because N=1.
    expected_Psi = np.array([
        [0.0, 1.00, 0.00],
        [0.0, 0.75, 0.25],
        [0.0, 0.50, 0.50],
        [0.0, 0.25, 0.75],
        [0.0, 0.00, 1.00],
    ])

    expected_y = np.array([
        [0.00],
        [0.75],
        [1.50],
        [2.25],
        [3.00],
    ])

    np.testing.assert_allclose(Psi, expected_Psi)
    np.testing.assert_allclose(y, expected_y)

    # Normal equations
    A = Psi.T @ Psi
    b = Psi.T @ y

    solver = LinearSystemSolver(method="lstsq")
    theta_hat, _ = solver.solve(A, b)

    # [radial coefficient, B1 coefficient, B2 coefficient]
    expected_theta = np.array([
        [0.0],
        [0.0],
        [3.0],
    ])

    np.testing.assert_allclose(
        theta_hat,
        expected_theta,
        atol=1e-12,
    )
    


def test_combined_radial_and_environment_recovery():
    """
    Integration test for simultaneous recovery of

        phi(r) = 2
        f(x)   = 3x

    in the first-order model

        xdot_i = f(x_i)
                 + (1/N) sum_j phi(r_ij)(x_j - x_i).

    The expected coefficient vector is

        [theta_phi, theta_f1, theta_f2]
        = [2, 0, 3].
    """

    # Three snapshots, N=2, D=1.
    #
    # The pair sums deliberately vary so that the radial column
    # is not a linear combination of the environmental columns.
    X = np.array([
        [[0.1], [0.4]],
        [[0.2], [0.9]],
        [[0.6], [0.8]],
    ])

    # Hand-computed target from
    #
    #   phi(r) = 2
    #   f(x)   = 3x
    #
    # Snapshot 1:
    #   x1=0.1, x2=0.4
    #   interaction = [ 0.3, -0.3]
    #   environment = [ 0.3,  1.2]
    #   total       = [ 0.6,  0.9]
    #
    # Snapshot 2:
    #   total       = [1.3, 2.0]
    #
    # Snapshot 3:
    #   total       = [2.0, 2.2]
    LHS = np.array([
        [[0.6], [0.9]],
        [[1.3], [2.0]],
        [[2.0], [2.2]],
    ])

    # ---------------------------------------------------------
    # Pairwise geometry
    # ---------------------------------------------------------
    R_D = (
        X[:, np.newaxis, :, :]
        - X[:, :, np.newaxis, :]
    )

    R = np.linalg.norm(
        R_D,
        axis=-1,
        keepdims=True,
    )

    # ---------------------------------------------------------
    # Radial basis:
    #
    # degree 0 + one basis function gives
    #
    #   psi_phi(r) = 1
    #
    # on [0, 1].
    # ---------------------------------------------------------
    radial_basis = BSplineRadialBasisBuilder().build(
        degree=0,
        n_basis=1,
        r_min=0.0,
        r_max=1.0,
    )

    # ---------------------------------------------------------
    # Environmental basis:
    #
    # degree 1 + two basis functions on [0,1]:
    #
    #   B1(x) = 1 - x
    #   B2(x) = x
    #
    # Therefore
    #
    #   f(x) = 3x
    #        = 0*B1(x) + 3*B2(x).
    # ---------------------------------------------------------
    environment_basis = BSplineEnvironmentBasisBuilder().build(
        degree=1,
        n_basis_per_dim=2,
        domain_min=np.array([0.0]),
        domain_max=np.array([1.0]),
    )

    basis_bundle = BasisBundle(
        radial_basis=radial_basis,
        environment_basis=environment_basis,
    )

    builder = DesignBlockBuilder(
        interaction_type="energy",
        system_order=1,
        algorithm="nonparametric_f",
        include_environment=True,
    )

    Psi, y = builder.build_for_one_trajectory(
        X=X,
        LHS=LHS,
        R=R,
        R_D=R_D,
        basis_bundle=basis_bundle,
    )

    # ---------------------------------------------------------
    # Expected column order:
    #
    # [radial | environmental B1 | environmental B2]
    #
    # Radial column:
    #
    #   (1/N)(x_j - x_i)
    #
    # because N=2 and psi_phi(r)=1.
    # ---------------------------------------------------------
    expected_Psi = np.array([
        [ 0.15, 0.90, 0.10],
        [-0.15, 0.60, 0.40],

        [ 0.35, 0.80, 0.20],
        [-0.35, 0.10, 0.90],

        [ 0.10, 0.40, 0.60],
        [-0.10, 0.20, 0.80],
    ])

    expected_y = np.array([
        [0.6],
        [0.9],
        [1.3],
        [2.0],
        [2.0],
        [2.2],
    ])

    np.testing.assert_allclose(
        Psi,
        expected_Psi,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        y,
        expected_y,
        atol=1e-12,
    )

    # Make sure the three columns are identifiable.
    assert np.linalg.matrix_rank(Psi) == 3

    # ---------------------------------------------------------
    # Normal equations
    # ---------------------------------------------------------
    A = Psi.T @ Psi
    b = Psi.T @ y

    solver = LinearSystemSolver(method="lstsq")
    theta_hat, diagnostics = solver.solve(A, b)

    # Coefficient order:
    #
    #   theta_phi = 2
    #   theta_B1  = 0
    #   theta_B2  = 3
    expected_theta = np.array([
        [2.0],
        [0.0],
        [3.0],
    ])

    np.testing.assert_allclose(
        theta_hat,
        expected_theta,
        atol=1e-12,
    )

    assert diagnostics.rank == 3


#------------------------second order system with radial and environment basis-----------------------------
def test_second_order_energy_and_environment_recovery():
    """
    Second-order energy-system integration test.

    Recover simultaneously

        phi_E(r) = 2
        f(v)     = 3v

    from

        xddot_i = f(v_i)
                  + (1/N) sum_j phi_E(r_ij)(x_j - x_i).

    Expected coefficients:

        [theta_energy, theta_f1, theta_f2]
        = [2, 0, 3].
    """

    # ---------------------------------------------------------
    # Positions: shape (L, N, D) = (3, 2, 1)
    # ---------------------------------------------------------
    X = np.array([
        [[0.1], [0.4]],
        [[0.2], [0.9]],
        [[0.6], [0.8]],
    ])

    # ---------------------------------------------------------
    # Velocities.
    #
    # For a second-order system, f is evaluated on V.
    # ---------------------------------------------------------
    V = np.array([
        [[0.2], [0.7]],
        [[0.1], [0.9]],
        [[0.4], [0.6]],
    ])

    # ---------------------------------------------------------
    # Hand-computed accelerations.
    #
    # phi_E = 2 and N = 2 imply
    #
    #   energy interaction = x_j - x_i.
    #
    # f(v) = 3v.
    #
    # Snapshot 1:
    #   interaction = [ 0.3, -0.3]
    #   f(v)        = [ 0.6,  2.1]
    #   total       = [ 0.9,  1.8]
    #
    # Snapshot 2:
    #   interaction = [ 0.7, -0.7]
    #   f(v)        = [ 0.3,  2.7]
    #   total       = [ 1.0,  2.0]
    #
    # Snapshot 3:
    #   interaction = [ 0.2, -0.2]
    #   f(v)        = [ 1.2,  1.8]
    #   total       = [ 1.4,  1.6]
    # ---------------------------------------------------------
    LHS = np.array([
        [[0.9], [1.8]],
        [[1.0], [2.0]],
        [[1.4], [1.6]],
    ])

    # Pairwise position differences
    R_D = (
        X[:, np.newaxis, :, :]
        - X[:, :, np.newaxis, :]
    )

    R = np.linalg.norm(
        R_D,
        axis=-1,
        keepdims=True,
    )

    # Constant radial basis psi_E(r) = 1
    radial_basis = BSplineRadialBasisBuilder().build(
        degree=0,
        n_basis=1,
        r_min=0.0,
        r_max=1.0,
    )

    # On [0,1]:
    #
    #   B1(v) = 1-v
    #   B2(v) = v
    #
    # so
    #
    #   3v = 0*B1(v) + 3*B2(v).
    environment_basis = BSplineEnvironmentBasisBuilder().build(
        degree=1,
        n_basis_per_dim=2,
        domain_min=np.array([0.0]),
        domain_max=np.array([1.0]),
    )

    basis_bundle = BasisBundle(
        radial_basis=radial_basis,
        environment_basis=environment_basis,
    )

    builder = DesignBlockBuilder(
        interaction_type="energy",
        system_order=2,
        algorithm="nonparametric_f",
        include_environment=True,
    )

    Psi, y = builder.build_for_one_trajectory(
        X=X,
        V=V,
        LHS=LHS,
        R=R,
        R_D=R_D,
        basis_bundle=basis_bundle,
    )

    # ---------------------------------------------------------
    # Column order:
    #
    # [energy radial | environment B1 | environment B2]
    # ---------------------------------------------------------
    expected_Psi = np.array([
        [ 0.15, 0.80, 0.20],
        [-0.15, 0.30, 0.70],

        [ 0.35, 0.90, 0.10],
        [-0.35, 0.10, 0.90],

        [ 0.10, 0.60, 0.40],
        [-0.10, 0.40, 0.60],
    ])

    expected_y = np.array([
        [0.9],
        [1.8],
        [1.0],
        [2.0],
        [1.4],
        [1.6],
    ])

    np.testing.assert_allclose(
        Psi,
        expected_Psi,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        y,
        expected_y,
        atol=1e-12,
    )

    # All three unknown coefficients must be identifiable.
    assert np.linalg.matrix_rank(Psi) == 3

    A = Psi.T @ Psi
    b = Psi.T @ y

    solver = LinearSystemSolver(method="lstsq")
    theta_hat, diagnostics = solver.solve(A, b)

    expected_theta = np.array([
        [2.0],  # phi_E
        [0.0],  # environmental B1
        [3.0],  # environmental B2
    ])

    np.testing.assert_allclose(
        theta_hat,
        expected_theta,
        atol=1e-12,
    )

    assert diagnostics.rank == 3

def test_second_order_alignment_and_environment_recovery():
    """
    Second-order alignment-system integration test.

    Recover simultaneously

        phi_A(r) = 2
        f(v)     = 3v

    from

        xddot_i = f(v_i)
                  + (1/N) sum_j phi_A(r_ij)(v_j - v_i).

    Expected coefficients:

        [theta_alignment, theta_f1, theta_f2]
        = [2, 0, 3].
    """

    # ---------------------------------------------------------
    # Positions determine the pairwise distances r_ij.
    # Shape: (L, N, D) = (3, 2, 1)
    # ---------------------------------------------------------
    X = np.array([
        [[0.1], [0.4]],
        [[0.2], [0.9]],
        [[0.6], [0.8]],
    ])

    # ---------------------------------------------------------
    # Velocities determine:
    #
    #   1. alignment direction v_j - v_i
    #   2. environmental input f(v_i)
    # ---------------------------------------------------------
    V = np.array([
        [[0.2], [0.7]],
        [[0.1], [0.9]],
        [[0.4], [0.6]],
    ])

    # ---------------------------------------------------------
    # phi_A(r) = 2 and N = 2:
    #
    #   alignment_i = v_j - v_i
    #
    # f(v_i) = 3v_i.
    #
    # Snapshot 1:
    #
    #   V = [0.2, 0.7]
    #
    #   alignment = [ 0.5, -0.5]
    #   environment = [0.6, 2.1]
    #   total = [1.1, 1.6]
    #
    # Snapshot 2:
    #
    #   alignment = [ 0.8, -0.8]
    #   environment = [0.3, 2.7]
    #   total = [1.1, 1.9]
    #
    # Snapshot 3:
    #
    #   alignment = [ 0.2, -0.2]
    #   environment = [1.2, 1.8]
    #   total = [1.4, 1.6]
    # ---------------------------------------------------------
    LHS = np.array([
        [[1.1], [1.6]],
        [[1.1], [1.9]],
        [[1.4], [1.6]],
    ])

    # ---------------------------------------------------------
    # Position pairwise differences
    #
    # R_D[l,i,j] = X[l,j] - X[l,i]
    # ---------------------------------------------------------
    R_D = (
        X[:, np.newaxis, :, :]
        - X[:, :, np.newaxis, :]
    )

    R = np.linalg.norm(
        R_D,
        axis=-1,
        keepdims=True,
    )

    # ---------------------------------------------------------
    # Velocity pairwise differences
    #
    # VR_D[l,i,j] = V[l,j] - V[l,i]
    # ---------------------------------------------------------
    VR_D = (
        V[:, np.newaxis, :, :]
        - V[:, :, np.newaxis, :]
    )

    # ---------------------------------------------------------
    # Constant alignment kernel basis:
    #
    # psi_A(r) = 1
    #
    # so theta_A = 2 represents phi_A(r)=2.
    # ---------------------------------------------------------
    radial_basis = BSplineRadialBasisBuilder().build(
        degree=0,
        n_basis=1,
        r_min=0.0,
        r_max=1.0,
    )

    # ---------------------------------------------------------
    # Environmental basis:
    #
    # B1(v) = 1-v
    # B2(v) = v
    #
    # Therefore
    #
    # f(v) = 3v = 0*B1(v) + 3*B2(v).
    # ---------------------------------------------------------
    environment_basis = BSplineEnvironmentBasisBuilder().build(
        degree=1,
        n_basis_per_dim=2,
        domain_min=np.array([0.0]),
        domain_max=np.array([1.0]),
    )

    basis_bundle = BasisBundle(
        radial_basis=radial_basis,
        environment_basis=environment_basis,
    )

    builder = DesignBlockBuilder(
        interaction_type="alignment",
        system_order=2,
        algorithm="nonparametric_f",
        include_environment=True,
    )

    Psi, y = builder.build_for_one_trajectory(
        X=X,
        V=V,
        LHS=LHS,
        R=R,
        R_D=R_D,
        VR_D=VR_D,
        basis_bundle=basis_bundle,
    )

    # ---------------------------------------------------------
    # Column order:
    #
    # [alignment radial | environment B1 | environment B2]
    #
    # Alignment column:
    #
    #   (1/N)(v_j - v_i)
    #
    # since psi_A(r)=1.
    # ---------------------------------------------------------
    expected_Psi = np.array([
        [ 0.25, 0.80, 0.20],
        [-0.25, 0.30, 0.70],

        [ 0.40, 0.90, 0.10],
        [-0.40, 0.10, 0.90],

        [ 0.10, 0.60, 0.40],
        [-0.10, 0.40, 0.60],
    ])

    expected_y = np.array([
        [1.1],
        [1.6],
        [1.1],
        [1.9],
        [1.4],
        [1.6],
    ])

    np.testing.assert_allclose(
        Psi,
        expected_Psi,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        y,
        expected_y,
        atol=1e-12,
    )

    # The three unknown coefficients must be identifiable.
    assert np.linalg.matrix_rank(Psi) == 3

    # ---------------------------------------------------------
    # Solve the normal equations
    # ---------------------------------------------------------
    A = Psi.T @ Psi
    b = Psi.T @ y

    solver = LinearSystemSolver(method="lstsq")
    theta_hat, diagnostics = solver.solve(A, b)

    expected_theta = np.array([
        [2.0],  # phi_A
        [0.0],  # environmental B1
        [3.0],  # environmental B2
    ])

    np.testing.assert_allclose(
        theta_hat,
        expected_theta,
        atol=1e-12,
    )

    assert diagnostics.rank == 3

def test_second_order_energy_alignment_environment_recovery():
    """
    Recover simultaneously

        phi_E(r) = 2
        phi_A(r) = -1
        f(v)     = 3v

    from the full second-order model

        xddot_i =
            f(v_i)
            + (1/N) sum_j phi_E(r_ij)(x_j - x_i)
            + (1/N) sum_j phi_A(r_ij)(v_j - v_i).

    Expected coefficient ordering:

        [theta_E, theta_A, theta_f1, theta_f2]
        = [2, -1, 0, 3].
    """

    X = np.array([
        [[0.1], [0.4]],
        [[0.2], [0.9]],
        [[0.6], [0.8]],
    ])

    V = np.array([
        [[0.2], [0.7]],
        [[0.1], [0.9]],
        [[0.4], [0.6]],
    ])

    # ---------------------------------------------------------
    # Hand-computed accelerations.
    #
    # Energy:
    #   phi_E = 2, N=2
    #   => contribution = x_j - x_i
    #
    # Alignment:
    #   phi_A = -1, N=2
    #   => contribution = -(1/2)(v_j - v_i)
    #
    # Environment:
    #   f(v) = 3v
    #
    # Snapshot 1:
    #   energy      = [ 0.3, -0.3]
    #   alignment   = [-0.25, 0.25]
    #   environment = [ 0.6,  2.1]
    #   total       = [ 0.65, 2.05]
    #
    # Snapshot 2:
    #   total       = [0.60, 2.40]
    #
    # Snapshot 3:
    #   total       = [1.30, 1.70]
    # ---------------------------------------------------------
    LHS = np.array([
        [[0.65], [2.05]],
        [[0.60], [2.40]],
        [[1.30], [1.70]],
    ])

    # Position differences
    R_D = (
        X[:, np.newaxis, :, :]
        - X[:, :, np.newaxis, :]
    )

    R = np.linalg.norm(
        R_D,
        axis=-1,
        keepdims=True,
    )

    # Velocity differences
    VR_D = (
        V[:, np.newaxis, :, :]
        - V[:, :, np.newaxis, :]
    )

    # Constant basis for both interaction kernels
    energy_basis = BSplineRadialBasisBuilder().build(
        degree=0,
        n_basis=1,
        r_min=0.0,
        r_max=1.0,
    )

    alignment_basis = BSplineRadialBasisBuilder().build(
        degree=0,
        n_basis=1,
        r_min=0.0,
        r_max=1.0,
    )

    # B1(v)=1-v, B2(v)=v
    environment_basis = BSplineEnvironmentBasisBuilder().build(
        degree=1,
        n_basis_per_dim=2,
        domain_min=np.array([0.0]),
        domain_max=np.array([1.0]),
    )

    basis_bundle = BasisBundle(
        energy_basis=energy_basis,
        alignment_basis=alignment_basis,
        environment_basis=environment_basis,
    )

    builder = DesignBlockBuilder(
        interaction_type="energy_alignment",
        system_order=2,
        algorithm="nonparametric_f",
        include_environment=True,
    )

    Psi, y = builder.build_for_one_trajectory(
        X=X,
        V=V,
        LHS=LHS,
        R=R,
        R_D=R_D,
        VR_D=VR_D,
        basis_bundle=basis_bundle,
    )

    # ---------------------------------------------------------
    # Expected column order:
    #
    # [energy | alignment | env B1 | env B2]
    # ---------------------------------------------------------
    expected_Psi = np.array([
        [ 0.15,  0.25, 0.80, 0.20],
        [-0.15, -0.25, 0.30, 0.70],

        [ 0.35,  0.40, 0.90, 0.10],
        [-0.35, -0.40, 0.10, 0.90],

        [ 0.10,  0.10, 0.60, 0.40],
        [-0.10, -0.10, 0.40, 0.60],
    ])

    expected_y = np.array([
        [0.65],
        [2.05],
        [0.60],
        [2.40],
        [1.30],
        [1.70],
    ])

    np.testing.assert_allclose(
        Psi,
        expected_Psi,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        y,
        expected_y,
        atol=1e-12,
    )

    # All four coefficients should be identifiable.
    assert np.linalg.matrix_rank(Psi) == 4

    A = Psi.T @ Psi
    b = Psi.T @ y

    solver = LinearSystemSolver(method="lstsq")
    theta_hat, diagnostics = solver.solve(A, b)

    expected_theta = np.array([
        [ 2.0],  # phi_E
        [-1.0],  # phi_A
        [ 0.0],  # environment B1
        [ 3.0],  # environment B2
    ])

    np.testing.assert_allclose(
        theta_hat,
        expected_theta,
        atol=1e-11,
    )

    assert diagnostics.rank == 4

def test_second_order_parametric_f_recovery():
    """
    Recover simultaneously

        phi_E(r) = 2

    and the SPP parametric self-propulsion law

        f(v) = alpha v - beta ||v||^2 v

    with

        alpha = 1
        beta  = 0.5.

    The parametric scalar terms are

        [v1, v2, ||v||^2 v1, ||v||^2 v2]

    and are fitted componentwise.
    """
    from ips_learning.systems.spp import parametric_f_basis

    X = np.array([
        [[0.0, 0.0], [0.4, 0.1]],
        [[0.1, 0.2], [0.8, 0.5]],
        [[0.2, 0.7], [0.7, 0.1]],
        [[0.6, 0.2], [0.9, 0.8]],
        [[0.1, 0.9], [0.5, 0.4]],
    ])

    V = np.array([
        [[0.2, 0.1], [0.7, 0.3]],
        [[0.1, 0.8], [0.9, 0.2]],
        [[0.4, 0.6], [0.3, 0.9]],
        [[0.8, 0.5], [0.2, 0.4]],
        [[0.6, 0.7], [0.5, 0.1]],
    ])

    # ---------------------------------------------------------
    # True environmental force:
    #
    #   f(v) = v - 0.5 ||v||^2 v
    # ---------------------------------------------------------
    speed_squared = np.sum(
        V**2,
        axis=-1,
        keepdims=True,
    )

    F_env = (
        V
        - 0.5 * speed_squared * V
    )

    # ---------------------------------------------------------
    # True energy interaction:
    #
    # N = 2 and phi_E = 2, therefore
    #
    #   (1/N) phi_E (x_j - x_i)
    #       = x_j - x_i.
    # ---------------------------------------------------------
    F_interaction = np.zeros_like(X)

    F_interaction[:, 0, :] = (
        X[:, 1, :] - X[:, 0, :]
    )

    F_interaction[:, 1, :] = (
        X[:, 0, :] - X[:, 1, :]
    )

    LHS = F_env + F_interaction

    # Pairwise position geometry
    R_D = (
        X[:, np.newaxis, :, :]
        - X[:, :, np.newaxis, :]
    )

    R = np.linalg.norm(
        R_D,
        axis=-1,
        keepdims=True,
    )

    # Constant interaction basis psi(r)=1
    radial_basis = BSplineRadialBasisBuilder().build(
        degree=0,
        n_basis=1,
        r_min=0.0,
        r_max=2.0,
    )

    basis_bundle = BasisBundle(
        radial_basis=radial_basis,
        parametric_f_basis=parametric_f_basis(),
    )

    builder = DesignBlockBuilder(
        interaction_type="energy",
        system_order=2,
        algorithm="parametric_f",
        include_environment=True,
    )

    Psi, y = builder.build_for_one_trajectory(
        X=X,
        V=V,
        LHS=LHS,
        R=R,
        R_D=R_D,
        basis_bundle=basis_bundle,
    )

    # ---------------------------------------------------------
    # 1 interaction coefficient
    #
    # + 4 parametric terms for output component 1
    #
    # + 4 parametric terms for output component 2
    #
    # = 9 total coefficients.
    # ---------------------------------------------------------
    assert Psi.shape == (20, 9)
    assert np.linalg.matrix_rank(Psi) == 9

    A = Psi.T @ Psi
    b = Psi.T @ y

    solver = LinearSystemSolver(method="lstsq")
    theta_hat, diagnostics = solver.solve(A, b)

    # ---------------------------------------------------------
    # Coefficient layout:
    #
    # interaction:
    #   phi_E = 2
    #
    # f_1(v):
    #   +1 v1
    #   +0 v2
    #   -0.5 ||v||^2 v1
    #   +0 ||v||^2 v2
    #
    # f_2(v):
    #   +0 v1
    #   +1 v2
    #   +0 ||v||^2 v1
    #   -0.5 ||v||^2 v2
    # ---------------------------------------------------------
    expected_theta = np.array([
        [ 2.0],

        [ 1.0],
        [ 0.0],
        [-0.5],
        [ 0.0],

        [ 0.0],
        [ 1.0],
        [ 0.0],
        [-0.5],
    ])

    np.testing.assert_allclose(
        theta_hat,
        expected_theta,
        atol=1e-10,
    )

    assert diagnostics.rank == 9






