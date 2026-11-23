from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class InitialConditionBatch:
    """
    Container for a batch of initial conditions.

    Attributes
    ----------
    X0 : np.ndarray of shape (M, N, D)
        Initial positions.
    V0 : np.ndarray of shape (M, N, D)
        Initial velocities.
    U0 : np.ndarray of shape (M, 2 * N * D)
        Flattened simulation-ready states:
            U0[m] = [X0[m].flatten(), V0[m].flatten()]
    """
    X0: np.ndarray
    V0: np.ndarray
    U0: np.ndarray

class InitialConditionGenerator:
    """
    Generate batches of initial conditions for interacting particle systems.

    Parameters
    ----------
    N : int
        Number of agents.
    D : int, default=2
        Spatial dimension.
    """

    def __init__(self, N: int, D: int = 2) -> None:
        if D != 2:
            raise NotImplementedError(
                "This initial condition generator currently supports D=2 only."
            )

        self.N = int(N)
        self.D = int(D)

    @staticmethod
    def _normalize_rows(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """
        Normalize vectors row-wise along the last axis.

        Parameters
        ----------
        v : np.ndarray
            Input array whose last axis represents vector components.
        eps : float, default=1e-12
            Stability constant to avoid division by zero.

        Returns
        -------
        np.ndarray
            Row-normalized array with same shape as v.
        """
        norms = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / (norms + eps)

    def pack(self, X0: np.ndarray, V0: np.ndarray) -> InitialConditionBatch:
        """
        Convert structured initial conditions into a batch object.

        Parameters
        ----------
        X0 : np.ndarray of shape (M, N, D)
            Initial positions.
        V0 : np.ndarray of shape (M, N, D)
            Initial velocities.

        Returns
        -------
        InitialConditionBatch
            Structured and flattened initial conditions.
        """
        X0 = np.asarray(X0, dtype=float)
        V0 = np.asarray(V0, dtype=float)

        if X0.ndim != 3 or V0.ndim != 3:
            raise ValueError("X0 and V0 must both have shape (M, N, D).")
        if X0.shape != V0.shape:
            raise ValueError(f"X0 shape {X0.shape} must match V0 shape {V0.shape}.")
        if X0.shape[1:] != (self.N, self.D):
            raise ValueError(
                f"Expected X0 and V0 shape (M, {self.N}, {self.D}), got {X0.shape}."
            )

        M = X0.shape[0]
        U0 = np.concatenate(
            [X0.reshape(M, -1), V0.reshape(M, -1)],
            axis=1,
        )

        return InitialConditionBatch(X0=X0, V0=V0, U0=U0)

    # -------------------------------
    # Case 1: Clumps
    # -------------------------------

    def clumps(
        self,
        M: int,
        v_eq: float = np.sqrt(1.0 / 0.5),
        L: float = 2.0,
        seed: int | None = None,
    ) -> InitialConditionBatch:
        """
        Sample 'clumps' initial conditions.

        Positions are sampled uniformly in a square and velocities are random
        directions with magnitudes near an equilibrium speed.

        Parameters
        ----------
        M : int
            Number of initial conditions to generate.
        v_eq : float
            Reference velocity magnitude.
        L : float
            Side length of the square domain for positions.
        seed : int or None
            Random seed.

        Returns
        -------
        InitialConditionBatch
        """
        rng = np.random.default_rng(seed)

        X0 = rng.uniform(0.0, L, size=(M, self.N, self.D))
        dirs = rng.normal(size=(M, self.N, self.D))
        dirs = self._normalize_rows(dirs)
        speeds = rng.uniform(0.9 * v_eq, 1.1 * v_eq, size=(M, self.N, 1))
        V0 = dirs * speeds

        return self.pack(X0, V0)

    # -------------------------------
    # Case 2: Ring Clumps
    # -------------------------------
    def ring_clumps(
        self,
        M: int,
        v_eq: float = np.sqrt(1.0 / 0.5),
        R: float = 10.0,
        seed: int | None = None,
    ) -> InitialConditionBatch:
        """
        Sample 'ring clumps' initial conditions.

        Positions lie approximately on a circle and velocities are approximately
        tangential.

        Parameters
        ----------
        M : int
            Number of initial conditions.
        v_eq : float
            Reference velocity magnitude.
        R : float
            Circle radius.
        seed : int or None
            Random seed.

        Returns
        -------
        InitialConditionBatch
        """
        rng = np.random.default_rng(seed)

        angles = rng.uniform(0.0, 2.0 * np.pi, size=(M, self.N))
        X0 = np.stack([R * np.cos(angles), R * np.sin(angles)], axis=-1)

        tangents = np.stack([-np.sin(angles), np.cos(angles)], axis=-1)
        tangents = self._normalize_rows(tangents)
        speeds = rng.uniform(0.9 * v_eq, 1.1 * v_eq, size=(M, self.N, 1))
        V0 = tangents * speeds

        return self.pack(X0, V0)


    # -------------------------------
    # Case 3: Rings
    # -------------------------------
    def rings(
        self,
        M: int,
        R: float = 10.0,
        v_eq: float = np.sqrt(1.0 / 0.5),
        angle_noise_deg: float = 7.0,
        center: tuple[float, float] = (0.0, 0.0),
        seed: int | None = None,
    ) -> InitialConditionBatch:
        """
        Sample 'rings' initial conditions.

        Positions are sampled uniformly in a disk and velocities are roughly
        tangential with angular perturbation.

        Parameters
        ----------
        M : int
            Number of initial conditions.
        R : float
            Disk radius.
        v_eq : float
            Reference velocity magnitude.
        angle_noise_deg : float
            Angular noise around the tangential direction.
        center : tuple of floats
            Center of the disk.
        seed : int or None
            Random seed.

        Returns
        -------
        InitialConditionBatch
        """
        rng = np.random.default_rng(seed)

        theta = rng.uniform(0.0, 2.0 * np.pi, size=(M, self.N))
        r = R * np.sqrt(rng.uniform(0.0, 1.0, size=(M, self.N)))

        X0 = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=-1)
        X0 += np.asarray(center, dtype=float).reshape(1, 1, 2)

        angle_noise = np.deg2rad(
            rng.normal(loc=0.0, scale=angle_noise_deg, size=(M, self.N))
        )
        heading = theta + np.pi / 2.0 + angle_noise

        V0 = np.stack(
            [v_eq * np.cos(heading), v_eq * np.sin(heading)],
            axis=-1,
        )

        return self.pack(X0, V0)


    # -------------------------------
    # Case 4: Swarm
    # -------------------------------
    def swarm(
        self,
        M: int,
        v_eq: float = np.sqrt(1.6 / 0.5),
        L: float = 5.0,
        seed: int | None = None,
    ) -> InitialConditionBatch:
        """
        Sample 'swarm' initial conditions.

        Positions are sampled uniformly in a square and velocities are random
        directions with magnitudes near an equilibrium speed.

        Parameters
        ----------
        M : int
            Number of initial conditions.
        v_eq : float
            Reference velocity magnitude.
        L : float
            Side length of the square domain.
        seed : int or None
            Random seed.

        Returns
        -------
        InitialConditionBatch
        """
        rng = np.random.default_rng(seed)

        X0 = rng.uniform(0.0, L, size=(M, self.N, self.D))
        dirs = rng.normal(size=(M, self.N, self.D))
        dirs = self._normalize_rows(dirs)
        speeds = rng.uniform(0.9 * v_eq, 1.1 * v_eq, size=(M, self.N, 1))
        V0 = dirs * speeds

        return self.pack(X0, V0)
    

    # -------------------------------
    # Case 5: Clumps wide
    # -------------------------------
    def clumps_wide(
        self,
        M: int,
        v_eq: float = np.sqrt(1.0 / 0.5),
        L_values=(2.0, 4.0, 6.0, 8.0),
        speed_range=(0.2, 2.0),
        center_range=(-4.0, 4.0),
        seed: int | None = None,
    ) -> InitialConditionBatch:
        """
        Generate clump-type ICs with wider support in position and velocity.

        This improves coverage of the learned f(v) domain.
        """
        rng = np.random.default_rng(seed)

        X0 = np.empty((M, self.N, self.D))
        V0 = np.empty((M, self.N, self.D))

        for m in range(M):
            # Random spatial scale
            L = rng.choice(L_values)

            # Random center so clumps occupy different regions
            center = rng.uniform(center_range[0], center_range[1], size=(1, self.D))

            # Positions in a shifted square
            X0[m] = center + rng.uniform(-0.5 * L, 0.5 * L, size=(self.N, self.D))

            # Random velocity directions
            dirs = rng.normal(size=(self.N, self.D))
            dirs = self._normalize_rows(dirs)

            # Wider speed range, not only near v_eq
            speeds = rng.uniform(
                speed_range[0] * v_eq,
                speed_range[1] * v_eq,
                size=(self.N, 1),
            )

            V0[m] = dirs * speeds

        return self.pack(X0, V0)
    # -------------------------------
    # Case 6: Levi uniform
    # -------------------------------
    def levi_uniform(
        self,
        M: int,
        x_low: float = 0.0,
        x_high: float = 2.0,
        v_low: float = 0.0,
        v_high: float = 1.0,
        seed: int | None = None,
    ) -> InitialConditionBatch:
        """
        Sample Levi initial conditions.

        This matches the legacy Levi setup:

            X0 ~ Uniform([x_low, x_high])
            V0 ~ Uniform([v_low, v_high])

        Parameters
        ----------
        M : int
            Number of initial conditions.
        x_low, x_high : float
            Position sampling bounds.
        v_low, v_high : float
            Velocity sampling bounds.
        seed : int or None
            Random seed.

        Returns
        -------
        InitialConditionBatch
            X0, V0, and flattened U0.
        """
        rng = np.random.default_rng(seed)

        X0 = rng.uniform(
            low=x_low,
            high=x_high,
            size=(M, self.N, self.D),
        )

        V0 = rng.uniform(
            low=v_low,
            high=v_high,
            size=(M, self.N, self.D),
        )

        return self.pack(X0, V0)
    
    # -------------------------------
    # Case 7: SPP uniform
    # -------------------------------
    
    def spp_uniform_box(
        self,
        M: int,
        x_low: float = -4.0,
        x_high: float = 4.0,
        v_scale: float = 2.5,
        v_eq: float = np.sqrt(1.0 / 0.5),
        seed: int | None = None,
    ) -> InitialConditionBatch:
        """
        SPP ICs with rectangular velocity support.

        Positions:
            X0 ~ Uniform([x_low, x_high]^D)

        Velocities:
            V0 ~ Uniform([-v_scale*v_eq, v_scale*v_eq]^D)

        This gives better 2D f(v) tensor-basis coverage than speed-direction sampling.
        """
        rng = np.random.default_rng(seed)

        X0 = rng.uniform(
            low=x_low,
            high=x_high,
            size=(M, self.N, self.D),
        )

        v_bound = v_scale * v_eq

        V0 = rng.uniform(
            low=-v_bound,
            high=v_bound,
            size=(M, self.N, self.D),
        )

        return self.pack(X0, V0)
    
    # -------------------------------
    # Case 8: SPP grid clumps (cover more area)
    # -------------------------------
    def spp_velocity_grid_clumps(
        self,
        M: int,
        v_eq: float = np.sqrt(1.0 / 0.5),
        L_values=(2.0, 4.0, 6.0, 8.0),
        center_range=(-4.0, 4.0),
        v_box_scale: float = 2.0,
        jitter: float = 0.05,
        seed: int | None = None,
    ) -> InitialConditionBatch:
        """
        Generate SPP ICs with broad rectangular velocity-space coverage.

        Purpose
        -------
        This is designed for learning a 2D tensor-product f(v) basis.
        Unlike clumps_wide, which samples random velocity directions and speeds,
        this generator deliberately covers the (v1, v2) rectangle.

        Positions:
            shifted clumps with several spatial scales.

        Velocities:
            approximately grid-based samples in
                [-v_box_scale*v_eq, v_box_scale*v_eq]^2
            with small jitter.

        This improves coverage of tensor-product velocity basis cells.
        """
        rng = np.random.default_rng(seed)

        if self.D != 2:
            raise ValueError("spp_velocity_grid_clumps currently assumes D=2.")

        X0 = np.empty((M, self.N, self.D))
        V0 = np.empty((M, self.N, self.D))

        # Build a reusable velocity grid.
        n_side = int(np.ceil(np.sqrt(M * self.N)))
        v_min = -v_box_scale * v_eq
        v_max = v_box_scale * v_eq

        v1 = np.linspace(v_min, v_max, n_side)
        v2 = np.linspace(v_min, v_max, n_side)

        V1, V2 = np.meshgrid(v1, v2, indexing="ij")
        velocity_pool = np.column_stack([V1.ravel(), V2.ravel()])

        rng.shuffle(velocity_pool)

        needed = M * self.N
        if velocity_pool.shape[0] < needed:
            repeats = int(np.ceil(needed / velocity_pool.shape[0]))
            velocity_pool = np.tile(velocity_pool, (repeats, 1))

        velocity_pool = velocity_pool[:needed]

        # Add small jitter so samples do not lie exactly on grid lines.
        jitter_scale = jitter * (v_max - v_min)
        velocity_pool = velocity_pool + rng.normal(
            loc=0.0,
            scale=jitter_scale,
            size=velocity_pool.shape,
        )

        # Optional clipping keeps support controlled.
        velocity_pool = np.clip(velocity_pool, v_min, v_max)

        V0[:] = velocity_pool.reshape(M, self.N, self.D)

        for m in range(M):
            L = rng.choice(L_values)

            center = rng.uniform(
                center_range[0],
                center_range[1],
                size=(1, self.D),
            )

            X0[m] = center + rng.uniform(
                -0.5 * L,
                0.5 * L,
                size=(self.N, self.D),
            )

        return self.pack(X0, V0)
    
    # -------------------------------
    # Case 9: levi_velocity_grid (cover more area)
    # -------------------------------
    def levi_velocity_grid(
        self,
        M: int,
        x_low: float = 0.0,
        x_high: float = 2.0,
        v_low: float = -1.0,
        v_high: float = 1.0,
        jitter: float = 0.03,
        seed: int | None = None,
    ) -> InitialConditionBatch:
        """
        Levi ICs with grid-like coverage of 2D velocity space.
        Useful for conditioning a tensor-product f(v) basis.
        """
        rng = np.random.default_rng(seed)

        if self.D != 2:
            raise ValueError("levi_velocity_grid currently assumes D=2.")

        X0 = rng.uniform(
            low=x_low,
            high=x_high,
            size=(M, self.N, self.D),
        )

        n_side = int(np.ceil(np.sqrt(M * self.N)))

        v1 = np.linspace(v_low, v_high, n_side)
        v2 = np.linspace(v_low, v_high, n_side)

        V1, V2 = np.meshgrid(v1, v2, indexing="ij")
        velocity_pool = np.column_stack([V1.ravel(), V2.ravel()])
        rng.shuffle(velocity_pool)

        needed = M * self.N
        if velocity_pool.shape[0] < needed:
            repeats = int(np.ceil(needed / velocity_pool.shape[0]))
            velocity_pool = np.tile(velocity_pool, (repeats, 1))

        velocity_pool = velocity_pool[:needed]

        jitter_scale = jitter * (v_high - v_low)
        velocity_pool = velocity_pool + rng.normal(
            loc=0.0,
            scale=jitter_scale,
            size=velocity_pool.shape,
        )

        velocity_pool = np.clip(velocity_pool, v_low, v_high)

        V0 = velocity_pool.reshape(M, self.N, self.D)

        return self.pack(X0, V0)
    

    # -------------------------------
    # Case 10: levi_velocity_box (betetr f conditioning)
    # -------------------------------
    def levi_velocity_box(
        self,
        M: int,
        x_low: float = 0.0,
        x_high: float = 2.0,
        v_low: float = -1.0,
        v_high: float = 1.0,
        seed: int | None = None,
    ) -> InitialConditionBatch:
        """
        Levi ICs with broader rectangular velocity support.

        Positions:
            X0 ~ Uniform([x_low, x_high]^D)

        Velocities:
            V0 ~ Uniform([v_low, v_high]^D)

        Use this when learning nonparametric f(v), because it gives better
        tensor-product velocity-basis coverage than only positive velocities.
        """
        rng = np.random.default_rng(seed)

        X0 = rng.uniform(
            low=x_low,
            high=x_high,
            size=(M, self.N, self.D),
        )

        V0 = rng.uniform(
            low=v_low,
            high=v_high,
            size=(M, self.N, self.D),
        )

        return self.pack(X0, V0)
    
    # -------------------------------
    # Case 11: CS uniform
    # -------------------------------
    
    def cs_uniform(
        self,
        M: int,
        x_low: float = 0.1,
        x_high: float = 10.0,
        v_low: float = 0.1,
        v_high: float = 5.0,
        seed: int | None = None,
    ) -> InitialConditionBatch:
        """
        Cucker-Smale ICs matching ModelSelectionSystems/CS parameters.py.

        X0 ~ Uniform([x_low, x_high])
        V0 ~ Uniform([v_low, v_high])
        """
        rng = np.random.default_rng(seed)

        X0 = rng.uniform(
            low=x_low,
            high=x_high,
            size=(M, self.N, self.D),
        )

        V0 = rng.uniform(
            low=v_low,
            high=v_high,
            size=(M, self.N, self.D),
        )

        return self.pack(X0, V0)
    
    # -------------------------------
    # Case 12: Swarm_spread
    # -------------------------------
    def swarm_spread(
        self,
        M: int,
        v_eq: float = np.sqrt(1.6 / 0.5),
        L: float = 10.0,
        min_spread: float = 0.5,
        max_spread: float = 2.0,
        center_extent: float = 5.0,
        speed_variation: float = 0.1,
        seed: int | None = None,
    ) -> InitialConditionBatch:
        """
        Sample spatially diverse swarm initial conditions.

        Each trajectory has a different spatial scale. This produces both
        compact and widely dispersed initial swarms, thereby covering a broader
        range of pairwise distances.

        Parameters
        ----------
        M : int
            Number of initial conditions.
        v_eq : float
            Reference velocity magnitude.
        L : float
            Reference side length of the spatial domain.
        min_spread : float
            Minimum spatial scale relative to L.
        max_spread : float
            Maximum spatial scale relative to L.
        center_extent : float
            Maximum displacement of each swarm center from the origin.
        speed_variation : float
            Relative velocity-magnitude variation around v_eq.
            For example, 0.1 gives speeds in
            [0.9 * v_eq, 1.1 * v_eq].
        seed : int or None
            Random seed.

        Returns
        -------
        InitialConditionBatch
        """
        if M <= 0:
            raise ValueError("M must be positive.")

        if L <= 0:
            raise ValueError("L must be positive.")

        if not 0.0 < min_spread <= max_spread:
            raise ValueError(
                "Require 0 < min_spread <= max_spread."
            )

        if center_extent < 0:
            raise ValueError("center_extent must be nonnegative.")

        if not 0.0 <= speed_variation < 1.0:
            raise ValueError(
                "speed_variation must satisfy 0 <= speed_variation < 1."
            )

        rng = np.random.default_rng(seed)

        # --------------------------------------------------
        # Positions
        # --------------------------------------------------
        # Start with dimensionless positions in a centered unit square.
        X_relative = rng.uniform(
            -0.5,
            0.5,
            size=(M, self.N, self.D),
        )

        # Give every trajectory a different spatial extent.
        # The effective side length varies from
        # min_spread * L to max_spread * L.
        spatial_scales = rng.uniform(
            min_spread * L,
            max_spread * L,
            size=(M, 1, 1),
        )

        X_relative *= spatial_scales

        # Randomly move the center of each swarm.
        # Translation increases absolute state-space coverage, although it
        # does not change the within-swarm pairwise distances.
        centers = rng.uniform(
            -center_extent,
            center_extent,
            size=(M, 1, self.D),
        )

        X0 = X_relative + centers

        # --------------------------------------------------
        # Velocities
        # --------------------------------------------------
        directions = rng.normal(
            size=(M, self.N, self.D),
        )
        directions = self._normalize_rows(directions)

        speed_min = (1.0 - speed_variation) * v_eq
        speed_max = (1.0 + speed_variation) * v_eq

        speeds = rng.uniform(
            speed_min,
            speed_max,
            size=(M, self.N, 1),
        )

        V0 = directions * speeds

        return self.pack(X0, V0)

    # -------------------------------
    # Case 11: swarm uniform
    # -------------------------------
    
    def swarm_uniform(
        self,
        M: int,
        x_low: float = 0,
        x_high: float = 1,
        v_low: float = -2,
        v_high: float = 2,
        seed: int | None = None,
    ) -> InitialConditionBatch:
        """
        Cucker-Smale ICs matching ModelSelectionSystems/CS parameters.py.

        X0 ~ Uniform([x_low, x_high])
        V0 ~ Uniform([v_low, v_high])
        """
        rng = np.random.default_rng(seed)

        X0 = rng.uniform(
            low=x_low,
            high=x_high,
            size=(M, self.N, self.D),
        )

        V0 = rng.uniform(
            low=v_low,
            high=v_high,
            size=(M, self.N, self.D),
        )

        return self.pack(X0, V0)
