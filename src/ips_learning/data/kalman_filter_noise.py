from __future__ import annotations

import numpy as np
from ips_learning.data.dataset_repository import RawTrajectoryDataset


class KalmanFilterNoise:
    """
    Apply controlled noise to trajectory datasets.

    Supported noise types:
        additive_gaussian:
            A_noisy = A + zeta * std(A) * eta,
            eta ~ N(0, 1)

        multiplicative_uniform:
            A_noisy = A * (1 + epsilon),
            epsilon ~ Uniform[-zeta, zeta]
    """

    def __init__(
        self,
        noise_level: float,
        seed: int = 0,
        noise_type: str = "additive_gaussian",
    ) -> None:
        if noise_type not in {"additive_gaussian", "multiplicative_uniform"}:
            raise ValueError(
                "noise_type must be 'additive_gaussian' or "
                f"'multiplicative_uniform', got {noise_type!r}."
            )

        self.noise_level = float(noise_level)
        self.rng = np.random.default_rng(seed)
        self.noise_type = noise_type

    def perturb(self, dataset: RawTrajectoryDataset) -> tuple[RawTrajectoryDataset, RawTrajectoryDataset]:
        zeta = self.noise_level

        if self.noise_type == "additive_gaussian":

            def noisy(A):
                scale = np.std(A)
                return A + zeta * scale * self.rng.normal(size=A.shape)

            noise_model = "additive_gaussian_scaled_by_global_std"

        else:

            def noisy(A):
                eps = self.rng.uniform(
                    -zeta,
                    zeta,
                    size=A.shape,
                )
                return A * (1.0 + eps)

            noise_model = "multiplicative_uniform_relative"

        X_noisy = noisy(dataset.X)
        dt = dataset.t[1] - dataset.t[0]  # Calculate time step from time vector
            

        def filter_trajectory_dataset_2nd_order(X_noisy, dt):
            # 2. Extract dimensions from your dataset tensor (N trajectories, L timesteps, D dimensions)
            M, L, N, D = X_noisy.shape
            
            # 3. Initialize output arrays with the exact same shape
            X_smooth = np.zeros_like(X_noisy)
            V_smooth = np.zeros_like(X_noisy)
            A_smooth = np.zeros_like(X_noisy)
            
            # 4. Set up standard Kinematic Kalman Filter Matrices
            A_mat = np.array([
                        [1.0, dt, 0.5 * dt**2],
                        [0.0, 1.0, dt],
                        [0.0, 0.0, 1.0],
                    ])

            H_mat = np.array([
                        [1.0, 0.0, 0.0],
                    ])
            I_matrix = np.eye(3)
            
            # Hyperparameters based on your noise level (zeta)
            sensor_noise_std = float(zeta) if float(zeta) > 0 else 0.01
            R_mat = np.array([[sensor_noise_std**2]])
            
            q_accel_variance = 0.5 
            G_mat = np.array([[0.5 * dt**2], [dt], [1.0]])
            Q_mat = (G_mat @ G_mat.T * q_accel_variance)

            # 5. Core Filtering Loop over Agents and Spatial Dimensions
            for m in range(M):
                for n in range(N):
                    for d in range(D):
                        noisy_series = X_noisy[m, :, n, d]
                    
                        # --- FORWARD PASS (Kalman Filter) ---
                        # Containers to store forward predictions and updates for the backward pass
                        xf_estimates = np.zeros((3, L))
                        xf_predictions = np.zeros((3, L))
                        Pf_estimates = np.zeros((3, 3, L))
                        Pf_predictions = np.zeros((3, 3, L))
                        
                        x_est = np.array([[noisy_series[0]], [0.0], [0.0]])
                        P = np.eye(3) * 10.0 
                        
                        for k in range(L):
                            # Predict
                            x_pred = A_mat @ x_est
                            P_pred = A_mat @ P @ A_mat.T + Q_mat
                            
                            # Store predictions
                            xf_predictions[:, k] = x_pred.ravel()
                            Pf_predictions[:, :, k] = P_pred
                            
                            # Update
                            measurement = noisy_series[k]
                            residual = np.array([
                                    [measurement]
                                ]) - H_mat @ x_pred
                            S = H_mat @ P_pred @ H_mat.T + R_mat
                            K = P_pred @ H_mat.T @ np.linalg.inv(S)
                            
                            x_est = (
                                    x_pred
                                    + K @ np.atleast_1d(residual).reshape(1, 1)
                                )
                            P = (I_matrix - K @ H_mat) @ P_pred
                            
                            # Store updates
                            xf_estimates[:, k] = x_est.ravel()
                            Pf_estimates[:, :, k] = P
                        
                        # --- BACKWARD PASS (RTS Smoother) ---
                        # Eliminates time lag entirely for clean least-squares optimization matrices
                        xs_smooth = np.zeros((3, L))
                        xs_smooth[:, -1] = xf_estimates[:, -1]
                        
                        for k in range(L - 2, -1, -1):
                            xf_pred_k1 = xf_predictions[:, k+1].reshape(3, 1)
                            Pf_pred_k1 = Pf_predictions[:, :, k+1]
                            xf_est_k = xf_estimates[:, k].reshape(3, 1)
                            Pf_est_k = Pf_estimates[:, :, k]
                            xs_smooth_k1 = xs_smooth[:, k+1].reshape(3, 1)
                            
                            # Optimal smoothing gain
                            C = Pf_est_k @ A_mat.T @ np.linalg.inv(Pf_pred_k1)
                            
                            # Corrected state update
                            xs_k = xf_est_k + C @ (xs_smooth_k1 - xf_pred_k1)
                            xs_smooth[:, k] = xs_k.ravel()
                            
                        # Assign optimized arrays back to main tensor
                        X_smooth[m, :, n, d] = xs_smooth[0, :]# Cleaned Position
                        V_smooth[m, :, n, d] = xs_smooth[1, :]# Cleaned Velocity
                        A_smooth[m, :, n, d] = xs_smooth[2, :]# Cleaned Acceleration
                        
            return X_smooth, V_smooth, A_smooth
        
        def filter_trajectory_dataset_1st_order(X_noisy, dt):
            # 2. Extract dimensions from your dataset tensor (N trajectories, L timesteps, D dimensions)
            M, L, N, D = X_noisy.shape
            
            # 3. Initialize output arrays with the exact same shape
            X_smooth = np.zeros_like(X_noisy)
            V_smooth = np.zeros_like(X_noisy)
            
            # 4. Set up standard Kinematic Kalman Filter Matrices
            A_mat = np.array([
                        [1.0, dt],
                        [0.0, 1.0],
                    ])

            H_mat = np.array([[1.0, 0.0]])
            I_matrix = np.eye(2)
            # Hyperparameters based on your noise level (zeta)
            sensor_noise_var = max(  (float(zeta)**2 / 3.0) * np.mean(X_noisy**2), 1e-12)
            R_mat = np.array([[sensor_noise_var]])
            
            q_velocity_variance = 5

            G_mat = np.array([
                [0.5 * dt**2],
                [dt],
            ])

            Q_mat = (G_mat @ G_mat.T * q_velocity_variance)

            # 5. Core Filtering Loop over Agents and Spatial Dimensions
            for m in range(M):
                for n in range(N):
                    for d in range(D):
                        noisy_series = X_noisy[m, :, n, d]
                    
                        # --- FORWARD PASS (Kalman Filter) ---
                        # Containers to store forward predictions and updates for the backward pass
                        xf_estimates = np.zeros((2, L))
                        xf_predictions = np.zeros((2, L))
                        Pf_estimates = np.zeros((2, 2, L))
                        Pf_predictions = np.zeros((2, 2, L))
                        
                        x_est = np.array([[noisy_series[0]], [0.0]])
                        P = np.eye(2) * 10.0 
                        
                        for k in range(L):
                            # Predict
                            x_pred = A_mat @ x_est
                            P_pred = A_mat @ P @ A_mat.T + Q_mat
                            
                            # Store predictions
                            xf_predictions[:, k] = x_pred.ravel()
                            Pf_predictions[:, :, k] = P_pred
                            
                            # Update
                            measurement = noisy_series[k]
                            residual = np.array([
                                            [measurement]
                                        ]) - H_mat @ x_pred

                            S = H_mat @ P_pred @ H_mat.T + R_mat
                            K = P_pred @ H_mat.T @ np.linalg.inv(S)
                            
                            x_est = (
                                x_pred
                                + K @ np.atleast_1d(residual).reshape(1, 1)
                            )
                            P = (I_matrix - K * H_mat) @ P_pred
                            
                            # Store updates
                            xf_estimates[:, k] = x_est.ravel()
                            Pf_estimates[:, :, k] = P
                        
                        # --- BACKWARD PASS (RTS Smoother) ---
                        # Eliminates time lag entirely for clean least-squares optimization matrices
                        xs_smooth = np.zeros((2, L))
                        xs_smooth[:, -1] = xf_estimates[:, -1]
                        
                        for k in range(L - 2, -1, -1):
                            xf_pred_k1 = xf_predictions[:, k+1].reshape(2, 1)
                            Pf_pred_k1 = Pf_predictions[:, :, k+1]
                            xf_est_k = xf_estimates[:, k].reshape(2, 1)
                            Pf_est_k = Pf_estimates[:, :, k]
                            xs_smooth_k1 = xs_smooth[:, k+1].reshape(2, 1)
                            
                            # Optimal smoothing gain
                            C = Pf_est_k @ A_mat.T @ np.linalg.inv(Pf_pred_k1)
                            
                            # Corrected state update
                            xs_k = xf_est_k + C @ (xs_smooth_k1 - xf_pred_k1)
                            xs_smooth[:, k] = xs_k.ravel()
                            
                        # Assign optimized arrays back to main tensor
                        X_smooth[m, :, n, d] = xs_smooth[0, :]# Cleaned Position
                        V_smooth[m, :, n, d] = xs_smooth[1, :]# Cleaned Velocity

                        
            return X_smooth, V_smooth
        
       
        # 6. Align outputs to match your specific conditional logic structure
        if dataset.V is None:
            X_smooth, V_smooth = filter_trajectory_dataset_1st_order(X_noisy, dt)
            V_out = None
            V_noisy = None
            LHS_out = V_smooth
            LHS_noisy = np.gradient(X_noisy, dataset.t, axis=1)
        else:
            X_smooth, V_smooth, A_smooth = filter_trajectory_dataset_2nd_order(X_noisy, dt)
            V_out = V_smooth
            V_noisy = np.gradient(X_noisy, dataset.t, axis=1)
            
            LHS_out = A_smooth
            LHS_noisy = np.gradient(V_noisy, dataset.t, axis=1)


        noisy_dataset = RawTrajectoryDataset(
            t=dataset.t,
            X=X_noisy,
            V=V_noisy,
            LHS=LHS_noisy,
            U0=dataset.U0,
            metadata={
                **dataset.metadata,
                "noise_model": noise_model,
                "noise_type": self.noise_type,
                "noise_level": float(zeta),
                "filter_applied": "None (Raw Noisy Data)",
            },
        )

        smoothed_dataset = RawTrajectoryDataset(
            t=dataset.t,
            X=X_smooth,  
            V=V_out,
            LHS=LHS_out,
            U0=dataset.U0,
            metadata={
                **dataset.metadata,
                "noise_model": noise_model,
                "noise_type": self.noise_type,
                "noise_level": float(zeta),
                "filter_applied": "Kinematic RTS Smoother",
            },
        )           


        return  noisy_dataset, smoothed_dataset


