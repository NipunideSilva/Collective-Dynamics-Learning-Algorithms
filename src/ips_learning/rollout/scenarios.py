# It prevents confusing:

# training-horizon reconstruction
# longer-horizon prediction
# unseen-IC prediction


import numpy as np


class RolloutScenarioBuilder:
    """
    Define rollout scenarios in a clean, explicit way.
    """

    def training_horizon(self, dataset):
        """
        Use the same initial conditions and same time grid as the training dataset.

        Returns
        -------
        U0 : np.ndarray
        t_eval : np.ndarray
        metadata : dict
        """
        return (
            dataset.U0,
            dataset.t,
            {
                "scenario": "trajectory_reconstruction",
                "horizon_type": "training",
                "system_order": dataset.metadata.get("system_order"),
                "system_name": dataset.metadata.get("system_name"),
            },
        )

    def longer_horizon(self, U0: np.ndarray, t0: float, 
                       T_new: float, L_new: int, metadata: dict | None = None,):
        """
        Roll out from given initial conditions on a longer time horizon.

        Returns
        -------
        U0 : np.ndarray
        t_eval : np.ndarray
        metadata : dict
        """
        t_eval = np.linspace(t0, T_new, L_new)
        out = {} if metadata is None else dict(metadata)
        out.update(
            {
                "scenario": "trajectory_prediction",
                "horizon_type": "extended",
                "t0": float(t0),
                "T_new": float(T_new),
                "L_new": int(L_new),
            }
        )
        return U0, t_eval, out
    
#     builder.longer_horizon(
#     U0=dataset.U0,
#     t0=dataset.t[0],
#     T_new=2.0,
#     L_new=201,
#     metadata={
#         "system_order": dataset.metadata["system_order"],
#         "system_name": dataset.metadata["system_name"],
#     },
# )

    def unseen_initial_conditions(self, dataset):
        """
        Use the same horizon but unseen ICs from another dataset split.

        Returns
        -------
        U0 : np.ndarray
        t_eval : np.ndarray
        metadata : dict
        """
        return (
            dataset.U0,
            dataset.t,
            {
                "scenario": "trajectory_prediction",
                "horizon_type": "unseen_initial_conditions",
                "system_order": dataset.metadata.get("system_order"),
                "system_name": dataset.metadata.get("system_name"),
            },
        )