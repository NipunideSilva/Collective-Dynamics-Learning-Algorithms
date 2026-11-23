import numpy as np


class TrialSummaryBuilder:
    """
    Aggregate repeated-run metrics into mean/std summaries.
    """

    def from_trial_outputs(self, trial_outputs: list[dict]) -> dict:
        residuals = []
        recon = []
        pred = []

        for out in trial_outputs:
            report = out["evaluation_report"]

            residuals.append(report.residual_metrics["E_res"])
            recon.append(
                report.trajectory_metrics["reconstruction"]["reconstruction_Linf_mean"]
            )

            if "prediction" in report.trajectory_metrics:
                pred.append(
                    report.trajectory_metrics["prediction"]["reconstruction_Linf_mean"]
                )

        summary = {
            "num_trials": len(trial_outputs),
            "E_res_mean": float(np.mean(residuals)),
            "E_res_std": float(np.std(residuals)),
            "E_recon_mean": float(np.mean(recon)),
            "E_recon_std": float(np.std(recon)),
        }

        if len(pred) > 0:
            summary["E_pred_mean"] = float(np.mean(pred))
            summary["E_pred_std"] = float(np.std(pred))

        return summary