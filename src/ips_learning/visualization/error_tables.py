import pandas as pd


class ErrorTableBuilder:
    """
    Stage 2.6: build quantitative summary tables for reconstruction and prediction.
    """

    def build_reconstruction_prediction_summary(
        self,
        report_recon,
        metrics_extended_train,
        metrics_test,
    ) -> pd.DataFrame:
        rows = []

        # Reconstruction on train_fit [0,T]
        recon = report_recon.trajectory_metrics["reconstruction"]
        rows.append({
            "scenario": "train_fit [0,T]",
            "role": "reconstruction",
            "Linf_mean": recon["reconstruction_Linf_mean"],
            "Linf_std": recon["reconstruction_Linf_std"],
            "curve_source": "report_recon",
        })

        # Same-IC full train split
        rows.append({
            "scenario": "train same IC [0,T]",
            "role": "same-IC pre-T",
            "Linf_mean": metrics_extended_train["reconstruction_Linf_mean"],
            "Linf_std": metrics_extended_train["reconstruction_Linf_std"],
            "curve_source": "metrics_extended_train",
        })
        rows.append({
            "scenario": "train same IC [T,T_f]",
            "role": "same-IC prediction",
            "Linf_mean": metrics_extended_train["prediction_Linf_mean"],
            "Linf_std": metrics_extended_train["prediction_Linf_std"],
            "curve_source": "metrics_extended_train",
        })

        # Test split
        rows.append({
            "scenario": "test unseen IC [0,T]",
            "role": "unseen-IC pre-T",
            "Linf_mean": metrics_test["reconstruction_Linf_mean"],
            "Linf_std": metrics_test["reconstruction_Linf_std"],
            "curve_source": "metrics_test",
        })
        rows.append({
            "scenario": "test unseen IC [T,T_f]",
            "role": "unseen-IC prediction",
            "Linf_mean": metrics_test["prediction_Linf_mean"],
            "Linf_std": metrics_test["prediction_Linf_std"],
            "curve_source": "metrics_test",
        })

        return pd.DataFrame(rows)

    def build_scalar_metric_table(
        self,
        report_recon,
    ) -> pd.DataFrame:
        rows = []

        rows.append({
            "metric": "E_res",
            "value": report_recon.residual_metrics["E_res"],
            "notes": "Residual on train_fit observed states",
        })

        if "E_phi" in report_recon.function_metrics:
            rows.append({
                "metric": "E_phi",
                "value": report_recon.function_metrics["E_phi"],
                "notes": "Kernel error weighted by Quantity B density",
            })

        if "E_env" in report_recon.function_metrics:
            rows.append({
                "metric": "E_env",
                "value": report_recon.function_metrics["E_env"],
                "notes": "Environment error weighted by Quantity B density",
            })

        return pd.DataFrame(rows)