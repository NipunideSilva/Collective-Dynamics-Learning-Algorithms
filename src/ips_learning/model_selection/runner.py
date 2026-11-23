from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ips_learning.fitting.assembler import NormalEquationAssembler
from ips_learning.fitting.solver import LinearSystemSolver
from ips_learning.fitting.system_repository import NormalSystemRepository
from ips_learning.models.factories import LearnedModelFactory
from ips_learning.model_selection.candidates import ModelCandidate
from ips_learning.model_selection.repository import ModelSelectionRepository

from ips_learning.features.feature_builder import (
    FeatureBuilder,
    dataset_for_candidate_from_view,
    make_maximal_derivative_view,
)

from ips_learning.model_selection.scoring import (
    SelectionConfig,
    select_and_save,
)

from ips_learning.model_selection.information_criteria import (
    count_model_coefficients,
    compute_gaussian_information_criteria,
)

@dataclass
class CandidateRunConfig:
    system_name: str
    profile_name: str
    dataset_name: str

    overwrite_fit: bool = False
    overwrite_metrics: bool = True

    ridge_lambda: float = 1e-6
    solver_method: str = "lstsq"
    svd_tol: float = 1e-12

    parallel_jobs: int = 1

    primary_metric: str = "E_traj_full"

    # AIC/BIC configuration
    include_noise_variance_parameter: bool = True


@dataclass
class CandidateRunResult:
    candidate_name: str
    fit_path: Path | None
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any] | None
    artifact_references: dict[str, str]


class ModelSelectionRunner:
    """
    Run model-selection candidates using the existing pipeline.

    This class should not duplicate the fitting math. It only orchestrates:
        candidate config
        fit/load
        learned model construction
        rollout/evaluation callbacks
        artifact registration
    """

    def __init__(
        self,
        *,
        fit_repo: NormalSystemRepository | None = None,
        ms_repo: ModelSelectionRepository | None = None,
        model_factory: LearnedModelFactory | None = None,
    ) -> None:
        self.fit_repo = fit_repo or NormalSystemRepository()
        self.ms_repo = ms_repo or ModelSelectionRepository()
        self.model_factory = model_factory or LearnedModelFactory()
        self.feature_builder = FeatureBuilder()

    def run_candidate(
        self,
        *,
        candidate: ModelCandidate,
        train_dataset,
        features_train_fit,
        basis_bundle,
        validation_dataset,
        config: CandidateRunConfig,
        test_dataset=None,
        rollout_fn: Callable[..., Any] | None = None,
        evaluate_fn: Callable[..., dict[str, Any]] | None = None,
        kernel_diagnostics_fn: Callable[..., dict[str, Any]] | None = None,
    ) -> CandidateRunResult:
        """
        Run one candidate.

        Parameters
        ----------
        rollout_fn:
            User/project callback that creates rollout artifacts from learned model.

            Expected signature:
                rollout_fn(
                    learned_model=...,
                    dataset=...,
                    split="train","validation" or "test",
                    candidate=...,
                    config=...,
                )

        evaluate_fn:
            User/project callback that computes metrics.

            Expected signature:
                evaluate_fn(
                    learned_model=...,
                    dataset=...,
                    rollout_artifact=...,
                    split="validation" or "test",
                    candidate=...,
                    config=...,
                )

        kernel_diagnostics_fn:
            Optional callback for computing phi_E/phi_A/f diagnostics from fit_result.

            Expected signature:
                kernel_diagnostics_fn(
                    fit_result=...,
                    learned_model=...,
                    candidate=...,
                    config=...,
                )
        """

        candidate.validate()

        candidate_metadata = {
            "candidate": candidate.name,
            "system_order": candidate.system_order,
            "interaction_type": candidate.interaction_type,
            "algorithm": candidate.algorithm,
            "description": candidate.description,
            "include_environment": getattr(candidate, "include_environment", True),
            "complexity": getattr(candidate, "complexity", None),
        }

        self.ms_repo.save_candidate_metadata(
            system_name=config.system_name,
            profile_name=config.profile_name,
            dataset_name=config.dataset_name,
            candidate_name=candidate.name,
            metadata=candidate_metadata,
        )

        fit_case_name = self._candidate_fit_case_name(
            dataset_name=config.dataset_name,
            candidate_name=candidate.name,
        )

        fit_result = self._fit_or_load_candidate(
            candidate=candidate,
            features_train_fit=features_train_fit,
            basis_bundle=basis_bundle,
            config=config,
            fit_case_name=fit_case_name,
        )

        fit_path = self.fit_repo.case_dir(
            system_name=config.system_name,
            profile_name=config.profile_name,
            algorithm=candidate.algorithm,
            fit_case_name=fit_case_name,
        ) / "fit_result.pkl"

        learned_model = self.model_factory.build_from_fit_result(fit_result)

        artifact_references = {
            "fit_result": str(fit_path),
        }

        #----setup the train rollout dir----
        train_rollout, train_rollout_failure = self._rollout_or_failed(
            learned_model=learned_model,
            dataset=train_dataset,
            split="train",
            candidate=candidate,
            config=config,
            rollout_fn=rollout_fn,
        )

        if train_rollout_failure is not None:
            train_metrics = train_rollout_failure
        else:
            artifact_references["train_rollout"] = self._artifact_to_str(train_rollout)

            train_metrics = self._evaluate_or_default(
                learned_model=learned_model,
                dataset=train_dataset,
                rollout_artifact=train_rollout,
                split="train",
                candidate=candidate,
                config=config,
                evaluate_fn=evaluate_fn,
            )
        # Calculate training-likelihood AIC and BIC.
        train_metrics = self._attach_information_criteria(
            train_metrics=train_metrics,
            fit_result=fit_result,
            config=config,
        )

        self.ms_repo.save_candidate_metrics(
            system_name=config.system_name,
            profile_name=config.profile_name,
            dataset_name=config.dataset_name,
            candidate_name=candidate.name,
            metrics=train_metrics,
            split="train",
        )

         #----setup the validation rollout dir----
        validation_rollout, validation_rollout_failure = self._rollout_or_failed(
            learned_model=learned_model,
            dataset=validation_dataset,
            split="validation",
            candidate=candidate,
            config=config,
            rollout_fn=rollout_fn,
        )

        if validation_rollout_failure is not None:
            validation_metrics = validation_rollout_failure
        else:
            artifact_references["validation_rollout"] = self._artifact_to_str(
                validation_rollout
            )

            validation_metrics = self._evaluate_or_default(
                learned_model=learned_model,
                dataset=validation_dataset,
                rollout_artifact=validation_rollout,
                split="validation",
                candidate=candidate,
                config=config,
                evaluate_fn=evaluate_fn,
            )
        #--------------saving purpose-combine two diagnostics-----
        # --------------------------------------------------
        # Combine train and validation metrics
        # --------------------------------------------------
        combined_selection_metrics = {
            **train_metrics,
            **validation_metrics,
        }

        combined_selection_metrics["failed"] = (
            bool(train_metrics.get("failed", False))
            or bool(validation_metrics.get("failed", False))
        )

        combined_selection_metrics["status"] = (
            "failed_rollout"
            if combined_selection_metrics["failed"]
            else "ok"
        )

        combined_selection_metrics.setdefault(
            "condition_number",
            float(fit_result.diagnostics.condition_number),
        )

        self.ms_repo.save_candidate_metrics(
            system_name=config.system_name,
            profile_name=config.profile_name,
            dataset_name=config.dataset_name,
            candidate_name=candidate.name,
            metrics=combined_selection_metrics,#validation_metrics,
            split="validation",
        )

        # test_metrics = None
        # if test_dataset is not None:
        #     test_rollout = None
        #     if rollout_fn is not None:
        #         test_rollout = rollout_fn(
        #             learned_model=learned_model,
        #             dataset=test_dataset,
        #             split="test",
        #             candidate=candidate,
        #             config=config,
        #         )
        #         artifact_references["test_rollout"] = self._artifact_to_str(
        #             test_rollout
        #         )

        #     test_metrics = self._evaluate_or_default(
        #         learned_model=learned_model,
        #         dataset=test_dataset,
        #         rollout_artifact=test_rollout,
        #         split="test",
        #         candidate=candidate,
        #         config=config,
        #         evaluate_fn=evaluate_fn,
        #     )

        #     test_metrics.setdefault(
        #         "condition_number",
        #         float(fit_result.diagnostics.condition_number),
        #     )

        #     self.ms_repo.save_candidate_metrics(
        #         system_name=config.system_name,
        #         profile_name=config.profile_name,
        #         dataset_name=config.dataset_name,
        #         candidate_name=candidate.name,
        #         metrics=test_metrics,
        #         split="test",
        #     )

        if kernel_diagnostics_fn is not None:
            kernel_diagnostics = kernel_diagnostics_fn(
                fit_result=fit_result,
                learned_model=learned_model,
                candidate=candidate,
                config=config,
                features=features_train_fit,
            )

            self.ms_repo.save_kernel_diagnostics(
                system_name=config.system_name,
                profile_name=config.profile_name,
                dataset_name=config.dataset_name,
                candidate_name=candidate.name,
                diagnostics=kernel_diagnostics,
            )

            artifact_references["kernel_diagnostics"] = str(
                self.ms_repo.candidate_dir(
                    system_name=config.system_name,
                    profile_name=config.profile_name,
                    dataset_name=config.dataset_name,
                    candidate_name=candidate.name,
                )
                / "kernel_diagnostics.npz"
            )

        self.ms_repo.save_candidate_references(
            system_name=config.system_name,
            profile_name=config.profile_name,
            dataset_name=config.dataset_name,
            candidate_name=candidate.name,
            references=artifact_references,
        )

        return CandidateRunResult(
            candidate_name=candidate.name,
            fit_path=fit_path,
            validation_metrics=combined_selection_metrics,
            test_metrics=None,
            artifact_references=artifact_references,
        )

    def run_candidates(
        self,
        *,
        candidates: list[ModelCandidate],
        train_dataset,
        basis_bundle,
        validation_dataset,
        config: CandidateRunConfig,
        test_dataset=None,
        rollout_fn: Callable[..., Any] | None = None,
        evaluate_fn: Callable[..., dict[str, Any]] | None = None,
        kernel_diagnostics_fn: Callable[..., dict[str, Any]] | None = None,
    ) -> list[CandidateRunResult]:
        results = []


        for candidate in candidates:

            candidate_features_train_fit = self.feature_builder.build_for_candidate(
                dataset=train_dataset,
                candidate=candidate,
            )
            train_view = make_maximal_derivative_view(train_dataset)
            validation_view = make_maximal_derivative_view(validation_dataset)
            candidate_train_dataset = dataset_for_candidate_from_view(train_view, candidate)
            candidate_validation_dataset = dataset_for_candidate_from_view(
                validation_view,
                candidate,
            )
            result = self.run_candidate(
                candidate=candidate,
                train_dataset=candidate_train_dataset,
                features_train_fit=candidate_features_train_fit,
                basis_bundle=basis_bundle,
                validation_dataset=candidate_validation_dataset,
                test_dataset=test_dataset,
                config=config,
                rollout_fn=rollout_fn,
                evaluate_fn=evaluate_fn,
                kernel_diagnostics_fn=kernel_diagnostics_fn,
            )
            results.append(result)

        return results

    def _fit_or_load_candidate(
        self,
        *,
        candidate: ModelCandidate,
        features_train_fit,
        basis_bundle,
        config: CandidateRunConfig,
        fit_case_name: str,
    ):
        fit_dir = self.fit_repo.case_dir(
            system_name=config.system_name,
            profile_name=config.profile_name,
            algorithm=candidate.algorithm,
            fit_case_name=fit_case_name,
        )
        fit_path = fit_dir / "fit_result.pkl"

        if fit_path.exists() and not config.overwrite_fit:
            return self.fit_repo.load_fit_result(
                system_name=config.system_name,
                profile_name=config.profile_name,
                algorithm=candidate.algorithm,
                fit_case_name=fit_case_name,
            )

        solver = LinearSystemSolver(
            method=config.solver_method,
            ridge_lambda=config.ridge_lambda,
            svd_tol=config.svd_tol,
        )

        assembler = NormalEquationAssembler(
            parallel_jobs=config.parallel_jobs,
            interaction_type=candidate.interaction_type,
            algorithm=candidate.algorithm,
            include_environment=getattr(candidate, "include_environment", True),
        )

        fit_result = assembler.fit(
            features=features_train_fit,
            basis_bundle=basis_bundle,
            solver=solver,
            system_name=config.system_name,
        )

        self.fit_repo.save(
            system_name=config.system_name,
            profile_name=config.profile_name,
            fit_case_name=fit_case_name,
            A=fit_result.A,
            b=fit_result.b,
            theta=fit_result.theta,
            n_radial=fit_result.n_radial,
            n_energy=getattr(fit_result, "n_energy", None),
            n_alignment=getattr(fit_result, "n_alignment", None),
            n_environment=fit_result.n_environment,
            diagnostics=self.fit_repo._diagnostics_to_dict(
                fit_result.diagnostics
            ),
            metadata=fit_result.metadata,
            overwrite=True,
        )

        self.fit_repo.save_fit_result(
            system_name=config.system_name,
            profile_name=config.profile_name,
            fit_case_name=fit_case_name,
            fit_result=fit_result,
            algorithm=candidate.algorithm,
            overwrite=True,
        )

        return fit_result

    @staticmethod
    def _attach_information_criteria(
            *,
            train_metrics: dict[str, Any],
            fit_result,
            config: CandidateRunConfig,
        ) -> dict[str, Any]:
            """
            Attach Gaussian AIC and BIC values to the training metrics.

            AIC and BIC are based on the common position-trajectory
            reconstruction likelihood for all candidate frameworks.
            """
            metrics = dict(train_metrics)

            # Count every coefficient fitted for the candidate.
            n_model_coefficients = count_model_coefficients(fit_result)

            metrics["n_model_coefficients"] = n_model_coefficients

            # AIC/BIC cannot be calculated when the training rollout failed,
            # because no common trajectory reconstruction RSS is available.
            if bool(metrics.get("failed", False)):
                metrics.update(
                    {
                        "information_criteria_status": "not_available",
                        "n_variance_parameters": (
                            1
                            if config.include_noise_variance_parameter
                            else 0
                        ),
                        "n_likelihood_parameters": None,
                        "sigma2_mle": None,
                        "log_likelihood": None,
                        "negative_two_log_likelihood": None,
                        "AIC": None,
                        "BIC": None,
                    }
                )
                return metrics

            rss = metrics.get("trajectory_reconstruction_rss")
            n_observations = metrics.get(
                "trajectory_reconstruction_n_obs"
            )

            if rss is None:
                raise KeyError(
                    "Successful training metrics are missing "
                    "'trajectory_reconstruction_rss'. "
                    "Complete Step 2 in callbacks.py before calculating "
                    "AIC and BIC."
                )

            if n_observations is None:
                raise KeyError(
                    "Successful training metrics are missing "
                    "'trajectory_reconstruction_n_obs'. "
                    "Complete Step 2 in callbacks.py before calculating "
                    "AIC and BIC."
                )

            information_criteria = (
                compute_gaussian_information_criteria(
                    rss=float(rss),
                    n_observations=int(n_observations),
                    n_model_coefficients=n_model_coefficients,
                    include_noise_variance_parameter=(
                        config.include_noise_variance_parameter
                    ),
                )
            )

            metrics.update(information_criteria)
            metrics["information_criteria_status"] = "ok"

            return metrics

    def _evaluate_or_default(
        self,
        *,
        learned_model,
        dataset,
        rollout_artifact,
        split: str,
        candidate: ModelCandidate,
        config: CandidateRunConfig,
        evaluate_fn: Callable[..., dict[str, Any]] | None,
    ) -> dict[str, Any]:
        if evaluate_fn is None:
            return {
                "status": "not_evaluated",
                "failed": True,
                "candidate": candidate.name,
                "split": split,
                "reason": "No evaluate_fn was provided.",
            }

        try:
            metrics = evaluate_fn(
                learned_model=learned_model,
                dataset=dataset,
                rollout_artifact=rollout_artifact,
                split=split,
                candidate=candidate,
                config=config,
            )
        except Exception as exc:
            return {
                "status": "failed",
                "failed": True,
                "candidate": candidate.name,
                "split": split,
                "error": repr(exc),
            }

        metrics = dict(metrics)
        metrics.setdefault("status", "ok")
        metrics.setdefault("failed", False)
        metrics.setdefault("candidate", candidate.name)
        metrics.setdefault("split", split)

        return metrics

    @staticmethod
    def _candidate_fit_case_name(
        *,
        dataset_name: str,
        candidate_name: str,
    ) -> str:
        return f"{dataset_name}__{candidate_name}"

    @staticmethod
    def _artifact_to_str(obj: Any) -> str:
        if obj is None:
            return ""

        if isinstance(obj, (str, Path)):
            return str(obj)

        if hasattr(obj, "path"):
            return str(obj.path)

        if hasattr(obj, "save_path"):
            return str(obj.save_path)

        return repr(obj)
    
    #orchestration method that runs candidates, loads their saved metrics, applies scoring, and writes
    def run_select_and_save(
        self,
        *,
        candidates: list[ModelCandidate],
        train_dataset,
        basis_bundle,
        validation_dataset,
        config: CandidateRunConfig,
        test_dataset=None,
        rollout_fn: Callable[..., Any] | None = None,
        evaluate_fn: Callable[..., dict[str, Any]] | None = None,
        kernel_diagnostics_fn: Callable[..., dict[str, Any]] | None = None,
        selection_config: SelectionConfig | None = None,
    ):
        """
        Main model-selection entry point.

        What happens:
        1. Fit or load each candidate.
        2. Build learned model.
        3. Use rollout_fn to generate validation/test rollouts.
        Prefer existing src/ips_learning/rollout code inside rollout_fn.
        4. Use evaluate_fn to compute train/validation/test metrics.
        5. Save candidate metrics and artifact references.
        6. Rank candidates using validation metrics.
        7. Save model_selection_summary.csv/.tex.
        8. Save selected_model.json.
        """

        run_results = self.run_candidates(
            candidates=candidates,
            train_dataset=train_dataset,
            basis_bundle=basis_bundle,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
            config=config,
            rollout_fn=rollout_fn,
            evaluate_fn=evaluate_fn,
            kernel_diagnostics_fn=kernel_diagnostics_fn,
        )

        selected, summary_df = select_and_save(
            repo=self.ms_repo,
            system_name=config.system_name,
            profile_name=config.profile_name,
            dataset_name=config.dataset_name,
            candidates=candidates,
            split="validation",
            config=selection_config or SelectionConfig(),
        )

        selected_test_metrics = None

        if test_dataset is not None and selected.get("selection_status") == "selected":
            selected_test_metrics = self.run_selected_test(
                selected=selected,
                test_dataset=test_dataset,
                config=config,
                rollout_fn=rollout_fn,
                evaluate_fn=evaluate_fn,
            )
        

        return {
            "run_results": run_results,
            "selected": selected,
            "summary": summary_df,
            "selected_test_metrics": selected_test_metrics,
        }
    
    def _rollout_or_failed(
        self,
        *,
        learned_model,
        dataset,
        split: str,
        candidate: ModelCandidate,
        config: CandidateRunConfig,
        rollout_fn,
    ):
        if rollout_fn is None:
            return None, {
                "status": "not_rolled_out",
                "failed": True,
                "candidate": candidate.name,
                "split": split,
                "reason": "No rollout_fn was provided.",
            }

        try:
            rollout = rollout_fn(
                learned_model=learned_model,
                dataset=dataset,
                split=split,
                candidate=candidate,
                config=config,
            )
            return rollout, None

        except Exception as exc:
            return None, {
                "status": "failed_rollout",
                "failed": True,
                "candidate": candidate.name,
                "split": split,
                "error": repr(exc),
            }
        
    def run_selected_test(
        self,
        *,
        selected: dict[str, Any],
        test_dataset,
        config: CandidateRunConfig,
        rollout_fn,
        evaluate_fn,
    ) -> dict[str, Any] | None:
        candidate_name = selected.get("candidate")

        if not candidate_name or test_dataset is None:
            return None

        refs = self.ms_repo.load_candidate_references(
            system_name=config.system_name,
            profile_name=config.profile_name,
            dataset_name=config.dataset_name,
            candidate_name=candidate_name,
        )

        fit_result = self.fit_repo.load_fit_result(
            system_name=config.system_name,
            profile_name=config.profile_name,
            algorithm=selected.get("algorithm", "nonparametric_f"),
            fit_case_name=self._candidate_fit_case_name(
                dataset_name=config.dataset_name,
                candidate_name=candidate_name,
            ),
        )

        learned_model = self.model_factory.build_from_fit_result(fit_result)

        # minimal candidate-like object is not needed if callbacks only use name;
        # safer: find candidate object outside if needed later
        candidate = ModelCandidate(
            name=candidate_name,
            system_order=int(selected.get("system_order")),
            interaction_type=selected.get("interaction_type"),
            algorithm=selected.get("algorithm", "nonparametric_f"),
            include_environment=bool(selected.get("include_environment", True)),
            complexity=int(selected.get("complexity", 99)),
        )

        test_rollout, test_failure = self._rollout_or_failed(
            learned_model=learned_model,
            dataset=test_dataset,
            split="test",
            candidate=candidate,
            config=config,
            rollout_fn=rollout_fn,
        )

        if test_failure is not None:
            test_metrics = test_failure
        else:
            refs["test_rollout"] = self._artifact_to_str(test_rollout)

            test_metrics = self._evaluate_or_default(
                learned_model=learned_model,
                dataset=test_dataset,
                rollout_artifact=test_rollout,
                split="test",
                candidate=candidate,
                config=config,
                evaluate_fn=evaluate_fn,
            )
        test_metrics.setdefault(
            "condition_number",
            float(fit_result.diagnostics.condition_number),
        )

        self.ms_repo.save_candidate_metrics(
            system_name=config.system_name,
            profile_name=config.profile_name,
            dataset_name=config.dataset_name,
            candidate_name=candidate_name,
            metrics=test_metrics,
            split="test",
        )

        self.ms_repo.save_candidate_references(
            system_name=config.system_name,
            profile_name=config.profile_name,
            dataset_name=config.dataset_name,
            candidate_name=candidate_name,
            references=refs,
        )

        return test_metrics