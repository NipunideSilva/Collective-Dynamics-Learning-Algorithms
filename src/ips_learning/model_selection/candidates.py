from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCandidate:
    """
    Candidate model specification for model selection.

    This object does not run fitting. It only describes which model form
    should be passed to the existing pipeline.
    """

    name: str
    system_order: int
    interaction_type: str
    algorithm: str = "nonparametric_f"
    include_environment: bool = True
    complexity: int = 99
    description: str = ""

    def validate(self) -> None:
        if self.system_order not in (1, 2):
            raise ValueError(
                f"system_order must be 1 or 2, got {self.system_order}."
            )

        if self.interaction_type not in {
            "energy",
            "alignment",
            "energy_alignment",
        }:
            raise ValueError(
                "interaction_type must be 'energy', 'alignment', "
                f"or 'energy_alignment', got {self.interaction_type!r}."
            )

        if self.system_order == 1 and self.interaction_type != "energy":
            raise ValueError(
                "First-order candidates only support interaction_type='energy'."
            )

        if self.algorithm not in {"nonparametric_f", "parametric_f"}:
            raise ValueError(
                "algorithm must be 'nonparametric_f' or 'parametric_f', "
                f"got {self.algorithm!r}."
            )


# ============================================================
# Explicit full-structure candidates
# ============================================================

FIRST_ORDER_ENERGY_ONLY = ModelCandidate(
    name="first_order_energy_only",
    system_order=1,
    interaction_type="energy",
    algorithm="nonparametric_f",
    include_environment=False,
    complexity=1,
    description="First-order energy-only model: dot_X = f_phi^E(X).",
)

FIRST_ORDER_ENERGY_F = ModelCandidate(
    name="first_order_energy_f",
    system_order=1,
    interaction_type="energy",
    algorithm="nonparametric_f",
    include_environment=True,
    complexity=2,
    description="First-order energy model with environment force: dot_X = f(X) + f_phi^E(X).",
)

SECOND_ORDER_ENERGY_ONLY = ModelCandidate(
    name="second_order_energy_only",
    system_order=2,
    interaction_type="energy",
    algorithm="nonparametric_f",
    include_environment=False,
    complexity=3,
    description="Second-order energy-only model without environment force.",
)

SECOND_ORDER_ALIGNMENT_ONLY = ModelCandidate(
    name="second_order_alignment_only",
    system_order=2,
    interaction_type="alignment",
    algorithm="nonparametric_f",
    include_environment=False,
    complexity=3,
    description="Second-order alignment-only model without environment force.",
)

SECOND_ORDER_ENERGY_F = ModelCandidate(
    name="second_order_energy_f",
    system_order=2,
    interaction_type="energy",
    algorithm="nonparametric_f",
    include_environment=True,
    complexity=4,
    description="Second-order energy model with environment force.",
)

SECOND_ORDER_ALIGNMENT_F = ModelCandidate(
    name="second_order_alignment_f",
    system_order=2,
    interaction_type="alignment",
    algorithm="nonparametric_f",
    include_environment=True,
    complexity=4,
    description="Second-order alignment model with environment force.",
)

SECOND_ORDER_ENERGY_ALIGNMENT_ONLY = ModelCandidate(
    name="second_order_energy_alignment_only",
    system_order=2,
    interaction_type="energy_alignment",
    algorithm="nonparametric_f",
    include_environment=False,
    complexity=5,
    description="Second-order energy-alignment model without environment force.",
)

SECOND_ORDER_ENERGY_ALIGNMENT_F = ModelCandidate(
    name="second_order_energy_alignment_f",
    system_order=2,
    interaction_type="energy_alignment",
    algorithm="nonparametric_f",
    include_environment=True,
    complexity=6,
    description="Second-order energy-alignment model with environment force.",
)



# ============================================================
# Candidate groups
# ============================================================

FULL_STRUCTURE_SELECTION_CANDIDATES = [
    FIRST_ORDER_ENERGY_ONLY,
    FIRST_ORDER_ENERGY_F,
    SECOND_ORDER_ENERGY_ONLY,
    SECOND_ORDER_ALIGNMENT_ONLY,
    SECOND_ORDER_ENERGY_F,
    SECOND_ORDER_ALIGNMENT_F,
    SECOND_ORDER_ENERGY_ALIGNMENT_ONLY,
    SECOND_ORDER_ENERGY_ALIGNMENT_F,
]

FIRST_ORDER_STRUCTURE_CANDIDATES = [
    FIRST_ORDER_ENERGY_ONLY,
    FIRST_ORDER_ENERGY_F,
]

SECOND_ORDER_STRUCTURE_CANDIDATES = [
    SECOND_ORDER_ENERGY_ONLY,
    SECOND_ORDER_ALIGNMENT_ONLY,
    SECOND_ORDER_ENERGY_F,
    SECOND_ORDER_ALIGNMENT_F,
    SECOND_ORDER_ENERGY_ALIGNMENT_ONLY,
    SECOND_ORDER_ENERGY_ALIGNMENT_F,
]



def validate_candidates(candidates: list[ModelCandidate]) -> None:
    names = set()

    for candidate in candidates:
        candidate.validate()

        if candidate.name in names:
            raise ValueError(f"Duplicate candidate name: {candidate.name!r}")

        names.add(candidate.name)


def get_candidates(mode: str) -> list[ModelCandidate]:
    """
    Return candidate list for a requested model-selection mode.

    Modes
    -----

    first_order_structure_selection:
        New behavior. First-order energy candidates with and without f.

    second_order_structure_selection:
        New behavior. Second-order candidates with and without f.

    full_structure_selection:
        New behavior. First-order and second-order candidates with and without f.
    """

    if mode == "first_order_structure_selection":
        candidates = FIRST_ORDER_STRUCTURE_CANDIDATES

    elif mode == "second_order_structure_selection":
        candidates = SECOND_ORDER_STRUCTURE_CANDIDATES

    elif mode == "full_structure_selection":
        candidates = FULL_STRUCTURE_SELECTION_CANDIDATES

    else:
        raise ValueError(
            "Unsupported model-selection mode "
            f"{mode!r}. Expected one of: "
            "'first_order_structure_selection', "
            "'second_order_structure_selection', "
            "'full_structure_selection'."
        )

    validate_candidates(candidates)
    return list(candidates)



# from ips_learning.model_selection.candidates import get_candidates

# candidates = get_candidates("full_structure_selection")

# for c in candidates:
#     print(
#         c.name,
#         c.system_order,
#         c.interaction_type,
#         c.include_environment,
#         c.complexity,
#     )