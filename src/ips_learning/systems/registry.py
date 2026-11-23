from __future__ import annotations


LEGACY_SYSTEM_ALIASES = {
    "spp_swarm": "spp",
    "spp_rings": "spp",
}


def canonical_system_name(system_name: str) -> str:
    """
    Will remove later.
    Return the canonical mathematical-system name.

    Legacy aliases are retained only so historical artifact metadata
    can still be reconstructed.
    """
    return LEGACY_SYSTEM_ALIASES.get(system_name, system_name)


def system_params_from_metadata(metadata: dict) -> dict:
    """
    Support both metadata layouts.

    Old:
        metadata["system_params"]

    New:
        parameters stored directly in metadata
    """
    if "system_params" in metadata:
        return metadata["system_params"]

    return metadata


def get_system_factory(system_name: str):
    system_name = canonical_system_name(system_name)

    if system_name == "spp":
        from ips_learning.systems.spp import SPPSystem

        return SPPSystem

    if system_name == "levi":
        from ips_learning.systems.levi import LeviSystem

        return LeviSystem

    if system_name == "kuramoto":
        from ips_learning.systems.kuramoto import KuramotoSystem

        return KuramotoSystem

    raise ValueError(f"Unknown system_name: {system_name!r}")


def build_true_system_from_metadata(metadata: dict):
    raw_system_name = metadata.get("system_name")

    if raw_system_name is None:
        raise ValueError("metadata must contain 'system_name'.")

    system_name = canonical_system_name(raw_system_name)
    params = system_params_from_metadata(metadata)

    if system_name == "spp":
        from ips_learning.systems.spp import SPPSystem

        return SPPSystem(
            N=int(metadata["N"]),
            D=int(metadata["D"]),
            cr=params["cr"],
            lr=params["lr"],
            ca=params["ca"],
            la=params["la"],
            a=params["a"],
            b=params["b"],
            eps_r=params.get("eps_r", 1e-1),
        )

    if system_name == "levi":
        from ips_learning.systems.levi import LeviSystem

        return LeviSystem(
            N=int(metadata["N"]),
            D=int(metadata["D"]),
            u=params["u"],
            el=params["el"],
            I0=params["I0"],
            beta=params["beta"],
            lamda=params["lamda"],
            eps_r=params.get("eps_r", 1e-12),
        )

    if system_name == "kuramoto":
        from ips_learning.systems.kuramoto import KuramotoSystem

        return KuramotoSystem(
            N=int(metadata["N"]),
            D=int(metadata["D"]),
            K=params["K"],
            eps_r=params.get("eps_r", 1e-12),
            wrap_angles=params.get("wrap_angles", True),
        )

    raise ValueError(f"Unknown system_name: {raw_system_name!r}")


def true_kernel_from_metadata(metadata: dict):
    true_system = build_true_system_from_metadata(metadata)
    return true_system.kernel


def true_env_from_metadata(metadata: dict):
    true_system = build_true_system_from_metadata(metadata)
    return true_system.f_env