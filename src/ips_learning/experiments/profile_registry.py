from __future__ import annotations


def build_dataset_profile(system_name: str, profile_name: str):
    if system_name == "spp":
        from ips_learning.experiments.generate_spp_data import build_spp_dataset_profile
        return build_spp_dataset_profile(profile_name)

    if system_name == "levi":
        from ips_learning.experiments.generate_levi_data import build_levi_dataset_profile
        return build_levi_dataset_profile(profile_name)
    
    if system_name == "kuramoto":
        from ips_learning.experiments.generate_kuramoto_data import build_kuramoto_dataset_profile
        return build_kuramoto_dataset_profile(profile_name)
    
    if system_name == "sppcs":
        from ips_learning.experiments.generate_sppcs_data import build_sppcs_dataset_profile
        return build_sppcs_dataset_profile(profile_name)

    if system_name == "cs":
        from ips_learning.experiments.generate_cs_data import build_cs_dataset_profile
        return build_cs_dataset_profile(profile_name)
    if system_name == "opinion":
        from ips_learning.experiments.generate_op_data import build_op_dataset_profile
        return build_op_dataset_profile(profile_name)

    raise ValueError(f"Unknown system_name: {system_name!r}")


def get_profile_common_config(system_name: str, profile_name: str) -> dict:
    profile = build_dataset_profile(system_name, profile_name)
    return dict(profile.common_config)