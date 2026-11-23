from ips_learning.systems.base import SecondOrderSystemBase, FirstOrderSystemBase
from ips_learning.systems.spp import SPPSystem
from ips_learning.systems.levi import LeviSystem
from ips_learning.systems.kuramoto import KuramotoSystem
from ips_learning.systems.sppcs import SPPCSSystem
from ips_learning.systems.cucker_smale import CuckerSmaleSystem
from ips_learning.systems.opinion_dynamics import OpinionDynamicsSystem

__all__ = [
    "SecondOrderSystemBase",
    "FirstOrderSystemBase",
    "SPPSystem",
    "LeviSystem",
    "KuramotoSystem",
    "SPPCSSystem",
    "CuckerSmaleSystem",
    "OpinionDynamicsSystem"

]
