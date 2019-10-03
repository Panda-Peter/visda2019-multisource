# --------------------------------------------------------
# Domain Adaptation
# Written by VARMS
# --------------------------------------------------------

from losses.cross_entropy import CrossEntropyLoss
from losses.symm_entropy import SymmEntropyLoss
from losses.mmd import MMDLoss
from losses.tpn_task import TpnTaskLoss
from losses.selfens import SelfEnsLoss
from losses.general_entropy import GeneralEntropyLoss
from losses.bsp import BSPLoss
from losses.cdane import CDANELoss

__factory = {
    'CrossEntropy': CrossEntropyLoss,
    'MMD': MMDLoss,
    'TpnTask': TpnTaskLoss,
    'SelfEns': SelfEnsLoss,
    'GeneralEntropy': GeneralEntropyLoss,
    'BSP': BSPLoss,
    'CDANE': CDANELoss,
    'SymmEntropy': SymmEntropyLoss,
}

def names():
    return sorted(__factory.keys())

def create(name):
    """
    Create a loss instance.

    Parameters
    ----------
    name : str
        the name of loss function
    """
    if name not in __factory:
        raise KeyError("Unknown loss:", name)
    return __factory[name]()