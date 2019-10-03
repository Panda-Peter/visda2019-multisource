# --------------------------------------------------------
# Domain Adaptation
# Written by VARMS
# --------------------------------------------------------
from losses.cross_entropy import CrossEntropyLoss
from losses.general_entropy import GeneralEntropyLoss

__factory = {
    'CrossEntropy': CrossEntropyLoss,
    'GeneralEntropy': GeneralEntropyLoss,
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
