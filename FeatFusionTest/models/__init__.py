# --------------------------------------------------------
# Domain Adaptation
# Written by VARMS
# --------------------------------------------------------
from models.mlp import MLP
from models.bilinear import Bilinear

__factory = {
    'MLP': MLP,
    'Bilinear': Bilinear,
}

def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    """
    Create a loss instance.

    Parameters
    ----------
    name : str
        the name of loss function
    """
    if name not in __factory:
        raise KeyError("Unknown models:", name)
    return __factory[name](*args, **kwargs)