import os
import os.path as osp
import numpy as np

from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

__C.TRAIN = edict()

__C.TRAIN.BATCH_SIZE = 32

# ---------------------------------------------------------------------------- #
# Inference ('test') options
# ---------------------------------------------------------------------------- #
__C.TEST = edict()

# Minibatch size
__C.TEST.BATCH_SIZE = 128

# ---------------------------------------------------------------------------- #
# Data loader options
# ---------------------------------------------------------------------------- #
__C.DATA_LOADER = edict()

# Data directory
__C.DATA_LOADER.NUM_WORKERS = 4

__C.DATA_LOADER.PIN_MEMORY = True

__C.DATA_LOADER.DROP_LAST = True

__C.DATA_LOADER.DATA_ROOT = '/export1/dataset/visda2019/pkl'

__C.DATA_LOADER.TARGET = 'sketch'

__C.DATA_LOADER.TEST_QUICKDRAW = True

__C.DATA_LOADER.TEST_INFOGRAPH = True

__C.DATA_LOADER.SRC_CNT_PCLS = 3
__C.DATA_LOADER.TRG_CNT_PCLS = 3

# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
__C.MODEL = edict()

__C.MODEL.CLASS_NUM = 345

__C.MODEL.NETS = ['MLP', 'MLP', 'Bilinear']

__C.MODEL.NAMES = ['incv2-MLP', 'se154-MLP', 'incv2+se154-Bilinear']

__C.MODEL.INPUTS = [[0], [0], [0, 1]]

__C.MODEL.NET_TYPE = ['inceptionresnetv2', 'pnasnet5large', 'senet154']

__C.MODEL.NET_DIM = [1536, 4320, 2048]

__C.MODEL.EMBED_DIM = [1000, 1000, 1000]

__C.MODEL.MIN_CLS_NUM = 5

__C.MODEL.MIN_CONF = 0.5

__C.MODEL.WEIGHTS = [1.0, 1.0, 1.0]

__C.MODEL.TEACHER_ALPHA = 0.99

__C.MODEL.CONFIDENCE_THRESH = 0.9


# ---------------------------------------------------------------------------- #
# Solver options
# ---------------------------------------------------------------------------- #
__C.SOLVER = edict()

# Base learning rate for the specified schedule
__C.SOLVER.LR = 0.01

# Solver type
__C.SOLVER.TYPE = 'sgd'

# Schedule type (see functions in utils.lr_policy for options)
# E.g., 'fix', 'step', 'steps_with_decay', ...
__C.SOLVER.LR_POLICY = 'step'  #'multistep'

# For 'step', the current LR is multiplied by SOLVER.GAMMA at each step
__C.SOLVER.GAMMA = 0.1

# Uniform step size for 'steps' policy, at epoch level
__C.SOLVER.STEP_SIZE = 5

# Maximum number of SGD iterations
__C.SOLVER.MAX_EPOCH = 50

# Momentum to use with SGD
__C.SOLVER.MOMENTUM = 0.9

# L2 regularization hyperparameter
__C.SOLVER.WEIGHT_DECAY = 0.0005
__C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

__C.SOLVER.BIAS_LR_FACTOR = 1

__C.SOLVER.DISPLAY = 20

# Iterations between snapshots, at epoch level
__C.SOLVER.SNAPSHOT_ITERS = 1

__C.SOLVER.TEST_INTERVAL = 1

__C.SOLVER.EPOCH_K = 5

# ---------------------------------------------------------------------------- #
# Losses options
# ---------------------------------------------------------------------------- #
__C.LOSSES = edict()

__C.LOSSES.CROSS_ENT_WEIGHT = 1.0

__C.LOSSES.TRG_GXENT_WEIGHT = 1.0
__C.LOSSES.GXENT_K = 0.5
__C.LOSSES.GXENT_Q = 0.7

__C.LOSSES.UNSUP_WEIGHT = 10.0

__C.LOSSES.SYMM_XENT_WEIGHT = 1.0
__C.LOSSES.ALPHA = 0.1
__C.LOSSES.BETA = 1.0
__C.LOSSES.A = -6.0


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# A small number that's used many times
__C.EPS = 1e-14

# Root directory of project
__C.ROOT_DIR = os.getcwd()

# Logger name
__C.LOGGER_NAME = 'log'


__C.SEED = -1.0

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    #for k, v in a.iteritems():
    for k, v in a.items(): # python3
        # a must specify keys that are in b
        if not k in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)