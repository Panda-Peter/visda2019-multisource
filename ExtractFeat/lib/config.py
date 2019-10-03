import os
import os.path as osp
import numpy as np

from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
__C.TRAIN = edict()

# Minibatch size
__C.TRAIN.BATCH_SIZE = 56

# Fix first two layers
__C.TRAIN.FIX_TWO = True

__C.TRAIN.FIX_BN = False

__C.TRAIN.FIX_ALL = False

# ---------------------------------------------------------------------------- #
# Inference ('test') options
# ---------------------------------------------------------------------------- #
__C.TEST = edict()

# Minibatch size
__C.TEST.BATCH_SIZE = 24


# ---------------------------------------------------------------------------- #
# Data loader options
# ---------------------------------------------------------------------------- #
__C.DATA_LOADER = edict()

# Data directory
__C.DATA_LOADER.NUM_WORKERS = 4

__C.DATA_LOADER.PIN_MEMORY = True

__C.DATA_LOADER.DROP_LAST = True

__C.DATA_LOADER.SHUFFLE = True

__C.DATA_LOADER.DATA_ROOT = '/export1/dataset/visda2019'

__C.DATA_LOADER.LIST = 'list'

__C.DATA_LOADER.FOLDER = 'sketch'

# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
__C.MODEL = edict()

__C.MODEL.CLASS_NUM = 345

__C.MODEL.NET = 'se_resnext50_32x4d'

__C.MODEL.IN_DIM = 2048

__C.MODEL.EMBED_DIM = 1000

# ---------------------------------------------------------------------------- #
# Augmentation options
# ---------------------------------------------------------------------------- #
__C.AUG = edict()

# Crop Size at testing
__C.AUG.TEST_CROP = [160, 160]

# Resize the input PIL Image to the given size  
# size (sequence or int) - Desired output size. If size is a sequence
# like (h, w), output size will be matched to this. If size is an int, 
# smaller edge of the image will be matched to this number. i.e, 
# if height > width, then image will be rescaled to (size * height / width, size)
__C.AUG.RESIZE = [176, 176] # None

# size (sequence or self.args.root_folderDesired output size of the crop. If size is an 
# int instead of seqself.args.root_folderike (h, w), a square crop (size, size) is made.
__C.AUG.RND_CROP = [160, 160] # None

# Vertically flip thself.args.root_folder PIL Image randomly with a given probability.
__C.AUG.V_FLIP = 0.0

# Horizontally flip self.args.root_folderen PIL Image randomly with a given probability.
__C.AUG.H_FLIP = 0.5

# degrees (sequence or float or int) - Range of degrees to select from. 
# If degrees is a number instead of sequence like (min, max), the range 
# of degrees will be (-degrees, +degrees)
__C.AUG.ROTATION = 0.0

# https://github.com/facebook/fb.resnet.torch/blob/master/datasets/imagenet.lua
# brightness (float) - How much to jitter brightness. brightness_factor 
# is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
__C.AUG.BRIGHTNESS = 0.0

# contrast (float) - How much to jitter contrast. contrast_factor 
# is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
__C.AUG.CONTRAST = 0.0

# saturation (float) - How much to jitter saturation. saturation_factor 
# is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
__C.AUG.SATURATION = 0.0

# hue (float) - How much to jitter hue. hue_factor 
# is chosen uniformly from [-hue, hue]. Should be >=0 and <= 0.5
__C.AUG.HUE = 0.0

# Custom_transforms
__C.AUG.SCALE_RATIOS = [1, 0.875, 0.75, 0.66]
__C.AUG.MAX_DISTORT = 1
__C.AUG.MULTI_CROP_SIZE = 0 # 0

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# A small number that's used many times
__C.EPS = 1e-14

# Root directory of project
__C.ROOT_DIR = os.getcwd()

# Logger name
__C.LOGGER_NAME = 'log'

# Image Mean
__C.MEAN = [0.485, 0.456, 0.406]

# Image std
__C.STD = [0.229, 0.224, 0.225]

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

def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value
