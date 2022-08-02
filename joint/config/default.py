from yacs.config import CfgNode as CN
import SimpleITK as sitk

# copied from Jack's branch (thanks man :P)
# config definition

_C = CN()

# system
_C.SYSTEM = CN()

# filter
_C.FILTER = CN()
# thresholding, denoising and masking params



# intensity
_C.INTENSITY = CN()
# elastix params


# point
_C.POINT = CN()
# point-based regsitration params

# dataset
_C.DATASET = CN()
# dataset paths, fovs, base channels
_C.DATASET.VOL_FIX_PATH = '/path/to/vol_fix'
_C.DATASET.VOL_MOVE_PATH = '/path/to/vol_move'
_C.DATASET.BASE_CHANNEL = None


def get_cfg_defaults():
    r"""Get a yacs CfgNode object with default values for my_project."""
    # return a clone so that variables won't be altered.
    # this is for the "local variable use pattern"
    return _C.clone()

