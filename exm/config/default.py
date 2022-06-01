from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

# -----------------------------------------------------------------------------
# System
# -----------------------------------------------------------------------------
_C.SYSTEM = CN()

#_C.SYSTEM.NUM_GPUS = 4
_C.SYSTEM.NUM_CPUS = 4

# -----------------------------------------------------------------------------
# Align
# -----------------------------------------------------------------------------
_C.ALIGN = CN()

# Alignment set up parameters
_C.ALIGN.RESOLUION = [1.625,1.625,4.0]
_C.ALIGN.TRANSFORM_TYPE = ['rigid']
_C.ALIGN.NUM_ITERATION = -1

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()

# Dataset set paths, fov, and channel name
_C.DATASET.VOL_FIX_PATH = 'path/to/vol_fix'
_C.DATASET.VOL_MOVE_PATH = 'path/to/vol_move'
_C.DATASET.FOV = None
_C.DATASET.CHANNEL = '405 SD'
_C.DATASET.MASK_FIX_PATH = None

def get_cfg_defaults():
    r"""Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()



