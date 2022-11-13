from yacs.config import CfgNode as CN
import SimpleITK as sitk

# copied from Jack's branch (thanks man :P)
# config definition

_C = CN()

# system
_C.SYSTEM = CN()

# filter
_C.FILTER = CN()
_C.FILTER.FILTER_STRENGTH = 70  # denoising filter
_C.FILTER.THRESH_LOWER = 60  # adaptive thresholding lower bound
_C.FILTER.THRESH_UPPER = 255  # adaptive thresholding upper bound (good to leave)
_C.FILTER.MASK = False
_C.FILTER.MASK_INDEX = 6  # 'n' largest areas to leave from mask
_C.FILTER.DELTA = 230  # voxel extent of connected components (good to leave)


# intensity
_C.INTENSITY = CN()
# elastix params

# resolution in nm (or um?) -> need to check
_C.INTENSITY.RESOLUTION = [0.1625, 0.1625, 0.2500]  # used
# affine / rigid are good options
_C.INTENSITY.TRANSFORM_TYPE = ["affine"]  # used
# good to leave as it is
_C.INTENSITY.NumberOfSamplesForExactGradient = "100000"
_C.INTENSITY.MaximumNumberOfIterations = "10000"
_C.INTENSITY.MaximumNumberOfSamplingAttempts = "15"
_C.INTENSITY.FinalBSplineInterpolationOrder = "1"

# point
_C.POINT = CN()
# point-based registration params
_C.POINT.NUM_POINTS = 7  # number of control points for spline
_C.POINT.MAX_ITER = 50  # iterations to find a 'good' set of control points
_C.POINT.TOLERANCE = 1  # absolute cartesian error from ground truth points
_C.POINT.CHANNEL = 1  # channel to warp
# params for image cleaning (point registration)
# similar to intensity registration params
_C.POINT.FILTER_STRENGTH_FIX = 70
_C.POINT.FILTER_STRENGTH_MOVE = 70
_C.POINT.THRESH_LOWER_FIX = 20
_C.POINT.THRESH_LOWER_MOVE = 80

# dataset
_C.DATASET = CN()
# dataset paths, fovs, base channels
_C.DATASET.VOL_FIX_PATH = "/path/to/vol_fix"
_C.DATASET.VOL_MOVE_PATH = "/path/to/vol_move"
_C.DATASET.BASE_CHANNEL = 4  # GFAP + SMI + Lectin


def get_cfg_defaults():
    r"""Get a yacs CfgNode object with default values for my_project."""
    # return a clone so that variables won't be altered.
    # this is for the "local variable use pattern"
    return _C.clone()
