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
_C.FILTER.FILTER_STRENGTH = 70
_C.FILTER.THRESH_LOWER = 60
_C.FILTER.THRESH_UPPER = 255
_C.FILTER.MASK_INDEX = 6
_C.FILTER.DELTA = 230
_C.FILTER.DENOISE = False
_C.FILTER.MASK = False

# intensity
_C.INTENSITY = CN()
# elastix params
_C.INTENSITY.RESOLUTION = [0.1625, 0.1625, 0.2500]
_C.INTENSITY.TRANSFORM_TYPE = ["affine"]
_C.INTENSITY.NumberOfSamplesForExactGradient = ["100000"]
_C.INTENSITY.MaximumNumberOfIterations = ["10000"]
_C.INTENSITY.MaximumNumberOfSamplingAttempts = ["15"]
_C.INTENSITY.FinalBSplineInterpolationOrder = ["1"]

# point
_C.POINT = CN()
# point-based registration params
_C.POINT.NUM_POINTS = 20
_C.POINT.MAX_ITER = 1500
_C.POINT.CHANNEL = 2
_C.POINT.NN_DIST = 6
# params for image cleaning (point registration)
_C.POINT.FILTER_STRENGTH_FIX = 50
_C.POINT.FILTER_STRENGTH_MOVE = 80
_C.POINT.THRESH_LOWER_FIX = 70
_C.POINT.THRESH_LOWER_MOVE = 50
_C.POINT.TARGET_CHANNEL = 1

# dataset
_C.DATASET = CN()
# dataset paths, fovs, base channels
_C.DATASET.VOL_FIX_PATH = (
    "/mp/nas3/Margaret_mouse_new/2022.08_synapses/ROI4_round001_pp.tif"
)
_C.DATASET.VOL_MOVE_PATH = ('/mp/nas3/Margaret_mouse_new/2022.08_synapses/ROI4_round0010_pp.tif')
_C.DATASET.FIX_BASE_CHANNEL = 3
_C.DATASET.MOVE_BASE_CHANNEL = 0


def get_cfg_defaults():
    r"""Get a yacs CfgNode object with default values for my_project."""
    # return a clone so that variables won't be altered.
    # this is for the "local variable use pattern"
    return _C.clone()
