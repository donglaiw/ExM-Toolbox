import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from .sitk_tile import sitkTile
from yacs.config import CfgNode
from tiffile import imread, imsave


def computeTform(cfg: CfgNode):
    """
    Compute a transformation map for the anchoring
    channel
    """
    # basic settings
    m_transform_type = cfg.INTENSITY.TRANSFORM_TYPE
    m_resolution = cfg.INTENSITY.RESOLUTION

    elastixImageFilter = sitk.ElastixImageFilter()

    # set transformation parameters
    if len(m_transform_type) == 1:
        param_map = sitk.GetDefaultParameterMap(m_transform_type[0])
        param_map[
            "NumberOfSamplesForExactGradient"
        ] = cfg.INTENSITY.NumberOfSamplesForExactGradient
        param_map["MaximumNumberOfIterations"] = cfg.INTENSITY.MaximumNumberOfIterations
        param_map[
            "MaximumNumberOfSamplingAttempts"
        ] = cfg.INTENSITY.MaximumNumberOfSamplingAttempts
        param_map[
            "FinalBSplineInterpolationOrder"
        ] = cfg.INTENSITY.FinalBSplineInterpolationOrder
        elastixImageFilter.SetParameterMap(param_map)

    # set fixed image
    fixed_vol_np1 = imread(cfg.DATASET.VOL_FIX_PATH)[:, 0]
    print(f"Fixed volume shape: {fixed_vol_np1.shape}")
    fixed_vol = sitk.GetImageFromArray(fixed_vol_np1)
    fixed_vol.SetSpacing(m_resolution)
    elastixImageFilter.SetFixedImage(fixed_vol)

    # set moving image
    file_name = cfg.DATASET.VOL_MOVE_PATH[cfg.DATASET.VOL_MOVE_PATH.rfind("/") + 1 :]
    fov = file_name[: file_name.find("_")]
    round_name = file_name[file_name.find("_") + 1 :]
    round_num = round_name[: round_name.find("_") :]
    moving_vol_np1 = imread(f"./results/masks/{fov}_{round_num}_masked.tif")
    print(f"Moving volume shape: {moving_vol_np1.shape}")
    moving_vol = sitk.GetImageFromArray(moving_vol_np1)
    moving_vol.SetSpacing(m_resolution)
    elastixImageFilter.SetMovingImage(moving_vol)

    # compute transformation map
    elastixImageFilter.Execute()
    param_map = elastixImageFilter.GetTransformParameterMap()[0]
    param_name = file_name[file_name.rfind("/") + 1 :]
    param_file = param_name[: param_name.rfind(".")] + ".txt"

    if not os.path.exists("./results/transforms/"):
        os.makedirs("./results/transforms/")
    sitk.WriteParameterFile(
        param_map, os.path.join("./results/transforms/", param_file)
    )


def warpAll(
    cfg: CfgNode,
    volMove: str,
    outputName: str,
    outDir: str,
    tformPath: str,
    resolution: list,
):
    """
    Use transformation map to warp all channels of
    a particular FOV

    Arguments:
        volMove:    path to moving volume
        outputName: name of output volume
        outDir:     path of output directory
        tformPath:  path to transformation map
        resolution: resolution scale of image volume
    """
    sitktile = sitkTile(cfg)
    sitktile.setResolution(resolution)
    sitktile.setTransformType(transform_type=["affine"])
    tform = sitktile.readTransformMap(tformPath)
    vol_move = imread(volMove)
    for channel in tqdm(range(vol_move.shape[1])):
        result = sitktile.warpVolume(vol_move[:, channel], tform)
        imsave(
            f"{outDir}{outputName}_ch0{channel+1}_warped.tif", result.astype("uint16")
        )  # save as a 16-bit image
