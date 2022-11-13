import os
import numpy as np
import SimpleITK as sitk
from tiffile import imread, imsave
from .sitk_tile import sitkTile
from yacs.config import CfgNode


class Transform:
    # compute transformation map using a base channel
    # warp a single channel (for sanity checks) or an entire volume
    def __init__(self, cfg: CfgNode):
        self.cfg = cfg
        self.elastixImageFilter = sitk.ElastixImageFilter()
        self.resolution = None

    def computeMap(
        self,
        f_fixVol: str = None,
        f_moveVol: str = None,
        tformType: [str] = None,
        res: list = None,
        baseChannel: int = None,
    ):
        """
        Compute transform parameter map between a fixed and moving image volume

        Args:
                f_fixVol:    path to fixed volume image
                f_moveVol:   path to moving volume image
                tformType:   type of transformation computed (see Elastix manual for more options)
                baseChannel: base channel to use for registration

        Returns a transformation parameter text file
        """
        if f_fixVol is not None:
            fixVol = imread(f_fixVol)
        else:
            fixVol = imread(self.cfg.DATASET.VOL_FIX_PATH)

        if f_moveVol is not None:
            moveVol = imread(f_moveVol)
        else:
            moveVol = imread(self.cfg.DATASET.VOL_MOVE_PATH)

        if tformType is not None:
            self.tformType = tformType
        else:
            self.tformType = self.cfg.INTENSITY.TRANSFORM_TYPE

        if res is not None:
            self.res = res
        else:
            self.res = self.cfg.RESOLUTION

        if baseChannel is not None:
            self.baseChannel = baseChannel
        else:
            self.baseChannel = self.cfg.DATASET.BASE_CHANNEL
            assert (
                self.baseChannel > 0 and self.baseChannel <= fixVol.shape[0]
            ), "invalid base channel"

        baseFixVol = fixVol[:, baseChannel]
        baseMoveVol = moveVol[:, baseChannel]

        if len(tformType) == 1:
            paramMap = sitk.GetDefaultParameterMap(self.tformType[0])
            paramMap["NumberOfSamplesForExactGradient"] = [
                self.cfg.INTENSITY.NumberOfSamplesForExactGradient
            ]
            paramMap["MaximumNumberOfIterations"] = [
                self.cfg.INTENSITY.MaximumNumberOfIterations
            ]
            paramMap["MaximumNumberOfSamplingAttempts"] = [
                self.cfg.INTENSITY.MaximumNumberOfSamplingAttempts
            ]
            paramMap["FinalBSplineInterpolationOrder"] = [
                self.cfg.INTENSITY.FinalBSplineInterpolationOrder
            ]
            self.elastixImageFilter.SetParameterMap(paramMap)
        # fixed image
        print(f"Fixed volume shape: {baseFixVol.shape}")
        sitkFix = sitk.GetImageFromArray(baseFixVol)
        sitkFix.setSpacing(self.res)
        self.elastixImageFilter.SetFixedImage(sitkFix)
        # moving image
        print(f"Moving volume shape: {baseMoveVol.shape}")
        sitkMove = sitk.GetImageFromArray(baseMoveVol)
        sitk.setSpacing(self.res)
        self.elastixImageFilter.SetMovingImage(sitkMove)
        # compute transformation
        self.elastixImageFilter.Execute()
        paramMap = self.elastixImageFilter.GetTransformParameterMap()[0]
        if os.path.exists(self.cfg.DATASET.OUT_DIR):
            os.chdir(self.cfg.DATASET.OUT_DIR)
            sitk.WriteParameterFile(
                paramMap,
                self.cfg.DATASET.VOL_OUT[: self.cfg.DATASET.VOL_OUT.rfind(".")]
                + ".txt",
            )
        else:
            print(f"The path: {self.cfg.DATASET.OUT_DIR} does not exist")

    def warpChannel(
        self,
        f_moveVol: str = None,
        tformPath: str = "",
        res: list = None,
        tformType: list = None,
        channel: int = self.cfg.BASE_CHANNEL,
        sitkTile=sitkTile(),
    ):
        """
        Warp a single channel of the moving image volume with the given
        transformation map

        Args:
                f_moveVol: path to moving image volume
                tformPath: path to transformation parameters file
                res:       resolution of the warped channel
                tformType: type of transformation for warping image volume
                channel:   channel to be warped (base channel by default)

        Returns a warped volume
        """
        self.channel = channel
        if res is not None:
            self.res = res
        else:
            self.res = self.cfg.INTENSITY.RESOLUTION

        if tformType is not None:
            self.tformType = tformType
        else:
            self.tformType = self.cfg.INTENSITY.TRANSFORM_TYPE

        if f_moveVol is not None:
            self.moveVol = imread(f_moveVol)[:, self.channel]
        else:
            self.moveVol = imread(self.cfg.DATASET.VOL_MOVE_PATH)[:, self.channel]

        tform = self.sitkTile.readTransformMap(tformPath)
        # perform transformation
        warped = self.sitkTile.warpVolume(f_moveVol, tformPath)
        return warped

    def warpAll(
        self,
        f_moveVol: str,
        outputName: str,
        tformPath: str,
        outDir: str,
        res: list,
        sitkTile=sitkTile(),
    ):
        """
        Warp all channels of the moving image volume with the given
        transformation map

        Args:
                 f_moveVol:  path to moving image volume
                 outputName: base name of output volume
                 tformPath:  path to transformation parameters file
                 outDir:     output directory
                 res:        resolution of warped volume

        Returns a warping of all channels of the moving image saved in outDir
        """
        self.tformPath = tformPath
        self.f_moveVol = self.cfg.DATASET.VOL_FIX_PATH
        self.res = self.cfg.INTENSITY.RESOLUTION
        self.tformType = self.cfg.INTENSITY.TRANSFORM_TYPE
        self.tform = self.sitkTile.readTransformMap(tformPath)
        self.moveVol = imread(f_moveVol)
        # warp all channels
        for channel in range(self.moveVol.shape[0]):
            warped = self.sitkTile.warpVolume(self.f_moveVol[:, channel], self.tform)
            if os.path.exists(outDir):
                os.chdir(outDir)
                imsave(f"{outputName}_ch0{i+1}_warped.tif", warped.astype("uint16"))
            else:
                print(f"The path: {outDir} does not exist")
