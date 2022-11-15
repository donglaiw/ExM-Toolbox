import os
import SimpleITK as sitk
from yacs.config import CfgNode


class sitkTile:
    # 1. estimate transformation between input volumes
    # 2. warp one volume with the transformation
    def __init__(self, cfg: CfgNode):
        self.elastix = sitk.ElastixImageFilter()
        self.transformix = sitk.TransformixImageFilter()
        self.cfg = cfg
        self.parameter_type = None
        self.transform_type = None

    # set resolution
    def setResolution(self, resolution=None):
        self.resolution = self.cfg.INTENSITY.RESOLUTION

    # setup
    def setTransformType(self, transform_type, num_iterations=-1):
        self.transform_type = self.cfg.INTENSITY.TRANSFORM_TYPE
        self.parameter_map = self.createParameterMap(transform_type, num_iterations)
        self.elastix.SetParameterMap(self.parameter_map)

    # update parameter map
    def updateParameterMap(self, parameter_map=None):
        if parameter_map is not None:
            self.parameter_map = parameter_map
        self.elastix.SetParameterMap(self.parameter_map)

    # get parameter map
    def getParameterMap(self):
        return self.parameter_map

    # read transformation map
    def readTransformMap(self, filename):
        return sitk.ReadParameterFile(filename)

    # write transformation map
    def writeTransformMap(self, filename, transform_type):
        return sitk.WriteParameterFile(transform_type, filename)

    # create a parameter map
    def createParameterMap(self, transform_type=None, num_iterations=-1):
        if transform_type is not None:
            self.transform_type = transform_type

        if len(self.transform_type) == 1:
            parameter_map = sitk.GetDefaultParameterMap(transform_type[0])
            parameter_map["NumberOfSampleForExactGradient"] = ["5000"]
            if num_iterations > 0:
                parameter_map["MaximumNumberOfIterations"] = [str(num_iterations)]
            else:
                parameter_map["MaximumNumberOfIterations"] = ["5000"]
            parameter_map["MaximumNumberOfSamplingAttempts"] = ["100"]
            parameter_map["FinalBSplineInterpolationOrder"] = ["1"]
        else:
            parameter_map = sitk.VectorOfParameterMap()
            for trans in transform_type:
                parameter_map.append(self.createParameterMap(trans, num_iterations))
        return parameter_map

    # estimate and warp with transformation
    def convertSitkImage(self, vol_np, res_np):
        vol = sitk.GetImageFromArray(vol_np)
        vol.SetSpacing(res_np)
        return vol

    def computeTransformMap(
        self,
        vol_fix,
        vol_move,
        res_fix=None,
        res_move=None,
        mask_fix=None,
        mask_move=None,
    ):
        # work with mask correctly
        # https://github.com/SuperElastix/SimpleElastix/issues/198
        # not enough samples in the mask
        self.elastix.SetParameter("ImageSampler", "RandomSparseMask")
        self.elastix.SetLogToConsole(False)

        if res_fix is None:
            res_fix = self.resolution
        if res_move is None:
            res_move = self.resolution

        # load volumes
        vol_fix = self.convertSitkImage(vol_fix, res_fix)
        self.elastix.SetFixedImage(vol_fix)
        vol_move = self.convertSitkImage(vol_move, res_move)
        self.elastix.SetMovingImage(vol_move)

        # image mask
        if mask_fix is not None:
            mask_fix = self.convertSitkImage(mask_fix, res_fix)
            mask_fix.CopyInformation(vol_fix)
            self.elastix.SetFixedMask(mask_fix)
        if mask_move is not None:
            mask_move = self.convertSitkImage(mask_move, res_move)
            mask_move.CopyInformation(vol_move)
            self.elastix.SetMovingMask(mask_move)

        # compute transformation
        self.elastix.Execute()

        # output transformation parameter
        return self.elastix.GetTransformParameterMap()[0]

    def warpVolume(self, vol_move, transform_map, res_move=None):
        self.transformix.SetLogToConsole(False)
        if res_move is None:
            res_move = self.resolution
        self.transformix.SetTransformParameterMap(transform_map)
        self.transformix.SetMovingImage(self.convertSitkImage(vol_move, res_move))
        self.transformix.Execute()
        out = sitk.GetArrayFromImage(self.transformix.GetResultImage())
        return out
