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
        self.resolution = None
        self.parameter_type = None
        self.transform_type = None
        self.num_iteration = None

    def setResolution(self, resolution=None):
        # xyz-order
        if resolution is not None:
            self.resolution = resolution
        else:
            self.resolution = self.cfg.INTENSITY.RESOLUTION

    # setup
    def setTransformType(self, transform_type=None, num_iterations=None):
        if transform_type is not None:
            self.transform_type = transform_type
        else:
            self.transform_type = self.cfg.TRANSFORM_TYPE

        if num_iterations is not None:
            self.num_iterations = num_iterations
        else:
            self.num_iterations = self.cfg.INTENSITY.NUM_ITERATIONS

        self.parameter_map = self.createParameterMap(self.transform_type, self.num_iteration)
        self.elastix.setParameterMap(self.paramter_map)

    def updateParameterMap(self, parameter_map=None):
        if parameter_map is not None:
            self.parameter_type = parameter_map
        self.elastix.SetParameterMap(self.parameter_map)

    def getParameterMap(self):
        return self.parameter_map

    def readTransformMap(self, filename):
        return sitk.ReadParameterFile(filename)

    def writeTransformMap(self, filename, transform_type):
        return sitk.WriteParameterFile(transform_type, filename)

    def createParameterMap(self, transform_type=None, num_iterations=None):
        if transform_type is not None:
            self.transform_type = transform_type
        else:
            self.transform_type = self.cfg.TRANSFORM_TYPE

        if num_iterations is not None:
            self.num_iterations = num_iterations
        else:
            self.num_iterations = self.cfg.NUM_ITERATIONS

        if len(self.transform_type) == 1:
            parameter_type = sitk.GetDefaultParameterMap(self.transform_type[0])
            parameter_type['NumberOfSampleForExactGradient'] = ['5000']
            if num_iterations > 0:
                parameter_map['MaximumNumberOfIterations'] = [str(num_iterations)]
            else:
                parameter_map['MaximumNumberOfIterations'] = ['5000']
            parameter_map['MaximumNumberOfSamplingAttempts'] = ['100']
            parameter_map['FinalBSplineInterpolationOrder'] = ['1']
        else:
            parameter_map = sitk.VectorOfParameterMap()
            for trans in transform_type:
                parameter_map.append(self.createParameterMap(trans, num_iterations))
        return parameter_map

    # estimate and warp with transformation
    def convertSitkImage(self, vol_np, res_np):
        vol = sitk.GetImageFromArray(vol_np)
        vol.set_spacing(res_np)
        return vol

    def computeTransformMap(self, vol_fix, vol_move, res_fix=None, res_move=None, mask_fix=None, mask_move=None, log='file', log_path='.sitk/log'):
        # work with mask correctly
        # https://github.com/SuperElastix/SimpleElastix/issues/198
        # not enough samples in the mask
        if log == 'console':
            self.elastix.SetLogToConsole(True)
            self.elastix.SetLogToFile(False)
        elif log == 'file':
            self.elastix.SetLogToConsole(False)
            self.elastix.SetLogToFile(True)
            if os.path.isdir(log_path):
                self.elastix.SetOutputDirectory(log_path)
            else:
                os.mkdir(log_path)
                self.elastix.SetOutputDirectory(log_path)
        elif log is None:
            self.elastix.SetLogToFile(False)
            self.elastix.SetLogToConsole(False)
        else:
            raise KeyError(f"log must be console, file or None not {log}")

        if res_fix is None:
            res_fix = self.resolution
        if res_move is None:
            res_move = self.resolution

        # 2. load volume
        vol_fix = self.convertSitkImage(vol_fix, res_fix)
        self.elastix.SetFixedImage(vol_fix)
        if mask_fix is not None:
            mask_fix = self.convertSitkImage(mask_fix, res_fix)
            mask_fix.CopyInformation(vol_fix)
            self.elastix.SetFixedMask(mask_fix)

        # 3. compute transformation
        self.elastix.Execute()

        # 4. output transformation parameter
        return self.elastix.GetTransformParameterMap()[0]

    def warpVolume(self, vol_move, transform_map, res_move=None):
        if log == 'console':
            self.transformix.SetLogToConsole(True)
        else:
            self.transformix.LogToFileOn()
        if res_move is None:
            res_move = self.resolution
        self.transformix.SetTransformParameterMap(transform_map)
        self.transformix.SetMovingImage(self.convertSitkImage(vol_move, res_move))
        self.transformix.Execute()
        out = sitk.GetArrayFromImage(self.transformix.GetResultImage())
        return out

