from .sitk_tile import sitkTile
import SimpleITK as sitk
from yacs.config import CfgNode

class alignBuild(sitkTile):

    def __init__(self, cfg: CfgNode):

        self.cfg = cfg
        self.transform_type = self.cfg.ALIGN.TRANSFORM_TYPE

        sitkTile.__init__(self, self.cfg)

    def createParameterMap(self, cfg = None, transform_type = None):
        if cfg is not None:
            self.cfg = cfg

        if transform_type is not None:
            self.transform_type = transform_type

        if len(self.cfg.ALIGN.TRANSFORM_TYPE) == 1:
            parameter_map = sitk.GetDefaultParameterMap(self.cfg.ALIGN.TRANSFORM_TYPE[0])
            parameter_map['NumberOfSamplesForExactGradient'] = [self.cfg.ALIGN.NumberOfSamplesForExactGradient]
            parameter_map['MaximumNumberOfIterations'] = [self.cfg.ALIGN.MaximumNumberOfIterations]
            parameter_map['MaximumNumberOfSamplingAttempts'] = [self.cfg.ALIGN.MaximumNumberOfSamplingAttempts]
            parameter_map['FinalBSplineInterpolationOrder'] = [self.cfg.ALIGN.FinalBSplineInterpolationOrder]
            #parameter_map['NumberOfResolutions'] = ['1']
        else:
            parameter_map = sitk.VectorOfParameterMap()
            for trans in self.transform_type:
                parameter_map.append(self.createParameterMap(trans))
        return parameter_map

    def buildSitkTile(self, cfg = None, transform_type = None):

        if cfg is not None:
            self.cfg = cfg
        
        if transform_type is not None:
            self.transform_type = transform_type

        self.setResolution()

        parameter_map = self.createParameterMap()

        self.elastix.SetParameterMap(parameter_map)
