from .sitk_tile import sitkTile
import SimpleITK as sitk
from yacs.config import CfgNode


def buildSitkTile(cfg, sitkTile = sitkTile(), transform_type = None, num_iteration = None):

    sitkTile.setResolution()

    parameter_map = createParameterMap(cfg,transform_type,num_iteration)

    sitkTile.elastix.SetParameterMap(parameter_map)

    return sitkTile



def createParameterMap(cfg, transform_type = None, num_iteration = None):

    if transform_type is None:
        transform_type = cfg.ALIGN.TRANSFORM_TYPE
    
    if num_iteration is None:
        num_iteration = cfg.ALIGN.NUM_ITERATION

    if len(transform_type) == 1:
        parameter_map = sitk.GetDefaultParameterMap(transform_type[0])
        parameter_map['NumberOfSamplesForExactGradient'] = [cfg.ALIGN.NumberOfSamplesForExactGradient]
        if num_iteration > 0:
            parameter_map['MaximumNumberOfIterations'] = [str(num_iteration)]
        else:
            parameter_map['MaximumNumberOfIterations'] = [cfg.ALIGN.MaximumNumberOfIterations]
        parameter_map['MaximumNumberOfSamplingAttempts'] = [cfg.ALIGN.MaximumNumberOfSamplingAttempts]
        parameter_map['FinalBSplineInterpolationOrder'] = [cfg.ALIGN.FinalBSplineInterpolationOrder]
        #parameter_map['NumberOfResolutions'] = ['1']
    else:
        parameter_map = sitk.VectorOfParameterMap()
        for trans in transform_type:
            parameter_map.append(createParameterMap(trans, num_iteration))
    return parameter_map