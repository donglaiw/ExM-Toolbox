from platform import platform
import itk
import numpy as np
import pyopencl as cl

class sitkTile:
    # 1. estimate transformation between input volumes
    # 2. warp one volume with the transformation
    def __init__(self):
        self.elastix = itk.ElastixRegistrationMethod.New()
        self.transformix = itk.TransformixFilter.New()
        self.parameter_map = None
        self.transform_type = None
    
    def setResolution(self, resolution):
        # xyz-order
        VectorType = itk.Vector[itk.F, len(resolution)]
        res_vec = VectorType()
        for ind, res in enumerate(resolution):
            res_vec[ind] = res
    
        self.resolution = res_vec

    #### Setup
    #def setTransformType(self, transform_type, num_iteration = -1, OpenCL = False):
    #    self.transform_type = transform_type
    
    def updateParameterMap(self, parameter_map=None):
        if parameter_map is not None:
            self.parameter_map = parameter_map
        self.elastix.SetParameterMap(self.parameter_map)

    def getParameterMap(self):
        return self.parameter_map
    
    def setParamterMap(self):
        self.parameter_map = self.createParameterMap(transform_type, num_iteration, OpenCL)
        self.elastix.SetParameterObject(self.parameter_map)

    def readTransformMap(self, filename):
        return itk.ReadParameterFile(filename)

    def writeTransformMap(self, filename, transform_map):
        return itk.WriteParameterFile(transform_map, filename)
    
    def createParameterMap(self, transform_type = None, num_iteration = -1, OpenCL = False):
        if transform_type is None:
            transform_type = self.transform_type
        if len(transform_type) == 1:
            parameter_object = itk.ParameterObject.New()
            parameter_map = parameter_object.GetDefaultParameterMap(transform_type[0])
            parameter_map['NumberOfSamplesForExactGradient'] = ['100000']
            if num_iteration > 0:
                parameter_map['MaximumNumberOfIterations'] = [str(num_iteration)]
            else:
                parameter_map['MaximumNumberOfIterations'] = ['10000']
            parameter_map['MaximumNumberOfSamplingAttempts'] = ['100']
            parameter_map['FinalBSplineInterpolationOrder'] = ['1']
            #parameter_map['NumberOfResolutions'] = ['1']
        else:
            parameter_map = itk.VectorOfParameterMap()
            for trans in transform_type:
                parameter_map.append(self.createParameterMap(trans, num_iteration))
        if OpenCL == True:
            #set id
            platforms = str(cl.get_platforms()[0])
            parameter_map['OpenCLDeviceID'] = platforms
            #set resampler
            parameter_map['Resampler'] = ["OpenCLResampler"]
            parameter_map['OpenCLResamplerUseOpenCL'] = ["true"]
            #set pyramids
            parameter_map['FixedImagePyramid'] = ["OpenCLFixedGenericImagePyramid"]
            parameter_map['OpenCLFixedGenericImagePyramidUseOpenCL'] = ["true"]

            parameter_map['OpenCLMovingGenericImagePyramidUseOpenCL'] = ["true"]
            parameter_map['MovingImagePyramid'] = ["OpenCLMovingGenericImagePyramid"]
            
            self.parameter_map = parameter_map

        return self.parameter_map

    #### Estimate and warp with transformation
    def convertSitkImage(self, vol_np, res):
        vol = itk.GetImageFromArray(vol_np)
        vol.SetSpacing(res)
        return vol
    
    def computeMask(vol, percentile = 60):
        thrsh = np.percentile(vol, percentile)
        mask = np.asarray(vol > thrsh, 'uint8')
        return mask
    
    def computeTransformMap(self, vol_fix, vol_move, res_fix=None, res_move=None, mask_fix=None, mask_move=None):
        # work with mask correctly
        # https://github.com/SuperElastix/SimpleElastix/issues/198
        # not enough samples in the mask
        #self.elastix.SetLogToConsole(False)
        
        # 1. load imgs 
        if res_fix is None:
            res_fix = self.resolution
        if res_move is None:
            res_move = self.resolution
            
        img_fix = self.convertSitkImage(vol_fix, res_fix)
        img_move = self.convertSitkImage(vol_move, res_move)
        
        # 2. set elastix obj
        self.elastix = itk.ElastixRegistrationMethod.New(img_fix,img_move)
        self.elastix.SetParameterObject(self.parameter_map)
        self.elastix.SetLogToConsole(True)

        # 3. set masks
        if mask_fix is not None:
            self.elastix.SetParameter("ImageSampler", "RandomSparseMask")
            mask_fix = self.convertSitkImage(mask_fix, res_fix)
            mask_fix.CopyInformation(vol_fix)
            self.elastix.SetFixedMask(mask_fix)

        if mask_move is not None:
            #self.elastix.SetParameter("ImageSampler", "RandomSparseMask")
            mask_move = self.convertSitkImage(mask_move, res_move)
            mask_move.CopyInformation(vol_move)
            self.elastix.SetMovingMask(mask_move)
            
        # 3. compute transformation
        self.elastix.UpdateLargestPossibleRegion()
        
        # 4. output transformation parameter
        return self.elastix.GetTransformParameterMap()[0]
        
    def warpVolume(self, vol_move, transform_map, res_move=None):
        self.transformix.SetLogToConsole(True)
        if res_move is None:
            res_move = self.resolution
            
        self.transformix.SetTransformParameterMap(transform_map)
        self.transformix.SetMovingImage(self.convertSitkImage(vol_move, res_move))
        self.transformix.UpdateLargestPossibleRegion()
        
        out = sitk.GetArrayFromImage(self.transformix.GetResultImage())
        return out