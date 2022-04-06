import SimpleITK as sitk

class sitkTile:
    # 1. estimate transformation between input volumes
    # 2. warp one volume with the transformation
    def __init__(self):
        self.elastix = sitk.ElastixImageFilter()
        self.transformix = sitk.TransformixImageFilter()
        self.parameter_map = None
        self.transform_type = None
    
    def setResolution(self, resolution):
        # xyz-order
        self.resolution = resolution

    #### Setup
    def setTransformType(self, transform_type, num_iteration = -1):
        self.transform_type = transform_type
        self.parameter_map = self.createParameterMap(transform_type, num_iteration)
        self.elastix.SetParameterMap(self.parameter_map)
    
    def updateParameterMap(self, parameter_map=None):
        if parameter_map is not None:
            self.parameter_map = parameter_map
        self.elastix.SetParameterMap(self.parameter_map)

    def getParameterMap(self):
        return self.parameter_map

    def readTransformMap(self, filename):
        return sitk.ReadParameterFile(filename)

    def writeTransformMap(self, filename, transform_map):
        return sitk.WriteParameterFile(transform_map, filename)
    
    def createParameterMap(self, transform_type = None, num_iteration = -1):
        if transform_type is None:
            transform_type = self.transform_type
        if len(transform_type) == 1:
            parameter_map = sitk.GetDefaultParameterMap(transform_type[0])
            parameter_map['NumberOfSamplesForExactGradient'] = ['5000']
            if num_iteration > 0:
                parameter_map['MaximumNumberOfIterations'] = [str(num_iteration)]
            else:
                parameter_map['MaximumNumberOfIterations'] = ['5000']
            parameter_map['MaximumNumberOfSamplingAttempts'] = ['100']
            parameter_map['FinalBSplineInterpolationOrder'] = ['1']            
        else:
            parameter_map = sitk.VectorOfParameterMap()
            for trans in transform_type:
                parameter_map.append(self.createParameterMap(trans, num_iteration))
        return parameter_map

    #### Estimate and warp with transformation
    def convertSitkImage(self, vol_np, res_np):
        vol = sitk.GetImageFromArray(vol_np)
        vol.SetSpacing(res_np)
        return vol
    
    def computeTransformMap(self, vol_fix, vol_move, res_fix=None, res_move=None, mask_fix=None, mask_move=None):
        # work with mask correctly
        # https://github.com/SuperElastix/SimpleElastix/issues/198
        # self.elastix.SetParameter("ImageSampler", "RandomSparseMask")
        self.elastix.SetLogToConsole(False)
        if res_fix is None:
            res_fix = self.resolution
        if res_move is None:
            res_move = self.resolution
        # 2. load volume
        # print('vol-fix shape:', vol_fix.shape)
        vol_fix = self.convertSitkImage(vol_fix, res_fix)
        self.elastix.SetFixedImage(vol_fix)
        if mask_fix is not None:
            mask_fix = self.convertSitkImage(mask_fix, res_fix)
            mask_fix.CopyInformation(vol_fix)
            self.elastix.SetFixedMask(mask_fix)

        # print('vol-move shape:', vol_move.shape)
        vol_move = self.convertSitkImage(vol_move, res_move)
        self.elastix.SetMovingImage(vol_move)
        if mask_move is not None:
            mask_move = self.convertSitkImage(mask_move, res_move)
            mask_move.CopyInformation(vol_move)
            self.elastix.SetMovingMask(mask_move)
            
        # 3. compute transformation
        self.elastix.Execute()

        # 4. output transformation parameter
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
