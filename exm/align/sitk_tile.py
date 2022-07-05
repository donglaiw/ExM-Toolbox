import SimpleITK as sitk
import numpy as np
from yacs.config import CfgNode
import cv2 as cv


class sitkTile:
    # 1. estimate transformation between input volumes
    # 2. warp one volume with the transformation
    def __init__(self, cfg: CfgNode):
        self.elastix = sitk.ElastixImageFilter()
        self.transformix = sitk.TransformixImageFilter()
        self.cfg = cfg
        self.resolution = None
        self.parameter_map = None
        self.transform_type = None
        self.num_iteration = None
    
    def setResolution(self, resolution= None):
        # xyz-order
        if resolution is not None:
            self.resolution = resolution
        else: 
            self.resolution = self.cfg.ALIGN.RESOLUTION

    #### Setup
    def setTransformType(self, transform_type = None, num_iteration = None):

        if transform_type is not None:
            self.transform_type = transform_type
        else:
            self.transform_type = self.cfg.ALIGN.TRANSFORM_TYPE

        if num_iteration is not None:
            self.num_iteration = num_iteration
        else:
            self.num_iteration = self.cfg.ALIGN.NUM_ITERATION

        self.parameter_map = self.createParameterMap(self.transform_type, self.num_iteration)
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
    
    def createParameterMap(self, transform_type = None, num_iteration = None):

        if transform_type is not None:
            self.transform_type = transform_type

        if num_iteration is not None:
            self.num_iteration = num_iteration

        if len(self.transform_type) == 1:
            parameter_map = sitk.GetDefaultParameterMap(self.transform_type[0])
            parameter_map['NumberOfSamplesForExactGradient'] = ['100000']
            if num_iteration > 0:
                parameter_map['MaximumNumberOfIterations'] = [str(self.num_iteration)]
            else:
                parameter_map['MaximumNumberOfIterations'] = ['10000']
            parameter_map['MaximumNumberOfSamplingAttempts'] = ['100']
            parameter_map['FinalBSplineInterpolationOrder'] = ['1']
            #parameter_map['NumberOfResolutions'] = ['1']

        else:
            parameter_map = sitk.VectorOfParameterMap()
            for trans in self.transform_type:
                parameter_map.append(self.createParameterMap(trans, self.num_iteration))

        return parameter_map

    #### Estimate and warp with transformation
    def convertSitkImage(self, vol_np, res_np = None):

        vol = sitk.GetImageFromArray(vol_np)

        if res_np is not None:
            vol.SetSpacing(res_np)
        else:
            vol.SetSpacing(self.resolution)

        return vol
    
    def compute_mask_cv(img, sigma = 2, thrsh = 200, kernel_size = 100):
    
        mask = np.zeros(img.shape)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(kernel_size,kernel_size))

        for ind, z in enumerate(img):
            # compute gaussian blur
            gaus = gaussian_filter(z, sigma = sigma)
            # binarize img
            binary = np.asarray(gaus > thrsh)
            #get convex hull
            hull = convex_hull_image(binary)
            #dilate
            dilation = cv.dilate(hull.astype('uint8'), kernel, iterations = 3)

            mask[ind, :, :] = dilation
    
        return mask
    
    def computeTransformMap(self, fix_dset, move_dset, res_fix=None, res_move=None, mask_fix=None, mask_move=None):
        # work with mask correctly
        # https://github.com/SuperElastix/SimpleElastix/issues/198
        # not enough samples in the mask
        self.elastix.SetLogToConsole(False)
        self.elastix.LogToFileOn()

        if res_fix is None:
            res_fix = self.resolution
        if res_move is None:
            res_move = self.resolution

        # 2. load volume
        # print('vol-fix shape:', vol_fix.shape)
        vol_fix = self.convertSitkImage(fix_dset, res_fix)
        self.elastix.SetFixedImage(vol_fix)

        if mask_fix is not None:
            self.elastix.SetParameter("ImageSampler", "RandomSparseMask")
            mask_fix = self.convertSitkImage(mask_fix, res_fix)
            mask_fix.CopyInformation(vol_fix)
            self.elastix.SetFixedMask(mask_fix)

        vol_move = self.convertSitkImage(move_dset, res_move)
        self.elastix.SetMovingImage(vol_move)

        if mask_move is not None:
            #self.elastix.SetParameter("ImageSampler", "RandomSparseMask")
            mask_move = self.convertSitkImage(mask_move, res_move)
            mask_move.CopyInformation(vol_move)
            self.elastix.SetMovingMask(mask_move)
            
        # 3. compute transformation
        self.elastix.Execute()

        # 4. output transformation parameter
        return self.elastix.GetTransformParameterMap()[0]
        
    def warpVolume(self, vol_move, transform_map, res_move=None):

        self.transformix.SetLogToConsole(False)
        self.transformix.LogToFileOn()

        if res_move is None:
            res_move = self.resolution
            
        self.transformix.SetTransformParameterMap(transform_map)
        self.transformix.SetMovingImage(self.convertSitkImage(vol_move, res_move))
        self.transformix.Execute()
        out = sitk.GetArrayFromImage(self.transformix.GetResultImage())
        return out
    
    def localToGlobalTform(self, fix, mov, resolution: list, ROI_min_fix: list, ROI_max_fix: list, ROI_min_mov: list, ROI_max_mov):
        
        assert resolution[0] == resolution[1], "x and y size are not identical, resolution must be in x,y,z order"
        
        f_crop = fix[ROI_min_fix[0]:ROI_max_fix[0],ROI_min_fix[1]:ROI_max_fix[1],ROI_min_fix[2]:ROI_max_fix[2]]
        m_crop = mov[ROI_min_mov[0]:ROI_max_mov[0],ROI_min_mov[1]:ROI_max_mov[1],ROI_min_mov[2]:ROI_max_mov[2]]

        tform = self.computeTransformMap(f_crop,m_crop)
        
        x_glob = str((ROI_max_mov[2] + ROI_min_mov[2]) / 2 * resolution[0])
        y_glob = str((ROI_max_mov[1] + ROI_min_mov[1]) / 2 * resolution[1])
        z_glob = str((ROI_max_mov[0] + ROI_min_mov[0]) / 2 * resolution[2])
        
        tform['CenterOfRotationPoint'] = (x_glob,y_glob,z_glob)
        tform['Size'] = (str(mov.shape[2]),str(mov.shape[1]),str(mov.shape[0]))
        
        params = list(tform['TransformParameters'])
        #set z
        params[-1] = str(float(tform['TransformParameters'][-1])-(4*(ROI_min_fix[0]-ROI_min_mov[0])))
        #set y
        params[-2] = str(float(tform['TransformParameters'][-2])-(1.625*(ROI_min_fix[1]-ROI_min_mov[1])))
        #set x
        params[-3] = str(float(tform['TransformParameters'][-3])-(1.625*(ROI_min_fix[2]-ROI_min_mov[2])))
        
        tform['TransformParameters'] = tuple(params)
        
        global_result = self.warpVolume(mov, tform)
        
        return tform, global_result
        