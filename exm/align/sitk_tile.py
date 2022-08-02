import SimpleITK as sitk
import numpy as np
from yacs.config import CfgNode
import cv2 as cv
import os
from scipy.ndimage import gaussian_filter
from skimage.morphology import convex_hull_image
from scipy.spatial.distance import cdist

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

    #### Setup/IO
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
    
    def convertSitkImage(self, vol_np, res_np = None):

        vol = sitk.GetImageFromArray(vol_np)

        if res_np is not None:
            vol.SetSpacing(res_np)
        else:
            vol.SetSpacing(self.resolution)

        return vol
    
    ### Debug compute 
    
    def mutual_information(self,img1,img2, bins = 20):
        """ Mutual information for joint histogram
        """
        # get histogram
        hgram, _, _ = np.histogram2d(img1.ravel(),img2.ravel(),bins=bins)
        # Convert bins counts to probability values
        pxy = hgram / float(np.sum(hgram))
        px = np.sum(pxy, axis=1) # marginal for x over y
        py = np.sum(pxy, axis=0) # marginal for y over x
        px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
        # Now we can do the calculation using the pxy, px_py 2D arrays
        nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
        return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    
    def computeMask(self, img, sigma = 2, thrsh = 200, kernel_size = 100):
    
        mask = np.zeros(img.shape)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(kernel_size,kernel_size))
        
        if len(img.shape) > 2:

            for ind, z in enumerate(img):
                # compute gaussian blur
                gaus = gaussian_filter(z, sigma = sigma)
                # binarize img
                binary = np.asarray(gaus > thrsh)
                #get convex hull
                hull = convex_hull_image(binary)
                #dilate
                dilation = cv.dilate(hull.astype('uint8'), kernel, iterations = 1)
    
                mask[ind, :, :] = dilation
        
        elif len(img.shape) == 2:
            
            # compute gaussian blur
            gaus = gaussian_filter(img, sigma = sigma)
            # binarize img
            binary = np.asarray(gaus > thrsh)
            #get convex hull
            hull = convex_hull_image(binary)
            #dilate
            dilation = cv.dilate(hull.astype('uint8'), kernel, iterations = 3)

            mask[:, :] = dilation
    
        return mask
    
    def computeMinNorm(self,point_cloud1, point_cloud2):
        
        temp1 = np.copy(point_cloud1)
        temp2 = np.copy(point_cloud2)
        # get distance btw point clouds
        distance = cdist(temp1, temp2, 'euclidean')
        # get point in pc2 w/ min distance to a given point in pc1
        min_dist = np.min(distance, axis = 0)
        # norm this vector
        norm = np.linalg.norm(min_dist)
        return norm
    
    def computeMinFlann(self,fix, mov, k = 1, flann_idx_kdtree = 0, flann_trees = 5, checks = 50,
                              sift_mask = None, flann_mask = False, ratio = .75):
        
        sift = cv.SIFT_create()
        
        # FLANN parameters
        index_params = dict(algorithm = flann_idx_kdtree, trees = flann_trees)
        if checks is not None:
            search_params = dict(checks=checks)   # or pass empty dictionary
        else:
            search_params = {}
            
        flann = cv.FlannBasedMatcher(index_params,search_params)
        
        kpf, desf = sift.detectAndCompute(fix.astype('uint8'), sift_mask)
        kpm, desm = sift.detectAndCompute(mov.astype('uint8'), sift_mask)
        matches = flann.knnMatch(desf,desm,k=k)
        
        
        if flann_mask is True:
            # Need to draw only good matches, so create a mask
            matchesMask = [[0,0] for i in range(len(matches))]
            # ratio test as per Lowe's paper
            for i,(m,n) in enumerate(matches):
                if m.distance < ratio*n.distance:
                    matchesMask[i]=[1,0]
                else:
                    pass

            dists = [pt[0].distance for ind, pt in enumerate(matches) if matchesMask[ind] == [1,0]]
            dists = np.asarray(dists)
            
            if dists.size > 0:
                norm = np.linalg.norm(dists)
                min_dist = np.min(dists)
            else:
                return None
            
        else:
            dists = [pt[0].distance for pt in matches]
            dists = np.asarray(dists)
            if dists.size > 0:
                norm = np.linalg.norm(dists)
                min_dist = np.min(dists) 
            else:
                return None
        
        return norm, min_dist
                     
    def computeMaxMI(self, fix, mov, min_array, z_min = 0, num_minima = 5):
                     
        min_sorted = np.sort(min_array)
        
        mi_result = {}
        
        for row, min_val in enumerate(min_sorted[:num_minima]):
        
            z_ind = np.argwhere(min_array == min_val)[0][0]+z_min
            mov_slice = mov[z_ind,:,:]
    
            mi = self.mutual_information(fix,mov_slice)
            mi_result[z_ind] = mi
        
        max_mi_ind = max(mi_result, key=mi_result.get)
        max_mi = max(mi_result.values())
        
        return max_mi, max_mi_ind
        
        
        
    #### Estimate and warp with transformation    
    
    def computeTransformMap(self, fix_dset, move_dset, 
                            res_fix=None, res_move=None, 
                            mask_fix=None, mask_move=None, 
                            log = 'file', log_path = './sitk_log/'):
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
            raise KeyError(f'log must be console, file, or None not {log}')

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
        
    def warpVolume(self, vol_move, transform_map, res_move=None, log = 'file', log_path = './transformix_log'):
        
        '''
        Warps a given volume using transformix and a given transformaiton matrix
        
        USE CASE:
        
        
        '''
        
        
        if log == 'console':
            self.transformix.SetLogToConsole(True)
            self.transformix.SetLogToFile(False)
        elif log == 'file':
            self.transformix.SetLogToConsole(False)
            self.transformix.SetLogToFile(True)
            if os.path.isdir(log_path):
                self.transformix.SetOutputDirectory(log_path)
            else:
                os.mkdir(log_path)
                self.transformix.SetOutputDirectory(log_path)
        elif log is None:
            self.transformix.SetLogToFile(False)
            self.transformix.SetLogToConsole(False)
        else:
            raise KeyError(f'log must be console, file, or None not {log}')
            
            
        self.transformix.SetTransformParameterMap(transform_map)
        self.transformix.SetMovingImage(self.convertSitkImage(vol_move, res_move))
        self.transformix.Execute()
        out = sitk.GetArrayFromImage(self.transformix.GetResultImage())
        return out
    
    def localToGlobalTform(self, 
                           fix, mov,
                           ROI_min_fix: list, ROI_max_fix: list,
                           ROI_min_mov: list, ROI_max_mov,
                           mask_fix=None, mask_move=None,
                           resolution: list = None):
        
        '''
        Applies a Euler3D tform to a cropped portion of a fixed and moving image hypothesized to have a known solution, the translation 
        parameters of that tform are then modified to correspond to the global image. Rotation parameters are not modified. 
        
        
        USE CASE:
        
        #define some ROI in fixed img
        ROI_min_fix = [50,700,100]
        ROI_max_fix = [100,1150,550]
        
        #define corresponding ROI in moving img
        ROI_min_mov = [89,700,100]
        ROI_max_mov = [139,1150,550]
        
        # align
        align_local = alignBuild(cfg)
        align_local.buildSitkTile()
        tform, result = align_local.localToGlobalTform(fix, mov, ROI_min_fix, ROI_max_fix, ROI_min_mov, ROI_max_mov)
        '''
        
        if resolution is None:
            resolution = self.resolution
        
        assert resolution[0] == resolution[1], "x and y size are not identical, resolution must be in x,y,z order"
        
        f_crop = fix[ROI_min_fix[0]:ROI_max_fix[0],ROI_min_fix[1]:ROI_max_fix[1],ROI_min_fix[2]:ROI_max_fix[2]]
        m_crop = mov[ROI_min_mov[0]:ROI_max_mov[0],ROI_min_mov[1]:ROI_max_mov[1],ROI_min_mov[2]:ROI_max_mov[2]]

        tform = self.computeTransformMap(f_crop,m_crop, mask_fix = mask_fix, mask_move =  mask_move)
        
        x_glob = str((ROI_max_mov[2] + ROI_min_mov[2]) / 2 * resolution[0])
        y_glob = str((ROI_max_mov[1] + ROI_min_mov[1]) / 2 * resolution[1])
        z_glob = str((ROI_max_mov[0] + ROI_min_mov[0]) / 2 * resolution[2])
        
        tform['CenterOfRotationPoint'] = (x_glob,y_glob,z_glob)
        tform['Size'] = (str(mov.shape[2]),str(mov.shape[1]),str(mov.shape[0]))
        
        params = list(tform['TransformParameters'])
        #set x
        params[-3] = str(float(tform['TransformParameters'][-3])-(resolution[0] * (ROI_min_fix[2]-ROI_min_mov[2])))
        #set y
        params[-2] = str(float(tform['TransformParameters'][-2])-(resolution[1]*(ROI_min_fix[1]-ROI_min_mov[1])))
        #set z
        params[-1] = str(float(tform['TransformParameters'][-1])-(resolution[2]*(ROI_min_fix[0]-ROI_min_mov[0])))
        
        tform['TransformParameters'] = tuple(params)
        
        global_result = self.warpVolume(mov, tform)
        
        return tform, global_result
    
    def computeTransformMapEvoOptim(self, fix, mov):
        
        '''
        Align two images using the OnePlusOneEvolutionaryOptimizer
        There are many parameters written to this method using the cfg dict that initilizes this object
        Make sure they are the actual ones you would like to use
        
        USER CASE:
        
        # some numpy arrays converted to sitkImage objs
        sitk_fix,sitk_mov = align.convertSitkImage(fix_img),align.convertSitkImage(mov_img)
        # make sure to cast as float32 or float64 or else function won't work
        sitk_fix,sitk_mov = sitk.Cast(sitk_fix, sitk.sitkFloat32),sitk.Cast(sitk_mov, sitk.sitkFloat32)
        
        tform = computeTransformMapEvoOptim(sitk_fix,sitk_mov)
        
        # now resample, not part of our function but necessary to get resulting img
        moving_resampled = sitk.Resample(
            sitk_fix,
            sitk_mov,
            final_transform,
            # below could also be cfg.ALIGN.Interpolator
            sitk.sitkLinear,
            0.0,
            sitk_mov.GetPixelID(),
        )
        result = sitk.GetArrayFromImage(moving_resampled)
        '''
        # init tform
        initial_transform = sitk.CenteredTransformInitializer(
            fix,
            mov,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        
        registration_method = sitk.ImageRegistrationMethod()
        
        # Similarity metric settings.
        registration_method.SetMetricAsMattesMutualInformation(self.cfg.ALIGN.NumberOfHistogramBins)
        registration_method.SetMetricSamplingStrategy(self.cfg.ALIGN.MetricSamplingStrategy)
        registration_method.SetMetricSamplingPercentage(self.cfg.ALIGN.MetricSamplingPercentage)
        
        # Set interpolator
        registration_method.SetInterpolator(self.cfg.ALIGN.Interpolator)
        
        # Optimizer settings.
        registration_method.SetOptimizerAsOnePlusOneEvolutionary(
            numberOfIterations = self.cfg.ALIGN.NumberOfIterations,
            epsilon = self.cfg.ALIGN.Epsilon,
            initialRadius = self.cfg.ALIGN.InitialRadius)
        registration_method.SetOptimizerScalesFromPhysicalShift()
        
        # Setup for the multi-resolution framework.
        registration_method.SetShrinkFactorsPerLevel(self.cfg.ALIGN.ShrinkFactors)
        # smoothing sigmas 
        registration_method.SetSmoothingSigmasPerLevel(self.cfg.ALIGN.SmoothingSigmas)
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        
        # Don't optimize in-place, we would possibly like to run this multiple times.
        # check this notebook to see if other method works better:
        # http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/63_Registration_Initialization.html
        # see if inplace param is important:
        # https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1ImageRegistrationMethod.html#a3492f3f1091a657c0c553bebad4bd3de
        registration_method.SetInitialTransform(initial_transform, inPlace=False)
        
        final_transform = registration_method.Execute(
            fix, mov
        )
        
        return final_transform
        
        
