import numpy as np
import tiffile
import SimpleITK as sitk
import h5py
import tqdm

outputs = []

f_vol_fix = '/mp/nas3/Margaret_mouse/round001_HIPV_ROI14_pp.tif'
f_vol_move = '/mp/nas3/Margaret_mouse/round002_HIPV_ROI14_pp.tif'
f_vol_out = '/home/ckapoor/margarets_data/round002_HIPV_ROI14_pp_warped.h5'

fixed_img = tiffile.imread(f_vol_fix)
moving_img = tiffile.imread(f_vol_move)


# find warping parameters
def find_params():
    for i in range(2, 3):

        m_transform_type = ['affine']
        m_channel_name = 'Lectin' # okay to be partial name
        m_resolution = [1.625,1.625, 4] # um: xyz. the image volume is in zyx-order

        # main function to run
        elastixImageFilter = sitk.ElastixImageFilter()

        # 1. set transformation parameters
        if len(m_transform_type) == 1:
            param_map = sitk.GetDefaultParameterMap(m_transform_type[0])
            param_map['NumberOfSamplesForExactGradient'] = ['100000']
            param_map['MaximumNumberOfIterations'] = ['10000']
            param_map['MaximumNumberOfSamplingAttempts'] = ['15']
            param_map['FinalBSplineInterpolationOrder'] = ['1']
            elastixImageFilter.SetParameterMap(param_map)
        else:
            parameterMapVector = sitk.VectorOfParameterMap()
            for trans in m_transform_type:
                parameterMapVector.append(sitk.GetDefaultParameterMap(trans))
            elastixImageFilter.SetParameterMap(parameterMapVector)

        # 2. load volume (for tif files)
        img_np = fixed_img[:, 3]
        print('vol-fix shape:', img_np.shape)
        img = sitk.GetImageFromArray(img_np)
        img.SetSpacing(m_resolution)
        elastixImageFilter.SetFixedImage(img)

        img_np = moving_img[:, 3]
        print('vol-move shape:', img_np.shape)
        img = sitk.GetImageFromArray(img_np)
        img.SetSpacing(m_resolution)
        elastixImageFilter.SetMovingImage(img)

        # 3. compute transformation
        elastixImageFilter.Execute()

        # 4. save output
        # save transformation param
        param_map = elastixImageFilter.GetTransformParameterMap()[0]
        sitk.WriteParameterFile(param_map, f_vol_out[:f_vol_out.rfind('.')] + '.txt')

        # save warped channels
        #channel_names = ND2Reader(f_vol_move).metadata['channels']
        channel_names = ['GFP', 'SynGAP', 'Bassoon', 'Lectin']

        if len(channel_names) == 1:
            # directly save
            sitk.WriteImage(sitk.Cast(elastixImageFilter.GetResultImage(), sitk.sitkUInt16), f_vol_out)

        else:
            fid = h5py.File(f_vol_out, 'w')
            ds = fid.create_dataset('spacing', [3], compression="gzip", dtype=int)
            ds[:] = np.array(m_resolution).astype(int)
            # image type: float -> np.uint16
            img_out = sitk.GetArrayFromImage(elastixImageFilter.GetResultImage()).astype(np.uint16)
            ds = fid.create_dataset([x for x in channel_names if m_channel_name in x][0], img_out.shape, compression="gzip", dtype=img_out.dtype)
            ds[:] = img_out

            # warp other channels
            transformixImageFilter = sitk.TransformixImageFilter()
            transformixImageFilter.SetTransformParameterMap(param_map)
            for channel_name in channel_names:
                if m_channel_name not in channel_name:
                    img_np = nd2ToVol(f_vol_move, channel_name)
                    print('vol 2:', channel_name, img_np.shape)
                    img = sitk.GetImageFromArray(img_np)
                    img.SetSpacing(m_resolution)
                    transformixImageFilter.SetMovingImage(img)
                    transformixImageFilter.Execute()
                    img_out = sitk.GetArrayFromImage(transformixImageFilter.GetResultImage()).astype(np.uint16)
                    ds = fid.create_dataset(channel_name, img_out.shape, compression="gzip", dtype=img_out.dtype)
                    ds[:] = img_out
            fid.close()

        outputs.append(f_vol_out)


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
        # not enough samples in the mask
        self.elastix.SetParameter("ImageSampler", "RandomSparseMask")
        self.elastix.SetLogToConsole(False)
        #self.elastix.SetLogToConsole(True)
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


# warp all channels
def warpAllChannels(output_path, vol_mov_path, sitkTile = sitkTile()):
    # set channel names
    channel_names = ['GFP', 'SynGAP', 'Bassoon', 'Lectin']

    # set sitk
    res = [1.625, 1.625, 4]
    sitkTile.setResolution(res)
    sitkTile.setTransformType(transform_type=['rigid'])

    for ind, channel in tqdm.tqdm(enumerate(channel_names)):
        tform = sitkTile.readTransformMap(tform_path)
        vol_mov = tiffile.imread(vol_mov_path)[: ,ind]

        # perform transform
        result = sitkTile.warpVolume(vol_mov, tform)

        # write transform
        with h5py.File(output_path, 'r+') as f:
            if channel in f.keys():
                del f[channel]
                f.create_dataset(channel, result.shape, compression='gzip',
                                 dtype=result.dtype, data=result)
            else:
                f.create_dataset(channel, result.shape, compression='gzip',
                                 dtype=result.dtype, data=result)

#warpAllChannels(output_path='/home/ckapoor/margarets_data/round002_HIPV_ROI14_pp_warped.h5',
#                vol_mov_path=f_vol_move)

# create output numpu array from h5 file
def createTIFF(f_vol_out, channel_names):
    '''
    args: f_vol_out     -> path of the output volume (h5 file)
          channel_names -> channel names in the volume

    returns a numpy image array
    '''
    output = np.zeros((4, 41, 2066, 2091)).astype(np.uint16)

    for i in range(len(channel_names)):
        channel_name = channel_names[i]
        output[i] = np.array(h5py.File(f_vol_out, 'r')[channel_name].astype(np.uint16))

    output = np.swapaxes(output, 0, 1)

    return output


# write output TIFF file
def writeTIFF(img_name, np_img, channel_names):
    '''
    args: img_name      -> name of TIFF file
          np_img        -> numpy array from h5 file
          channel_names -> channel names in volume

    writes a TIFF file
    '''
    tiffile.imwrite(img_name, np_img,
                    metadata={'Channel': {'Name': channel_names}})

