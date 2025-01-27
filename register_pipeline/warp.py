import SimpleITK as sitk
from tiffile import imwrite
from coarse import *
from utils import genFileName, getTformPath

def warpSingle(tform_path: str, f_vol_move:str, n_channel:int, sitkTile=sitkTile()):
    '''
    warp a single (specified) channel in the specified image
    '''
    res = [0.1625, 0.1625, 0.2500]
    sitkTile.setResolution(res)
    sitkTile.setTransformType(transform_type=['affine'])

    tform = sitkTile.readTransformMap(tform_path)
    vol_mov = tiffile.imread(f_vol_move)

    # perform transform
    result = sitkTile.warpVolume(vol_move[:, n_channel], tform)

    return result


def warpAll(fov_path: str, fovs: list, output_dir: str, tform_dir: str):
    '''
    warp all channels for specified fovs in a directory
    args:   fov_path -> '/mp/nas3/Margaret_mouse_new/2022.03_5xFAD/preprocessed'
            fovs -> fields of view requiring registration, for this case:
                    ['5xFAD-Cortex-ROI4', '5xFAD-Hippo-ROI1', '5xFAD-Hippo-ROI2',
                    '5xFAD-Hippo-ROI3', '5xFAD-Hippo-ROI4', 'WT-Cortex-ROI3']
            output_dir -> directory to store output tiff files
            tform_dir -> directory containing transform parameter files
    '''

    # sanity check on path
    if not os.path.exists(fov_path):
        print('non-existent directory location')
        return

    extra_fov = ['round001', 'round002', 'round003', 'round004', 'round006']
    warped_fov = ['round007', 'round008', 'round008', 'round009', 'round010', 'round011']

    # list of all moving images
    f_move = list()
    for file in os.listdir(fov_path):
        for fov in fovs:
            if fov in file:
                f_move.append(file)

    # remove extra moving images
    for fov in extra_fov:
        for img in f_move:
            if fov in img:
                f_move.remove(img)
    f_move.remove('5xFAD-Cortex-ROI4_round002_pp.tif')
    f_move = sorted(f_move)

    _, tform_paths = getTformPath(tfrom_dir)

    # warp and save image
    for i in range(len(f_move)):
        for fov in warped_fov:
            if fov in f_move[i]:
                n_channel = fov[-1]
                for channel in range(1, 5):
                    f_path_op = os.path.join(output_dir, f"{f_move[i][0:len(f_move[i])-7]}_ch0{channel}_warped.tif")
                    # check if image is already warped
                    if not os.path.exists(f_path_op):
                        warped_vol = warpSingle(tform_path=tform_paths[i],
                                                f_vol_move=os.path.join(fov_path, f_move[i]),
                                                n_channel=channel-1)
                        # save a 16-bit image
                        tiffile.imsave(os.path.join(output_dir, f_path_op), warped_vol.astype('uint16'))

