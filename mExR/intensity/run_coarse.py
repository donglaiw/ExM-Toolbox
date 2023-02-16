import os
import numpy as np
from .mask import roiSlice
from yacs.config import CfgNode
from .sitk_tile import sitkTile
from tiffile import imread, imwrite
from .transformation import computeTform, warpAll


def saveMask(cfg: CfgNode):
    '''
    save masked image volume in the results directory
    '''
    print(f'Reading moving volume...')
    move_vol = imread(cfg.DATASET.VOL_MOVE_PATH)
    ref_channel = cfg.DATASET.MOVE_BASE_CHANNEL
    move_vol_anchor = move_vol[:,ref_channel] 
    print(f'Done!\n')
    
    print(f'Masking volume...')
    masked_vol = roiSlice(z_init=0, n_slices=move_vol_anchor.shape[0], im_channel=move_vol_anchor, min_idx=cfg.FILTER.MASK_INDEX, h_val=cfg.FILTER.FILTER_STRENGTH, thresh_val=cfg.FILTER.THRESH_LOWER, denoise=cfg.FILTER.DENOISE, mask=cfg.FILTER.MASK)
    print(f'Done!\n')
    
    file_name = cfg.DATASET.VOL_MOVE_PATH[cfg.DATASET.VOL_MOVE_PATH.rfind("/") + 1 :]
    fov = file_name[: file_name.find("_")]
    round_name = file_name[file_name.find("_") + 1 :]
    round_num = round_name[: round_name.find("_") :]

    if not os.path.exists('./results/masks/'):
        os.makedirs('./results/masks/')
    imwrite(f'./results/masks/{fov}_{round_num}_masked.tif', masked_vol)

    
def saveTform(cfg: CfgNode):
    # compute tform (and save tform)
    computeTform(cfg)


def warpVol(cfg: CfgNode):
    # 3. warp using tform (and save warps)
    if not os.path.exists('./results/coarse/'):
        os.makedirs('./results/coarse/')
        
    # regex parsing
    file_name = cfg.DATASET.VOL_MOVE_PATH[cfg.DATASET.VOL_MOVE_PATH.rfind("/") + 1 :]
    fov = file_name[: file_name.find("_")]
    round_name = file_name[file_name.find("_") + 1 :]
    round_num = round_name[: round_name.find("_") :]
    
    volMove = cfg.DATASET.VOL_MOVE_PATH
    outputName = fov + '_' + round_num
    
    param_name = file_name[file_name.rfind('/')+1:]
    param_file = param_name[:param_name.rfind('.')] + '.txt'
    tformPath = f'./results/transforms/{param_file}'
    warpAll(cfg=cfg, volMove=volMove, outputName=outputName, outDir='./results/coarse/', tformPath=tformPath, resolution=cfg.INTENSITY.RESOLUTION)


def runCoarse(cfg: CfgNode):
    # run coarse registration
    
    saveMask(cfg=cfg)
    saveTform(cfg=cfg)
    warpVol(cfg=cfg)