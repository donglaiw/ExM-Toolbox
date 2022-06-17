from .dataset_volume import datasetVolume


def getDataset (cfg, dataset_class = datasetVolume):

    fix_dset = dataset_class()
    fix_dset.loadVol(vol_fix_path = cfg.DATASET.VOL_FIX_PATH,
                                    iter_axes = cfg.DATASET.ITER_AXES,
                                    fov = cfg.DATASET.FOV,
                                    channel = cfg.DATASET.CHANNEL)
    
    move_dset = dataset_class()
    move_dset.loadVol(vol_fix_path = cfg.DATASET.VOL_MOVE_PATH,
                                    iter_axes = cfg.DATASET.ITER_AXES,
                                    fov = cfg.DATASET.FOV,
                                    channel = cfg.DATASET.CHANNEL)
    
    return fix_dset, move_dset