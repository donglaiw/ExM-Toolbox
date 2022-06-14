from .dataset_tile import datasetTile


def get_dataset (cfg, dataset_class = datasetTile()):

    dataset = dataset_class.loadVols(fov = cfg.DATASET.FOV, 
                                    vol_fix_path = cfg.DATASET.VOL_FIX_PATH,
                                    vol_mov_path = cfg.DATASET.VOL_MOV_PATH,
                                    channel = cfg.DATASET.CHANNEL)
    return dataset