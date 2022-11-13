import os
import json
from point.warp import warp
from yacs.config import CfgNode
from config.utils import load_cfg
from point.pointcloud import PointCloud
from preprocess.preprocess import Filter


def main():
    """
    preprocess image, run joint image registration
    """
    # get config file, display it
    cfg = load_cfg()
    pretty_cfg = json.dumps(cfg, indent=4)
    print(f"Using the following configuration parameters:\n {pretty_cfg}")
    # run preprocessing
    print("Preprocessing image volumes...")
    preprocess_obj = Filter(cfg=cfg)
    print("Denoising image volume")
    clean = preprocess_obj.denoiseImg(
        f_volImg=cfg.DATASET.VOL_MOVE_PATH, channel=cfg.DATASET.BASE_CHANNEL
    )
    print("Thresholding image volume")
    clean = preprocess_obj.threshold(imgVol=clean)
    if cfg.FILTER.MASK:
        print("Masking image volume")
        clean = preprocess_obj.maskSmall(imgVol=clean)

    # intensity registration
    print("Performing intensity registration...")

    print("Computing tranformation paramters")

    print("Warping image volume")

    # point registration
    print("Performing point-based registration...")
    cloud = PointCloud(cfg=cfg)
    print("Generating fixed and moving point clouds")
    pc_fix = cloud.genPointClouds()
    pc_move = cloud.genPointClouds()


if __name__ == "__main__":
    main()
