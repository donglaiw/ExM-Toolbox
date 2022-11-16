import os
import numpy as np
from yacs.config import CfgNode
from .interpolate import getAllIdx
from skimage.transform import warp
from tiffile import imread, imwrite
from scipy.interpolate import RBFInterpolator


def rbf_interpolate(
    source: np.ndarray, target: np.ndarray, cspace: np.ndarray, indices: list, dim: int
):
    """
    Compute interpolated vectors using a RBF and thin plate
    kernel, and return the deformed coordinate space mesh

    Arguments:
        source:  matrix of source points
        target:  matrix of target points
        cspace:  dense coordinate mesh of the image volume
        indices: indices of control points used to generate
                 interpolation vectors

    Returns a deformed image coordinate mesh
    """
    # source and targets for specified dimension
    target_small = target[indices, :]
    source_small = source[indices, :]
    d_small = np.asarray(target_small - source_small)[:, dim]

    # compute RBF interpolation vectors
    rbf = RBFInterpolator(source_small, d_small, kernel="thin_plate_spline")

    # compute error and interpolate control points
    difference = rbf(source)
    transformed = source
    transformed[:, dim] += difference

    print(
        f"Mean interpolation error (pixel space): {np.mean(abs(transformed[:,dim] - target[:,dim]))}"
    )
    print(f"Interpolating along dimension {dim}...")
    return rbf(cspace)


def warpImage(cfg: CfgNode):
    """
    Combine all helper functions to warp the image volume
    """
    file_name = cfg.DATASET.VOL_MOVE_PATH[cfg.DATASET.VOL_MOVE_PATH.rfind("/") + 1 :]
    fov = file_name[: file_name.find("_")]
    round_name = file_name[file_name.find("_") + 1 :]
    round_num = round_name[: round_name.find("_") :]

    # get indices
    indices = getAllIdx(cfg=cfg)

    # load corresponding point pairs
    corr_fix = np.loadtxt(
        f"./results/points/{fov}/{round_num}/round001_warped.txt", delimiter=" "
    )
    corr_move = np.loadtxt(
        f"./results/points/{fov}/{round_num}/{round_num}_warped.txt", delimiter=" "
    )

    # reading moving image volume
    print(f"Reading moving image volume...")
    vol_name = (
        f"./results/coarse/{fov}_{round_num}_ch0{cfg.POINT.TARGET_CHANNEL}_warped.tif"
    )
    f_vol_move = imread(vol_name)
    print(f"Done!")
    print(f"moving volume: {vol_name}")

    # generate pixel mesh
    zz, yy, xx = np.meshgrid(
        np.arange(f_vol_move.shape[0]),
        np.arange(f_vol_move.shape[1]),
        np.arange(f_vol_move.shape[2]),
        indexing="ij",
    )

    if not os.path.exists("./results/mesh.npy"):
        print(f"Generating coordinate mesh, this might take a while...")
        cspace = np.stack(
            [
                x
                for x in np.ndindex(
                    f_vol_move.shape[0], f_vol_move.shape[1], f_vol_move.shape[2]
                )
            ]
        )
        np.save("./results/mesh.npy", cspace, allow_pickle=True)
    else:
        print(f"Loading coordinate mesh...")
        with open("./results/mesh.npy", "rb") as f:
            cspace = np.load(f)
    print(f"Done!")

    # compute interpolation vectors
    dz = rbf_interpolate(
        source=corr_move, target=corr_fix, cspace=cspace, indices=indices["z"], dim=2
    )
    dy = rbf_interpolate(
        source=corr_move, target=corr_fix, cspace=cspace, indices=indices["y"], dim=1
    )
    dx = rbf_interpolate(
        source=corr_move, target=corr_fix, cspace=cspace, indices=indices["x"], dim=0
    )

    # reshape vectors
    dz = dz.reshape(f_vol_move.shape[0], f_vol_move.shape[1], f_vol_move.shape[2])
    dy = dy.reshape(f_vol_move.shape[0], f_vol_move.shape[1], f_vol_move.shape[2])
    dx = dx.reshape(f_vol_move.shape[0], f_vol_move.shape[1], f_vol_move.shape[2])

    # warp image
    print(f"Warping image volume...")
    move_warp = warp(
        imread(vol_name),
        np.array([zz + dz, yy + dy, xx + dx]),
        order=3,
        preserve_range=True,
    )
    print(f"Done!")

    # save image volume
    if not os.path.exists("./results/fine/"):
        os.makedirs("./results/fine/")
    imwrite(
        f"./results/fine/{fov}_{round_num}_ch0{cfg.POINT.TARGET_CHANNEL}_warped.tif",
        move_warp.astype("uint16"),
    )
