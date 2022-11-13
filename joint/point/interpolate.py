import numpy as np
from math import inf
from yacs.config import CfgNode
from .pointcloud import PointCloud
from scipy.spatial.distance import cdist
from scipy.interpolate import RBFInterpolator


def corrPts(cloud1: np.ndarray, cloud2: np.ndarray, cfg: CfgNode) -> list:
    """
    Find corresponding point pairs between cloud1
    and cloud2 within a 1st nearest neighbor distance

    Args:
            cloud1: point cloud 1
            cloud2: point cloud 2
            cfg: configuration file

    Returns a list of corresponding point pairs
    """

    # scale points (z, y coordinates already scaled)
    temp1 = np.asarray([x for x in cloud1])
    temp1[:, 0] = (
        temp1[:, 0] * cfg.INTENSITY.RESOLUTION[2] / cfg.INTENSITY.RESOLUTION[0]
    )
    temp2 = np.asarray([x for x in cloud2])
    temp2[:, 0] = (
        temp2[:, 0] * cfg.INTENSITY.RESOLUTION[2] / cfg.INTENSITY.RESOLUTION[0]
    )
    print("total:", len(temp1))

    distance = cdist(temp1, temp2, "euclidean")
    index = np.argmin(distance, axis=1)

    pairs = [
        {"point0": cloud1[i], "point1": cloud2[index[i]], "index": i}
        for i in range(len(temp1))
        if distance[i, index[i]] < cfg.POINT.NN_DIST
    ]
    for i, pair in enumerate(pairs):
        pair["index"] = i
        pair["valid"] = True
    print("pairs:", len(pairs))
    return pairs


def getPointIdx(corrPair: list, dim: int, num_pts: int, cfg: CfgNode):
    """
    a random selection of points from the corresponding point
    pair list, and returning the set of point indices with
    minimum error after interpolation

    Arguments:
        corrPair: list of corresponding point pairs
        dim:      [z, y, x] dimensions for point indices
        cfg:      configuration file

    returns a list of point indices for the specified dimension
    """
    source, target = list(), list()

    # get corresponding points in fixed and moving cloud
    for pt in corrPair:
        target.append(pt["point0"])
        source.append(pt["point1"])
    source, target = np.array(source), np.array(target)

    target_index = np.empty((num_pts))
    actual_error = np.mean(abs(source[:, dim] - target[:, dim]))
    min_error = inf

    for i in range(cfg.POINT.MAX_ITER):
        index_small = np.random.choice(range(len(target)), int(num_pts), replace=False)
        target_small = target[index_small, :]
        source_small = source[index_small, :]
        d_small = np.asarray(target_small - source_small)[:, dim]
        try:
            rbf = RBFInterpolator(
                source_small, d_small, kernel="thin_plate_spline", degree=1
            )
            difference = 0  # ensure that differences are not accumulated
            difference = rbf(source)
            transformed = source
            transformed[:, dim] += difference

            mean_error = np.mean(abs(transformed[:, dim] - target[:, dim]))
            if abs(mean_error - actual_error) < 0.5:
                if mean_error < min_error:
                    min_error = mean_error
                    target_index = index_small
        except:
            continue

    if abs(min_error - actual_error) < 0.5:
        # print(f"Final error: {min_error}")
        if min_error is not None:
            return min_error, target_index


def getAllIdx(cfg: CfgNode) -> dict:
    # generate point clouds
    print(f"Generating point clouds...")
    pc = PointCloud(cfg)
    fixed_cloud, moving_cloud = pc.genPointClouds()
    print(f"Done!")

    # generate corresponding pairs
    print(f"Generating corresponding point pairs...")
    pairs = corrPts(cloud1=fixed_cloud, cloud2=moving_cloud, cfg=cfg)
    print(f"Done!")

    # get point indices for every dimension
    final_indices = dict()

    for dim in range(2, -1, -1):
        idx_dict = dict()
        for num_pts in range(5, cfg.POINT.NUM_POINTS):
            try:
                error, idx = getPointIdx(
                    corrPair=pairs, dim=dim, num_pts=num_pts, cfg=cfg
                )
            except:
                continue
            if error is not None:
                idx_dict[error] = idx

        # select point indices with minimum error
        if dim == 2:
            final_indices["z"] = idx_dict[sorted(idx_dict.keys())[0]]
        if dim == 1:
            final_indices["y"] = idx_dict[sorted(idx_dict.keys())[0]]
        if dim == 0:
            final_indices["x"] = idx_dict[sorted(idx_dict.keys())[0]]

    return final_indices
