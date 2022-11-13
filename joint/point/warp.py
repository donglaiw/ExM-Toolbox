import numpy as np
from tifffile import imread, imwrite
from math import inf
from yacs.config import CfgNode
from skimage.transform import warp
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
    print("pairs", len(pairs))
    return pairs


def rbfInterpolate(corrPair: list, dim: int, cfg: CfgNode):
    """ """
    source, target = list(), list()

    # get corresponding points in fixed and moving cloud
    for pt in corrPair:
        target.append(pt["point0"])
        source.append(pt["point1"])
    source, target = np.array(source), np.array(target)

    target_index = np.empty((cfg.POINTS.NUM_POINTS))
    actual_error = np.mean(abs(source[:, dim] - target[:, dim]))
    min_error = inf

    for i in range(cfg.POINT.MAX_ITER):
        index_small = np.random.choice(
            range(len(target)), int(cfg.POINT.NUM_POINTS), replace=False
        )
        target_small = target[index_small, :]
        source_small = source[source_small, :]
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
            # print(mean_error)
            if abs(mean_error - actual_error) < cfg.POINT.TOLERANCE:
                if mean_error < min_error:
                    min_error = mean_error
                    target_index = index_small
        except:
            print(f"Singular matrix, skipping iteration...")
            continue

    if abs(min_error - actual_error) < cfg.POINT.TOLERANCE:
        print(f"Final error: {min_error}")
        return target_index


def warp(fix_cloud: np.ndarray, move_cloud: np.ndarray):
    print("<to-do>")
