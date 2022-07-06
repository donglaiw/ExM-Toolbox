import math
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.interpolate import RBFInterpolator
from visualize import plot_pts


def filter_outliers(fpath_src: str, fpath_dst: str):
    src = np.loadtxt(fpath_src, delimiter=' ')
    dst = np.loadtxt(fpath_dst, delimiter=' ')

    x = src[:,0]
    y = src[:,1]
    z = src[:,2]

    new_src = list()
    for coordinate in dst:
        if(coordinate[0]<max(x) and coordinate[1]<max(y) and coordinate[3]<max(z)):
            new_src.append(coordinate)
    new_src = np.array(new_src)

    final_src = list()
    for coordinate in new_src:
        if(coordinate[0]>min(x) and coordinate[1]>min(y) and coordinate[2]>min(z)):
            final_src.append(coordinate)

    return final_src


def find_matching_points(point_cloud1,point_cloud2):
    temp1 = np.asarray([x for x in point_cloud1])
    temp1[:,0] = temp1[:,0] * 0.4/0.1625
    temp2 = np.asarray([x for x in point_cloud2])
    temp2[:,0] = temp2[:,0] * 0.4/0.1625
    print('total:',len(temp1))

    distance = cdist(temp1, temp2, 'euclidean')
    index = np.argmin(distance, axis = 1)

    pairs = [ {'point0':point_cloud1[i],'point1':point_cloud2[index[i]],'index':i} for i in range(len(temp1)) if distance[i,index[i]] < 8]
    for i, pair in enumerate(pairs):
        pair['index'] = i
        pair['valid'] = True
    print('pairs',len(pairs))
    return pairs


def generate_subpairs(pairs, n_pairs: int, replace=False):
    subpairs = np.random.choice(pairs, n_pairs, replace=replace)
    print('Subpairs:', len(subpairs))

    new_src = list()
    new_target = list()

    for pair in subpairs:
        new_src.append(elem['point0'])
        new_target.append(elem['point1'])

    new_src = np.array(new_src)
    new_target = np.array(new_target)


def rbf_interpolate(src: np.ndarray, target: np.ndarray, percentage):

    assert percentage <= 100, "invalid percentage value (between 0-100)"

    index_small = np.random.choice(range(len(target)),int(percentage*len(target)),replace = False)
    target_small = target[index_small,:]
    source_small = source[index_small,:]
    d_small = np.asarray(target_small-source_small)[:,0]
    rbf = RBFInterpolator(source_small, d_small, kernel = 'thin_plate_spline')

    difference = rbf(source)
    transformed = source
    transformed[:,0] += difference

    print(np.mean(abs(transformed[:,0]-target[:,0])))

    plot_pts(src, target, symbol1='circle', symbol2='cross')

