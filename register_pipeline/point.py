import cc3d
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import SimpleITK as sitk
from tiffile import imread, imwrite
from skimage.measure import label, regionprops
from scipy.spatial.distance import cdist


class PointCloud:
    '''
    compute and register point clouds
    use interpolation to warp image volume
    '''
    def __init__(self):
        self.fix = fix
        self.move = move

    def clean(self, f_path: str, h: int, threshold: int, channel=1):
        '''
        denoise and threshold image to enhance puncta in the image
        args:   f_path    -> path to tiffile
                h         -> denoising parameter
                threshold -> lower bound thresholding parameter
                channel   -> defaults to 1, used only for images having multiple channels

        returns a numpy array of cleaned image
        '''
        img = imread(f_path)
        clean_img = list()
        if len(img.shape) == 4:
            im_channel = img[:, channel-1]
        elif len(img.shape) == 3:
            im_channel = img
        else:
            print('img must have either 3 or 4 dimensions')
            return
        for i in range(im_channel.shape[0]):
            denoised = cv.fastNlMeansDenoising(im_channel[i, :, :, ].astype('uint8'), h=h,
                                               templateWindowSize=7, searchWindowSize=21)
            (T, thresh) = cv.threshold(denoised, thresh=threshold, maxval=255,
                                       type=cv.THRESH_BINARY_INV)
            thresh = 255 - thresh
            clean_img.append(thresh)
        return np.array(clean_img)

    def segment(self, img: np.ndarray, delta: int):
        '''
        perform instance segmentation followed by stitching of 3D
        connected components
        args:   img   -> numpy image array
                delta -> neighbour voxel values <= delta are considered the
                         same component

        returns a segmented image with stitched 3D components
        '''
        seg_img = list()
        # segment image
        for i in range(0, img.shape[0]):
            im_slice = img[i, :, :,]
            seg = label(im_slice, background=0)
            seg_img.append(seg)
        seg_img = np.array(seg_img)
        # return 3D stitched components
        return cc3d.connected_components(seg_img, delta=delta)

    def genPoints(self, img: np.ndarray):
        '''
        generate a Nx3 points array constructed from the centroids
        of the segment masks
        '''
        props = regionprops(img)
        pts = np.zeros(shape=(len(props), 3))
        for i in range(len(props)):
            pts[i] = np.asarray(getattr((props[i]), 'centroid'))
        return pts

    def corrPts(self, fix: np.ndarray, move: np.ndarray, thresh: int):
        '''
        compute corresponding point pairs (nearest neighbours) within a
        threshold distance
        args:   fix    -> fixed point cloud
                move   -> moving point cloud
                thresh -> thresholding distance

        returns a dictionary of corresponding point pairs
        '''
        self.fix = fix
        self.move = move
        pc1 = np.asarray([x for x in fix])
        pc1[:, 0] = pc1[:, 0] * 0.25/0.1625
        pc2 = np.asarray([x for x in move])
        pc2[:, 0] = pc2[:, 0] * 0.25/0.1625
        print(f"Fixed points: {len(fix)}")

        distance = cdist(pc1, pc2, 'euclidean')
        index = np.argmin(distance, axis=1)

        pairs = [{'point0': fix[i], 'point1': move[index[i]], 'index': i}
                 for i in range(len(pc1)) if distance[i, index[i]] < thresh]
        for i, pair in enumerate(pairs):
            pair['index'] = i
            pair['valid'] = True
        print(f"Pairs: {len(pairs)}")
        return pairs

