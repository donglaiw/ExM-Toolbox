import cc3d
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import SimpleITK as sitk
from tiffile import imread, imwrite
from skimage.measure import label, regionprops
from skimage.transform import warp
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import cdist


class PointCloud:
    '''
    compute and register point clouds
    use interpolation to warp image volume
    '''
    def __init__(self, fix, move):
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

    def RBF(self, fix: np.ndarray, move: np.ndarray, cspace: np.ndarray, percentage=0.05):
        '''
        compute the interpolation vectors using a thin plate spline
        kernel between the fixed and moving point clouds

        args:   fix        -> fixed point cloud
                move       -> moving point cloud
                cspace     -> original coordinate space of the image
                percentage -> fraction of corresponding points to be used for
                              registration (default = 0.05)

        returns interpolation matrices along the X, Y and Z axes for the
        original coordinate space
        '''
        self.fix = fix
        self.move = move
        index_small = np.random.choice(range(len(fix)), int(percentage*len(fix)), replace=False)
        fix_small  = fix[index_small,:]
        move_small = move[index_small,:]
        # differences along X, Y, Z axes (order is [z, y, x])
        dz_small = np.asarray(fix_small-move_small)[:, 0]
        dy_small = np.asarray(fix_small-move_small)[:, 1]
        dx_small = np.asarray(fix_small-move_small)[:, 2]
        # interpolation along all axes
        rbf_z = RBFInterpolator(move_small, dz_small, kernel='thin_plate_spline', degree=3)
        rbf_y = RBFInterpolator(move_small, dy_small, kernel='thin_plate_spline', degree=3)
        rbf_x = RBFInterpolator(move_small, dx_small, kernel='thin_plate_spline', degree=3)
        # interpolation vectors
        z_diff = rbf_z(move)
        y_diff = rbf_y(move)
        x_diff = rbf_x(move)
        transformed = move
        transformed[:, 0] += z_diff
        transformed[:, 1] += y_diff
        transformed[:, 2] += x_diff
        # mean differences along axes
        print(f"Mean differences: Z-mean = {np.mean(abs(transformed[:, 0]-fix[:, 0]))}\n
                                  Y-mean = {np.mean(abs(transformed[:, 1]-fix[:, 1]))}\n
                                  X-mean = {np.mean(abs(transformed[:, 2]-fix[:, 2]))}")

        if (np.mean(abs(transformed[:,0]-target[:,0])) < 15  and np.mean(abs(transformed[:,1]-target[:,1])) < 15 and
            np.mean(abs(transformed[:,2]-target[:,2])) < 15):
            return transformed, rbf_x(cspace), rbf_y(cspace), rbf_z(cspace)
        else:
            print('high mean, redoing interpolation')
            RBF(fix=fix, move=move, cspace=cspace)

    def warp(f_path: str, f_fix: str, f_move: str):
        '''
        warp an image volume using an initial and final coordinate space representation

        args:   f_path -> path to image volume
                f_fix  -> path to fixed points
                f_move -> path to moving points

        returns an image (16-bit) warped with a new coordinate space
        '''
        img = imread(f_path)
        # generate coordinate mesh
        zz, yy, xx = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), np.arange[2],
                                 indexing='ij')
        # read points
        fix = np.loadtxt(f_fix, delimiter=' ')
        move = np.loadtxt(f_move, delimiter=' ')
        self.fix = fix
        self.move = move
        # generate coordinate indices
        print(f"Generating coordinate indices (this might take a while)...")
        cspace = np.stack([x for x in np.ndindex(img.shape[0], img.shape[1], img.shape[2])])
        print(f"Completed coordinate generation\n\n")
        # compute interpolation vectors
        print(f"Computing interpolation vectors (this might take a while)...")
        _, dx, dy, dz = RBF(move, fix, cspace, percentage=0.05)
        print(f"Computed interpolation vectors\n\n")
        # reshape interpolation vector
        dx = dx.reshape(img.shape[0], img.shape[1], img.shape[2])
        dy = dy.reshape(img.shape[0], img.shape[1], img.shape[2])
        dz = dz.reshape(img.shape[0], img.shape[1], img.shape[2])
        # warp image (16-bit)
        print(f"Warping image...")
        img_warp = warp(img, np.array([zz+dz, yy+dy, xx+dx]), order=3, preserve_range=True).astype('uint16')
        return img_warp

