import os
import cc3d
import numpy as np
import cv2 as cv
from tiffile import imread
from scipy.spatial.distance import cdist
from yacs.config import CfgNode
from skimage.measure import label, regionprops


class PointCloud:
    """
    Various point cloud utility options, main one being
    the generation of the fixed and moving clouds
    """

    def __init__(self, cfg: CfgNode):
        self.cfg = cfg

    def clean(self, fpath: str, h: int, threshold: int, channel: int) -> np.ndarray:
        """
        Clean a noisy image, similar to intensity based registration

        Args:
                    fpath: path to image file (either a volume or single channel)
                    h: strength of denoising filter
                    threshold: lower bound of pixel value
                    channel: channel to be cleaned

        Returns a denoised, binary masked image volume
        """
        assert os.path.exists(fpath), f"Non-existent path: {fpath}"
        img = imread(fpath)
        imgVol = list()

        if len(img.shape) == 4:
            im_channel = img[:, channel - 1]
        elif len(img.shape) == 3:
            im_channel = img
        else:
            print("Image must have either 3 or 4 dimensions")
            return

        for i in range(im_channel.shape[0]):
            denoised = cv.fastNlMeansDenoising(
                im_channel[
                    i,
                    :,
                    :,
                ].astype("uint8"),
                h=h,
                templateWindowSize=7,
                searchWindowSize=21,
            )
            (T, thresh) = cv.threshold(
                denoised, thresh=threshold, maxval=255, type=cv.THRESH_BINARY_INV
            )
            thresh = 255 - thresh
            imgVol.append(thresh)

        return np.array(imgVol)

    def segment(self, img: np.ndarray, delta: int) -> np.ndarray:
        """
        Instance segmentation of binary image volume,
        stitch synapses across volumes

        Args:
                img: image volume (ideally binary)
                delta: extent in voxel measure of puncta

        Return a segmented image with stitched 3D components
        """
        segImg = list()

        # segment image
        for i in range(img.shape[0]):
            imgSlice = img[
                i,
                :,
                :,
            ]
            seg = label(imgSlice, background=0)
            segImg.append(imgSlice)

        segImg = np.array(segImg)

        return cc3d.connected_components(segImg, delta=delta)

    def genPoints(self, img: np.ndarray) -> np.ndarray:
        """
        Generate a set of points corresponding to the centroid
        of the segment masks

        Args:
                img: image volume having stitched segment masks

        Returns a set of 3D points corresponding to the centroid
        of segment masks
        """
        props = regionprops(img)
        pts = np.zeros(shape=(len(props), 3))

        for i in range(len(props)):
            pts[i] = np.asarray(getattr((props[i]), "centroid"))

        return pts

    def genPointClouds(self):
        """
        Generate a fixed and moving point cloud using the
        above utility functions

        Args: None

        Returns a fixed and moving point cloud using the fixed
        moving image volumes defined in the default configuration file
        """

        # fixed point cloud
        fixVol = self.clean(
            fpath=self.cfg.DATASET.VOL_FIX_PATH,
            h=self.cfg.POINT.FILTER_STRENGTH_FIX,
            threshold=self.cfg.POINT.THRESH_LOWER_FIX,
            channel=self.cfg.POINT.CHANNEL,
        )
        fixSeg = self.segment(img=fixVol, delta=self.cfg.FILTER.DELTA)
        fixCloud = self.genPoints(fixSeg)

        # moving point cloud
        moveVol = self.clean(
            fpath=self.cfg.DATASET.VOL_MOVE_PATH,
            h=self.cfg.POINT.FILTER_STRENGTH_MOVE,
            threshold=self.cfg.POINT.THRESH_LOWER_MOVE,
            channel=self.cfg.POINT.CHANNEL,
        )
        moveSeg = self.segment(img=moveVol, delta=self.cfg.FILTER.DELTA)
        moveCloud = self.genPoints(moveSeg)

        return fixCloud, moveCloud
