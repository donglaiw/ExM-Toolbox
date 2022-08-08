import os
import cv2 as cv
import numpy as np
from tiffile import imread, imwrite
from yacs.config import CfgNode


class Filter:
    """
    Denoise, threshold and mask image volumes. This is useful when
    microscopy images have been corrupted with noise due to imaging artifacts.
    In such cases, intensity-based registration generally fails. Denoising and
    thresholding to form binary image masks may prove to be useful. We also use
    this to compute the centroid of segment masks (eg: synapses) to serve as points
    in a point cloud for point-based registration.
    """

    def __init__(self, cfg: CfgNode):
        self.cfg = cfg

    def denoiseImg(
        self, f_volImg: str, channel: int, filterStrength: int = None
    ) -> np.ndarray:
        """
        Denoise an image volume at a specific channel

        Args:
                f_volImg:       path to image volume
                channel:        image channel for denoising
                filterStrength: strength of denoising filter; a larger number
                                removes more noise as well as image features

        Returns a denoised image matrix
        """
        assert os.path.exists(f_volImg), "The path: {f_volImg} does not exist"
        imgVol = imread(f_volImg)[:, channel - 1]
        if filterStrength is not None:
            self.filterStrength = filterStrength
        else:
            self.filterStrength = self.cfg.FILTER.FILTER_STRENGTH
        denoised = [
            cv.fastNlMeansDenoising(
                imgVol[
                    z,
                    :,
                    :,
                ].astype("uint8"),
                h=self.filterStrength,
                templateWindowSize=7,
                searchWindowSize=21,
            )
            for z in range(imgVol.shape[0])
        ]
        return np.array(denoised)

    def threshold(self, imgVol: np.ndarray, threshLower: int = None) -> np.ndarray:
        """
        adaptive thresholding on image volume to form a binary mask

        Args:
                imgVol:      Image volume as numpy matrix
                threshLower: Lower bound on pixel value

        Returns a binary thresholded image volume
        """
        assert imgVol.ndim == 3, "only volumetric images"
        if threshLower is not None:
            self.threshLower = threshLower
        else:
            self.threshLower = self.cfg.FILTER.THRESH_LOWER
        thresholdVol = list()
        for z in range(imgVol.shape[0]):
            (T, thresholdSlice) = cv.threshold(
                imgVol[
                    z,
                    :,
                    :,
                ],
                thresh=self.threshLower,
                maxval=self.cfg.FILTER.THRESH_UPPER,
                type=cv.THRESH_BINARY_INV,
            )
            thresholdVol.append(255 - np.array(thresholdSlice))

        return np.array(thresholdVol)

    def maskSmall(self, imgVol: np.ndarray, minIdx: int = -1) -> np.ndarray:
        """
        mask small noise artifacts in an image, ie: all except
        the 'minIdx' largest areas

        Args:
                imgVol: image volume to be masked
                minIdx: all except the minIdx areas are masked
                        to the background

        Returns a masked image volume
        """
        self.minIdx = self.cfg.FILTER.MASK_INDEX
        assert minIdx > 1, "minimum area index must be greater than 1"

        maskedImage = []
        for z in range(imgVol.shape[0]):
            upIdx = list()
            labeled_img = label(
                imgVol[
                    z,
                    :,
                    :,
                ],
                background=0,
            ).astype("uint8")
            indices, area = np.unique(labeled_img, return_counts=True)
            sortedArea = sorted(area)

            for i in range(len(area)):
                for j in range(2, minIdx):
                    if area[i] == sortedArea[len(area) - j]:
                        upIdx.append(i)

            lowIdx = (idx for idx in range(len(area)) if idx not in upIdx)

            for idx in lowIdx:
                labeled_img[np.where(labeled_img == idx)] = 0

            maskedImage.append(labeled_img)

        return np.array(maskedImage)
