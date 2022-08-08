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
