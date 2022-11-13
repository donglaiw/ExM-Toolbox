import os, sys
import numpy as np


def createFolderStruc(out_dir: str, fov: str):

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # results from coarse registration
    if not os.path.exists(os.path.join(out_dir, "coarse")):
        os.mkdir(os.path.join(out_dir, "coarse"))

    # results from fine registration
    if not os.path.exists(os.path.join(out_dir, "fine")):
        os.mkdir(os.path.join(out_dir, "fine"))

    # corresponding point set
    if not os.path.exists(os.path.join(out_dir, "points")):
        os.mkdir(os.path.join(out_dir, "points"))
        point_dir = os.path.join(out_dir, "points")
        os.chdir(point_dir)
        os.mkdir(fov)
