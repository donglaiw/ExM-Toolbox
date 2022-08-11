from yacs.config import CfgNode
from ..align.build import alignBuild
from ..io.io import nd2ToVol

class Runner:

    def __init__(self):

        self.cfg = None
        self.align = None
        
    def runAlign(self, cfg: CfgNode):

        self.cfg = cfg

        # fix_vol, mov_vol = getDataset(self.cfg)
        fix_vol = nd2ToVol(self.cfg.DATASET.VOL_FIX_PATH, cfg.DATASET.FOV)
        mov_vol = nd2ToVol(self.cfg.DATASET.VOL_MOVE_PATH, cfg.DATASET.FOV)

        self.align = alignBuild(self.cfg)

        self.align.buildSitkTile()

        tform = self.align.computeTransformMap(fix_vol, mov_vol)
        result = self.align.warpVolume(mov_vol, tform)

        return tform, result