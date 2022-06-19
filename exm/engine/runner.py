from yacs.config import CfgNode
from ..align.build import alignBuild
from ..data.build import getDataset

class Runner:

    def __init__(self):

        self.cfg = None
        self.align = None
        
    def runAlign(self, cfg: CfgNode):

        self.cfg = cfg

        fix_vol, mov_vol = getDataset(self.cfg)
        
        self.align = alignBuild(self.cfg)

        self.align.buildSitkTile(self.cfg)

        tform = self.align.computeTransformMap(fix_vol, mov_vol, mask_fix = fix_vol.mask, mask_move = mov_vol.mask)
        result = self.align.warpVolume(mov_vol, tform)

        return tform, result