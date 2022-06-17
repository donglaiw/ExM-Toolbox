from yacs.config import CfgNode
from ..align.build import buildSitkTile
from ..data.build import getDataset

class Runner:

    def __init__(self, 
                cfg: CfgNode,
                mode: str = 'align'):

        self.cfg = cfg

        self.fix_vol, self.mov_vol = getDataset(self.cfg)
        
        self.align = buildSitkTile(self.cfg)
        
    def runAlign(self, save_result = True):

        tform = self.align.computeTransformMap(self.fix_vol, self.mov_vol)
        result = self.align.warpVolume(self.mov_vol, tform)

        return tform, result