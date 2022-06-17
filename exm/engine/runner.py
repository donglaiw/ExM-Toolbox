from .base import RunnerBase
from yacs.config import CfgNode
from ..align.build import buildSitkTile
from ..data.build import get_dataset

class Runner:

    def __init__(self, 
                cfg: CfgNode,
                mode: str = 'align'):

        self.cfg = cfg

        self.fix_vol, self.mov_vol = get_dataset(self.cfg)
        self.align = buildSitkTile(self.cfg)
        
    def runAlign(self, save_result = True):

        tform = self.align.computeTransformMap(self.fix_vol, self.mov_vol)
        result = self.align.warpVolume(self.mov_vol, tform)

        return result







    def init_basics(self, *args):
        # This function is used for classes that inherit Trainer but only 
        # need to initialize basic attributes in TrainerBase.
        super().__init__(*args)