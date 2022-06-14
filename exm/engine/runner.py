from .base import RunnerBase
from yacs.config import CfgNode
from ..align.sitk_tile import sitkTile
from ..data.build import get_dataset

class Runner(RunnerBase):

    def __init__(self, 
                cfg: CfgNode,
                mode: str = 'align'):
        
        self.init_basics(cfg, mode)

        self.dataset = get_dataset(self.cfg)

        if self.args.mode == 'align':
            self.align = sitkTile()
            self.align.setResolution()
            self.align.setTransformType()
        
    def runAlign(self):

        tform = self.align.computeTransformMap(self.dataset)
        result = self.align.warpVolume(self.dataset.vol_move, tform)
        self.dataset.setResult(result)

        return self.dataset






    def init_basics(self, *args):
        # This function is used for classes that inherit Trainer but only 
        # need to initialize basic attributes in TrainerBase.
        super().__init__(*args)