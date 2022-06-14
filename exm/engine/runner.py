from .base import RunnerBase
from yacs.config import CfgNode

class Runner(RunnerBase):

    def __init__(self, 
                cfg: CfgNode,
                mode: str = 'align'):
        
        self.init_basics(cfg, mode)

        if self.args.mode == 'align':
            

    def init_basics(self, *args):
        # This function is used for classes that inherit Trainer but only 
        # need to initialize basic attributes in TrainerBase.
        super().__init__(*args)