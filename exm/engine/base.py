import numpy as np
from yacs.config import CfgNode

class RunnerBase(object):

    def __init__(self,
                cfg: CfgNode,
                mode: str = 'align'):
        self.output_dir = cfg.DATASET.OUTPUT_PATH
        self.mode = mode