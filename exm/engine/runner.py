from .base import RunnerBase

class Runner(RunnerBase):

    def __init__(self, cfg: CfgNode):

        if self.args.mode == 'align':
            return None