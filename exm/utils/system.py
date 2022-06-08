import os
import argparse

__all__ = [
    'get_args',
    'init_devices',
]

# structure again shamelessly stolen from zudi lin:https://github.com/zudi-lin/pytorch_connectomics/blob/c5669c9fdc4f28a96153dd33cf410e75fc6f1476/connectomics/utils/system.py#L16


def get_args():
    parser = argparse.ArgumentParser(description="Model Training & Inference")
    parser.add_argument('--config-file', type=str,
                        help='configuration file (yaml)')
    parser.add_argument('--config-base', type=str,
                        help='base configuration file (yaml)', default=None)
    parser.add_argument('--mode', type=str,
                        help='which step of processing pipeline to run', default='align')
    # Merge configs from command line (e.g., add 'SYSTEM.NUM_GPUS 8').
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args

    