import os
import warnings
import argparse
from yacs.config import CfgNode
from .default import get_cfg_defaults

# thanks Jack yet again :)

def load_cfg(config_file=None, freeze=True, add_cfg_func=None):
    """Load configurations
    """
    cfg = get_cfg_defaults()
    if add_cfg_func is not None:
        add_cfg_func(cfg)
    if config_file is not None:
        cfg.merge_from_file(config_file)

    if freeze:
        cfg.freeze()
    else:
        warnings.warn("Configs are mutable during the process,"
                      "please make sure that is expected")
    return cfg

def save_all_cfg(cfg: CfgNode, output_dir: str):
    r"""Save configs in the output directory
    """
    # save config.yaml in the experiment directory after combine all
    # non-default configurations from yaml file and command line
    path = os.path.join(output_dir, "config.yaml")
    with open(path, "w") as f:
        f.write(cfg.dump())
    print(f"Full config saved to {path}")
