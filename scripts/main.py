import os

from exm.utils.system import get_args
from exm.config import load_cfg, save_all_cfg
from exm.io.io import createFolderStruc

def main():
    args = get_args()
    cfg = load_cfg(args)

    if not os.path.exists(cfg.DATASET.OUTPUT_PATH):
        print('Output directory: ', cfg.DATASET.OUTPUT_PATH)
        createFolderStruc(cfg.DATASET.OUTPUT_PATH)
        save_all_cfg(cfg, cfg.DATASET.OUTPUT_PATH)

    # start pipeline
    #if args.mode == 'align':


if __name__ == "__main__":
    main()