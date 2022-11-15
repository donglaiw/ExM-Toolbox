import json
import time
from intensity.run_coarse import *
from config.utils import load_cfg
from point.run_fine import warpImage
from intensity.run_coarse import runCoarse


def main():
    # run coarse and fine registration on a single image volume
    # which is specified in the configuration file

    # load configuration
    cfg = load_cfg()
    print(f"Current configration used:")
    print(json.dumps(cfg, indent=4))

    # run coarse alignment
    print(f"\n\nrunning coarse alignment...")
    tick = time.time()
    runCoarse(cfg)
    tock = time.time()
    print(f"Coarse alignment over!")
    print(f"{(tock - tick):.4f} seconds for coarse alignment!\n")

    # run fine alignment
    print(f"running fine alignment...")
    tick = time.time()
    warpImage(cfg)
    tock = time.time()
    print(f"Fine alignment over!")
    print(f"{(tock - tick):.4f} seconds for fine alignment!")


if __name__ == "__main__":
    main()
