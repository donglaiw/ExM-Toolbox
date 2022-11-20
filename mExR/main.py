import json
import time
import argparse
from intensity.run_coarse import *
from config.utils import load_cfg
from point.run_fine import warpImage
from intensity.run_coarse import runCoarse


def main(args):
    # run coarse and fine registration on a single image volume
    # which is specified in the configuration file

    # load configuration
    cfg = load_cfg()
    print(f"Current configration used:")
    print(json.dumps(cfg, indent=4))

    # run coarse alignment
    if args.coarse:
        print(f"\n\nRunning coarse alignment...")
        tick = time.time()
        runCoarse(cfg)
        tock = time.time()
        print(f"Coarse alignment over!")
        print(f"{(tock - tick):.4f} seconds for coarse alignment!\n")

    # run fine alignment
    if args.fine:
        print(f"\n\nRunning fine alignment...")
        tick = time.time()
        warpImage(cfg)
        tock = time.time()
        print(f"Fine alignment over!")
        print(f"{(tock - tick):.4f} seconds for fine alignment!")


if __name__ == "__main__":
    # choose registration type
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--coarse", default=False, type=bool, help="coarse alignment"
    )
    argparser.add_argument("--fine", default=False, type=bool, help="fine alignment")
    args = argparser.parse_args()

    main(args=args)
