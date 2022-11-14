import json
import time
from intensity.run_coarse import *
from config.utils import load_cfg
from point.run_fine import warp_image


# run coarse and fine registration on a single image volume
# which is specified in the configuration file

# load configuration
cfg = load_cfg()
print(f"Current configration used:")
print(json.dumps(cfg, indent=4))


# create results directory structure
# createFolderStruc(out_dir='results/')

# run coarse alignment
print(f"\n\nrunning coarse alignment...")
tick = time.time()
# <run coarse registration>
tock = time.time()
print(f"Coarse alignment over!")
print(f"{tock - tick} seconds for coarse alignment!\n")

# run fine alignment
print(f"running fine alignment...")

tick = time.time()
warp_image(cfg)
tock = time.time()

print(f"Fine alignment over!")
print(f"{tock - tick} seconds for fine alignment!")
