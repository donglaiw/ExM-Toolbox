from exm.config.utils import load_cfg
from exm.engine.runner import Runner
import matplotlib.pyplot as plt
from time import time

t_start = time()

#get config file
cfg = load_cfg()

#defrost to add vol paths and fov
cfg.defrost()

cfg.DATASET.VOL_FIX_PATH = '/mp/nas2/ruihan/20220512/20220512_Code3/Channel405 SD_Seq0004.nd2'
cfg.DATASET.VOL_MOVE_PATH = '/mp/nas2/ruihan/20220512/20220512_Code5/Channel405 SD_Seq0004.nd2'
cfg.DATASET.FOV = 23

cfg.freeze()
# init runner obj and run align

run = Runner()
tform, result = run.runAlign(cfg)

t_end = time()

print(t_end-t_start)
print(tform.items())
plt.imshow(result[200,:,:])
plt.savefig('./example_result.jpg')

