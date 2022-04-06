import numpy as np
from .io import writeH5, readH5, readNd2, readXlsx
from .image import imAdjust, imTrimBlack
import matplotlib.pyplot as plt

class tilesData:
    # input: nd2 and xlsx file
    # output: on-demand volume at given downsample ratio
    # assume zyx order
    def __init__(self, tiles_vol, tiles_info, tiles_loc):
        self.vol = tiles_vol
        self.tiles_size = tiles_info['tiles_size']
        self.resolution = tiles_info['resolution']
        self.channels = tiles_info['channels']
        # read xlsx metadata
        self.tiles_loc = tiles_loc
        self.tiles_loc_min = self.tiles_loc[:,:-1].min(axis=0)
        self.tiles_loc_max = self.tiles_loc[:,:-1].max(axis=0)
        self.vol_size = (self.tiles_loc_max - self.tiles_loc_min)/self.resolution + self.tiles_size        
        self.tiles_num = self.tiles_loc.shape[0]
        
        self.im_thres = None

    def setImThres(self, im_thres):
        self.im_thres = im_thres

    def setChannel(self, channel_name):
        self.channel_id = 0
        if len(channel_name) != 0:
            channel_id = [x for x in range(len(self.channels)) if channel_name in self.channels[x]]
            print(self.channels)
            assert len(channel_id) == 1
            self.channel_id = channel_id[0]

    def setRatio(self, ratio=[8,16,16]):
        self.ratio = ratio
        # output parameters
        self.tiles_size_o = self.getTileSize(ratio)
        self.vol_size_o = self.getVolumeSize(ratio)
        print('original size: ', self.vol_size.astype(int), '(downsampled):', self.vol_size_o)        
           
    def getResolution(self, ratio = None):
        if ratio is None:
            ratio = self.ratio        
        return self.resolution * ratio

    def getTileSize(self, ratio = None):
        if ratio is None:
            return self.tiles_size_o
        else:
            return np.ceil(self.tiles_size/ratio).astype(int)

    def getVolumeSize(self, ratio = None):        
        if ratio is None:
            return self.vol_size_o
        else:
            return np.ceil(self.vol_size/ratio).astype(int)
    
    def getTileVolume(self, tiles_id, ratio = None, im_thres = None, autoscale = None):
        # zyx order        
        ratio = self.ratio if ratio is None else ratio
        tiles_size_o = self.getTileSize(ratio)
        if im_thres is None:
            im_thres = self.im_thres

        if im_thres is None:
            tiles_vol = np.zeros(tiles_size_o, np.uint16)
            for z in range(tiles_size_o[0]):
                tiles_vol[z] = self.vol.get_frame_2D(c=self.channel_id, t=0, z=z*ratio[0], x=0, y=0, v=tiles_id)[::ratio[1], ::ratio[2]]
        else:
            # reduce the memory needed
            tiles_vol = np.zeros(tiles_size_o, np.uint8)
            # autoscale can cause drastic lighting change
            for z in range(tiles_size_o[0]):
                tiles_vol[z] = imAdjust(self.vol.get_frame_2D(c=self.channel_id, t=0, z=z*ratio[0], x=0, y=0, v=tiles_id)[::ratio[1], ::ratio[2]],\
                                      im_thres, autoscale=autoscale)
                    
        return tiles_vol

    def getTilePixPosition(self, tiles_id, ratio = None):
        if ratio is None:
            ratio = self.ratio
        top_left = (self.tiles_loc[tiles_id, :-1] - self.tiles_loc_min)/self.resolution/ratio
        return top_left
        
    def getTileRawStitch(self, ratio = None, im_thres = None, autoscale = None):
        # stitch volumes based on the xlsx locations
        if im_thres is None:
            im_thres = self.im_thres
        vol_size_o = self.getVolumeSize(ratio)
        ratio = self.ratio if ratio is None else ratio        
        output = np.zeros(vol_size_o, np.uint16)
        
        for v in range(self.tiles_num):
            tiles_vol = self.getTileVolume(self.tiles_loc[v, -1], ratio, im_thres)
            top_left = np.floor(self.getTilePixPosition(v)).astype(int)
            print('Load tile %d: '%v, 'pos=', top_left)
            output[top_left[0] : top_left[0] + tiles_vol.shape[0],\
                   top_left[1] : top_left[1] + tiles_vol.shape[1],\
                   top_left[2] : top_left[2] + tiles_vol.shape[2]] = tiles_vol
        if im_thres is not None:
            output = imAdjust(output, im_thres, autoscale).astype(np.uint8)
        return output

    def displayTileLoc(self):
        xy = self.tiles_loc[:,1:3]
        xy_scaled = (xy - xy.min(axis=0))/(xy.max(axis=0) - xy.min(axis=0))
        for vid in range(xy.shape[0]):
            plt.text(xy_scaled[vid,1],-xy_scaled[vid,0], '%d'% self.tiles_loc[vid,3])
        plt.axis('off')
