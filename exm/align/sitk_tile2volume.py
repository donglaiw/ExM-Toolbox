import os,copy
import numpy as np
import h5py
import SimpleITK as sitk
from ..io import imAdjust, imTrimBlack, readH5, writeH5

class sitkTile2Volume:
    # 1. estimate transformation between input volumes
    # 2. warp one volume with the transformation
    # xyz order
    def __init__(self, tiles_data, volume_h5, aligner, ratio_output=[1,1,1]):
        self.tiles_data = tiles_data
        self.tiles_data.setRatio(ratio_output)
        self.ratio_output = np.array(ratio_output)
        self.res = self.tiles_data.getResolution()

        self.volume_h5 = volume_h5
        self.volume_sz = volume_h5.shape

        self.aligner = aligner
        self.aligner.setResolution(self.res[::-1])
        self.aligner.parameter_map['NumberOfResolutions'] = ['2'] # only need high-res 
        self.aligner.updateParameterMap()


    def updateOutputSize(self, pad_ratio = 2.5):
        # zyx order
        tile_size = self.tiles_data.getTileSize()
        self.pad_half = ((tile_size * (pad_ratio-1))//2).astype(int)
        self.output_size = tile_size + 2 * self.pad_half
        self.transform_init['Size'] = tuple([str(x) for x in self.output_size[::-1]])

    def setInitialTransform(self, transform_init):
        # unit: um
        self.transform_init = sitk.ParameterMap()
        for kk in transform_init.keys():
            self.transform_init[kk] = transform_init[kk]
        self.transform_init['Spacing'] = tuple([str(x) for x in self.res[::-1]])

        self.global_transform = np.array([float(x) for x in transform_init["TransformParameters"]])
        self.global_center = np.array([float(x) for x in transform_init['CenterOfRotationPoint']])[::-1]


    def alignTiles(self, tile_id, fn_out):
        fn_out = fn_out % (self.ratio_output[0], tile_id)
        # get tile data
        tile = self.tiles_data.getTileVolume(tile_id)
        # step 1. warp tile with transform_init
        # top-left corner position in the raw stitched volume
        tile_top_left = self.tiles_data.getTilePixPosition(tile_id) * self.res
        local_shift = - self.pad_half * self.res
        local_transform = self.global_transform.copy()
        local_transform[-3:] = local_shift[::-1]
        self.transform_init["TransformParameters"] = tuple([str(x) for x in local_transform])
        local_center = [str(x) for x in self.global_center - (tile_top_left + local_shift)]
        self.transform_init['CenterOfRotationPoint'] = tuple(local_center[::-1])
        # print(self.output_size)
        tile_warp = self.aligner.warpVolume(tile, transform_map=self.transform_init).astype(np.uint8)
        # writeH5(fn_out + '_db.h5', [tile, tile_warp], ['v0','v1'])
        #print(tile.mean(), tile_warp.mean())
       
        # step 2: find target region: target region in the reference volume
        # top-left point position
        global_shift = self.global_transform[-3:][::-1] - tile_top_left
        vol_top_left = (((local_shift - global_shift) / self.res)).astype(int)
        tile_crop_left = np.maximum(0, - vol_top_left)
        vol_bottom_right = np.minimum(self.volume_sz, vol_top_left + self.output_size)
        vol_top_left = np.maximum(0, vol_top_left)

        vol_crop = np.array(self.volume_h5[vol_top_left[0] : vol_bottom_right[0],\
                                           vol_top_left[1] : vol_bottom_right[1],\
                                           vol_top_left[2] : vol_bottom_right[2]])
        vol_crop = imAdjust(vol_crop, self.tiles_data.im_thres).astype(np.uint8)
        tile_crop_sz = vol_crop.shape
        #print(tile_warp.shape, tile_crop_left, tile_crop_sz)
        tile_warp_crop = tile_warp[tile_crop_left[0] : tile_crop_left[0] + tile_crop_sz[0],\
                                   tile_crop_left[1] : tile_crop_left[1] + tile_crop_sz[1],\
                                   tile_crop_left[2] : tile_crop_left[2] + tile_crop_sz[2]]

        # step 3: crop out empty space
        #print(tile_warp_crop.max())
        _, tile_crop_trim = imTrimBlack(tile_warp_crop>0, True)        
        tile_warp_crop_trim = tile_warp_crop[tile_crop_trim[0] : tile_crop_trim[1],\
                                            tile_crop_trim[2] : tile_crop_trim[3],\
                                            tile_crop_trim[4] : tile_crop_trim[5]]
        vol_crop_trim = vol_crop[tile_crop_trim[0] : tile_crop_trim[1],\
                               tile_crop_trim[2] : tile_crop_trim[3],\
                               tile_crop_trim[4] : tile_crop_trim[5]]
        # for final combination
        vol_top_left += tile_crop_trim[::2]
        np.savetxt(fn_out + '_coord.txt', vol_top_left, '%d')
        # debug initial transformation
        # writeH5(fn_out + '_db.h5', [vol_crop_trim, tile_warp_crop_trim], ['v0','v1'])
        
        # step 4: high-res warp
        #print('align', vol_crop_trim.shape, tile_warp_crop_trim.shape)
        try:
            transform_hr = self.aligner.computeTransformMap(vol_crop_trim, tile_warp_crop_trim,\
                                        mask_fix = (vol_crop_trim>0).astype(np.uint8),\
                                        mask_move = (tile_warp_crop_trim>0).astype(np.uint8))
            self.aligner.writeTransformMap(fn_out + '_transform.txt', transform_hr)
            tile_output = self.aligner.warpVolume(tile_warp_crop_trim, transform_hr).astype(np.uint8)
            writeH5(fn_out + '.h5', tile_output)
        except: # some tiles only have background and may cause error
            # sometimes, it's okay to make an error in background tiles
            print('error in alignment')
            writeH5(fn_out + '_err.h5', np.array([0]))

    def stitchTiles(self, fn_out, chunk_size=(100,1024,1024), bg_val=-1):
        # stitch volumes based on the xlsx locations
        fn_out_vol = fn_out[:fn_out.rfind('/')] + '_stitched-%d.h5'%(self.ratio_output[0])
        fid = h5py.File(fn_out_vol, 'w')
        ds = fid.create_dataset('main', self.volume_sz, compression="gzip", dtype=np.uint8, chunks=tuple(chunk_size)) 
        for tile_id in range(self.tiles_data.tiles_num):
            sn = fn_out % (self.ratio_output[0], tile_id) + '.h5'
            if os.path.exists(sn):
                tile_output = readH5(sn)
                vol_top_left = np.loadtxt(fn_out % (self.ratio_output[0],tile_id) + '_coord.txt').astype(int)
                vol_bottom_right = vol_top_left + tile_output.shape
                print('Load tile %d: '%tile_id, 'pos=', vol_top_left)
                vol = np.array(ds[vol_top_left[0] : vol_bottom_right[0], 
                                   vol_top_left[1] : vol_bottom_right[1],\
                                   vol_top_left[2] : vol_bottom_right[2]])
                if bg_val == -1:# element-wise max
                    vol = np.maximum(vol, tile_output)
                else: # greedy overlap
                    vol[tile_output>bg_val] = tile_output[tile_output>bg_val]
                

                ds[vol_top_left[0] : vol_bottom_right[0],\
                    vol_top_left[1] : vol_bottom_right[1],\
                    vol_top_left[2] : vol_bottom_right[2]] = vol
        fid.close()
