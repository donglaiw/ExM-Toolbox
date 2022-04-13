import os,copy
import numpy as np
import h5py
import SimpleITK as sitk
from ..io import imAdjust, imTrimBlack, readH5, writeH5

class sitkTile2Volume:
    # 1. estimate transformation between input volumes
    # 2. warp one volume with the transformation
    # xyz order
    def __init__(self, tiles_data, getVolume, aligner, ratio_output=[1,1,1]):
        self.tiles_data = tiles_data
        self.tiles_data.setRatio(ratio_output)
        self.ratio_output = np.array(ratio_output)
        self.res = self.tiles_data.getResolution()

        self.getVolume = getVolume

        self.aligner = aligner
        self.aligner.setResolution(self.res[::-1])
        self.aligner.parameter_map['NumberOfResolutions'] = ['1'] # only need high-res 
        self.aligner.updateParameterMap()
        self.trim_threshold = 0
        self.mask_threshold = 0

    def setTrimThreshold(self, threshold = 0):
        self.trim_threshold = threshold

    def setMaskThreshold(self, threshold = 0):
        self.mask_threshold = threshold

    def updateOutputSize(self, pad_ratio = 2.5):
        # ratio = im_pad/im
        # zyx order
        tile_size_o = self.tiles_data.getTileSize()
        self.pad_half = ((tile_size_o * (pad_ratio-1))//2).astype(int)
        self.output_size = tile_size_o + 2 * self.pad_half
        self.transform_init['Size'] = tuple([str(x) for x in self.output_size[::-1]])

    def setInitialTransform(self, transform_init):
        # unit: um
        self.transform_init = sitk.ParameterMap()
        for kk in transform_init.keys():
            self.transform_init[kk] = transform_init[kk]
        # xyz
        self.transform_init['Spacing'] = tuple([str(x) for x in self.res[::-1]])
        self.global_transform = np.array([float(x) for x in transform_init["TransformParameters"]])
        self.global_center = np.array([float(x) for x in transform_init['CenterOfRotationPoint']])


    def alignTiles(self, tile_id, fn_out, force_align=False):
        fn_out = fn_out + '-%d-%d' % (tile_id, self.ratio_output[0])
        # get tile data
        tile = self.tiles_data.getTileVolume(tile_id)
        # step 1. warp tile with transform_init
        # top-left corner position in the raw stitched volume
        # zyx
        tile_top_left = self.tiles_data.getTilePhysicalPosition(tile_id)
        local_shift = - self.pad_half * self.res
        local_transform = self.global_transform.copy()
        local_transform[-3:] = local_shift[::-1]
        self.transform_init["TransformParameters"] = tuple([str(x) for x in local_transform])

        local_center = [str(x) for x in self.global_center - (tile_top_left + local_shift)[::-1]]
        self.transform_init['CenterOfRotationPoint'] = tuple(local_center)
        # print(self.output_size)
        tile_warp = self.aligner.warpVolume(tile, transform_map=self.transform_init).astype(np.uint8)
        #writeH5(fn_out + '_db.h5', [tile, tile_warp], ['v0','v1'])
        #import pdb; pdb.set_trace()
        #print(tile.mean(), tile_warp.mean())
       
        # step 2: find target region: target region in the reference volume
        # top-left point position
        global_shift = self.global_transform[-3:][::-1] - tile_top_left
        vol_top_left = (((local_shift - global_shift) / self.res)).astype(int)
        tile_crop_left = np.maximum(0, - vol_top_left)
        #vol_bottom_right = np.minimum(self.volume_sz, vol_top_left + self.output_size)
        # the overflow index will be truncated automatically
        vol_bottom_right = vol_top_left + self.output_size
        vol_top_left = np.maximum(0, vol_top_left)

        vol_crop = self.getVolume(vol_top_left[0],vol_bottom_right[0],\
                                   vol_top_left[1], vol_bottom_right[1],\
                                   vol_top_left[2], vol_bottom_right[2])
        vol_crop = imAdjust(vol_crop, self.tiles_data.im_thres).astype(np.uint8)
        tile_crop_sz = vol_crop.shape
        #print(tile_warp.shape, tile_crop_left, tile_crop_sz)
        tile_warp_crop = tile_warp[tile_crop_left[0] : tile_crop_left[0] + tile_crop_sz[0],\
                                   tile_crop_left[1] : tile_crop_left[1] + tile_crop_sz[1],\
                                   tile_crop_left[2] : tile_crop_left[2] + tile_crop_sz[2]]
       
        # writeH5(fn_out + '_db.h5', [vol_crop, tile_warp_crop], ['v0','v1'])
        # shift: local_shift - global_shift
        if self.trim_threshold == -1: # no trim
            tile_warp_crop_trim = tile_warp_crop
            vol_crop_trim = vol_crop
        else:
            # step 3: crop out empty space
            #print(tile_warp_crop.max())
            _, tile_crop_trim = imTrimBlack((tile_warp_crop > self.trim_threshold) * (vol_crop > self.trim_threshold), return_ind=True)
            if tile_crop_trim[1::2].min() == 0:
                # trim upper bound shouldn't be 0
                print('trim 0: error in trim')
                writeH5(fn_out + '_err.h5', np.array([0]))
                return

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
        #writeH5(fn_out + '_db.h5', [vol_crop_trim, tile_warp_crop_trim], ['v0','v1'])
        #import pdb; pdb.set_trace()
        # visual: compare two volume
        
        # step 4: high-res warp
        #print('align', vol_crop_trim.shape, tile_warp_crop_trim.shape)
        try:
            if force_align:
                # use with caution: ignore not enough samples error
                self.aligner.parameter_map['CheckNumberOfSamples'] = ['false']
            else:
                self.aligner.parameter_map['CheckNumberOfSamples'] = ['true']
            self.aligner.elastix.SetParameterMap(self.aligner.parameter_map)

            transform_hr = self.aligner.computeTransformMap(vol_crop_trim, tile_warp_crop_trim,\
                                        mask_fix = (vol_crop_trim > self.mask_threshold).astype(np.uint8),\
                                        mask_move = (tile_warp_crop_trim > self.mask_threshold).astype(np.uint8))
            self.aligner.writeTransformMap(fn_out + '_transform.txt', transform_hr)
            tile_output = self.aligner.warpVolume(tile_warp_crop_trim, transform_hr).astype(np.uint8)
            writeH5(fn_out + '.h5', tile_output)
        except: # some tiles only have background and may cause error
            # sometimes, it's okay to make an error in background tiles
            print('error in alignment')
            import pdb; pdb.set_trace()
            writeH5(fn_out + '_err.h5', np.array([0]))

    def stitchTiles(self, volume_sz, fn_out, stitch_out, chunk_size=(100,1024,1024), bg_val=-1):
        # stitch volumes based on the xlsx locations
        fid = h5py.File(stitch_out, 'w')
        ds = fid.create_dataset('main', volume_sz, compression="gzip", dtype=np.uint8, chunks=tuple(chunk_size)) 
        for tile_id in range(self.tiles_data.tiles_num):
            sn = fn_out + '-%d-%d' % (tile_id, self.ratio_output[0])
            if os.path.exists(sn + '.h5'):
                tile_output = readH5(sn + '.h5')
                vol_top_left = np.loadtxt(sn + '_coord.txt').astype(int)
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
