from tiffile import imread, imsave

def createTIFF(f_vol_out: str, channel_names: list):
    '''
    args: f_vol_out     -> path of the output volume (h5 file)
          channel_names -> channel names in the volume
    returns a numpy image array
    '''
    output = np.zeros((4, 81, 2048, 2048)).astype(np.uint16)

    for i in range(len(channel_names)):
        channel_name = channel_names[i]
        output[i] = np.array(h5py.File(f_vol_out, 'r')[channel_name].astype(np.uint16))

    output = np.swapaxes(output, 0, 1)

    return output


def writeTIFF(img_name: str, work_path: str, np_img: np.ndarray, channel_names: list):
    '''
    args: img_name      -> name of TIFF file
          np_img        -> numpy array from h5 file
          channel_names -> channel names in volume
    writes a TIFF file
    '''
    os.chdir(work_path)
    tiffile.imwrite(img_name, np_img,
                    metadata={'Channel': {'Name': channel_names}})

