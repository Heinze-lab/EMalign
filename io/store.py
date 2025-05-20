import cv2
import numpy as np

from emprocess.utils.transform import rotate_image
from emprocess.utils.io import get_dataset_attributes

# WRITE
def write_slice(dataset, arr, z, x_offset=0, y_offset=0):

    y,x = arr.shape
    new_max = np.array([z+1, y+y_offset, x+x_offset])
    if np.any(np.array(dataset.domain.exclusive_max) < new_max):
        new_max = np.max([dataset.domain.exclusive_max, new_max], axis=0)
        dataset = dataset.resize(exclusive_max=new_max, expand_only=True).result()
    try:
        return dataset, dataset[z:z+1, y_offset:y+y_offset, x_offset:x+x_offset].write(arr).result()
    except Exception as e:
        raise e


# READ
def get_data_slice(dataset, z, offset, target_scale, rotation_angle=0):

    '''
    Get image from a tensorstore dataset. Optionally resize, rotate, and pad to match a calculated offset and rotation.
    '''

    data = dataset[z, ...].read().result()
    data = cv2.equalizeHist(data)

    if target_scale < 1:
        data = cv2.resize(data, None, fx=target_scale, fy=target_scale)

    if rotation_angle != 0:
        data = rotate_image(data, rotation_angle)

    data = np.pad(data, np.stack([offset[1:], [0,0]]).T)
    return data


def get_data_samples(dataset, step_slices, yx_target_resolution):

    resolution = np.array(get_dataset_attributes(dataset)['resolution'])[1:]

    z_max = dataset.domain.exclusive_max[0]-1

    z_list = np.arange(0, z_max, step_slices)
    z_list = np.append(z_list, z_max) if z_max not in z_list else z_list    

    data = []
    for z in z_list:
        arr = dataset[z].read().result()
        while not arr.any():
            z += 1
            arr = dataset[z].read().result()
        
        if np.any(resolution < yx_target_resolution):
            fy, fx = resolution/yx_target_resolution
            arr = cv2.resize(arr, None, fx=fx, fy=fy)
        elif np.any(resolution > yx_target_resolution):
            raise RuntimeError(f'Dataset resolution ({resolution.tolist()}) must be lower \
                               than target resolution ({yx_target_resolution.tolist()})')
        data.append(arr)

    return np.array(data)