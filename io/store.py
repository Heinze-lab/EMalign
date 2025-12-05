import os
import cv2
import numpy as np
import tensorstore as ts
from typing import List, Optional, Union
import json

from emprocess.utils.transform import rotate_image
from emprocess.utils.io import get_dataset_attributes


def open_store(
    path: str,
    mode: str = 'a',
    dtype: ts.dtype = ts.uint8,
    shape: Optional[List[int]] = None,
    chunks: Optional[List[int]] = None,
    axis_labels: Optional[List[str]] = None,
    fill_value: Optional[Union[float, int, bool]] = None
) -> ts.TensorStore:
    '''Open or create a Zarr store using TensorStore.

    Args:
        path (str): Absolute path to the zarr store.
        mode (str): Persistence mode, following zarr conventions:
            'r' - Read only (must exist)
            'r+' - Read/write (must exist)
            'a' - Read/write (create if doesn't exist) [default]
            'w' - Create (overwrite if exists)
            'w-' - Create (fail if exists)
        dtype (ts.dtype, optional): Data type of the array. Default: ts.uint8.
        shape (list of int or None): Shape of the array when creating a new store.
            Required for modes 'w' and 'w-', and for mode 'a' when store doesn't exist.
            Format typically [z, y, x] for 3D or [z, c, y, x] for 4D. Default: None.
        chunks (list of int or None): Chunk size when creating a new store. Must match
            the dimensionality of shape. Required when creating stores. Default: None.
        axis_labels (list of str or None): Labels for array dimensions in the transform.
            Common patterns:
            - ['z', 'y', 'x'] for 3D image stacks (default for 3D)
            - ['z', 'c', 'y', 'x'] for 4D arrays with channels
            - ['z', 'a', 'b'] for transformation matrices
            If None, will auto-infer based on shape dimensionality. Default: None.
        fill_value (float, int, bool, or None): Fill value for unwritten array elements. Only used when creating a new store. Default: None.

    Returns:
        tensorstore.TensorStore: Opened tensorstore object ready for reading or writing.

    Raises:
        ValueError: If required arguments are missing for the specified mode, or if
            incompatible arguments are provided.
        IOError: If path doesn't exist when required, or exists when it shouldn't.

    Examples:
        Read-only access to existing store:
        >>> dataset = open_store('/path/to/data.zarr', mode='r')

        Read/write to existing store:
        >>> dataset = open_store('/path/to/data.zarr', mode='r+', dtype=ts.uint8)

        Read/write, create if doesn't exist (default):
        >>> dataset = open_store('/path/to/data.zarr', dtype=ts.uint8,
        ...                       shape=[100, 2048, 2048], chunks=[1, 1024, 1024])

        Create new store, overwrite if exists:
        >>> dataset = open_store(
        ...     '/path/to/output.zarr',
        ...     mode='w',
        ...     dtype=ts.uint8,
        ...     shape=[100, 2048, 2048],
        ...     chunks=[1, 1024, 1024]
        ... )

        Create 4D flow field with NaN fill:
        >>> flow = open_store(
        ...     '/path/to/flow.zarr',
        ...     mode='w',
        ...     dtype=ts.float32,
        ...     shape=[100, 4, 1, 1],
        ...     chunks=[1, 4, 128, 128],
        ...     axis_labels=['z', 'c', 'y', 'x'],
        ...     fill_value=np.nan
        ... )
    '''
    # Validate mode
    valid_modes = ['r', 'r+', 'a', 'w', 'w-']
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")

    # Check if path exists
    path = os.path.abspath(path)
    path_exists = os.path.exists(path)

    # Mode: 'r' - Read only (must exist)
    if mode == 'r':
        if not path_exists:
            raise IOError(f'Zarr store not found at path: {path}')
        spec = {
            'driver': 'zarr',
            'kvstore': {'driver': 'file', 'path': path}
        }
        return ts.open(spec, dtype=dtype, read=True).result()

    # Mode: 'r+' - Read/write (must exist)
    if mode == 'r+':
        if not path_exists:
            raise IOError(f'Zarr store not found at path: {path}')
        spec = {
            'driver': 'zarr',
            'kvstore': {'driver': 'file', 'path': path}
        }
        return ts.open(spec, dtype=dtype).result()

    # Mode: 'w-' - Create (fail if exists)
    # Mode: 'w' - Create (overwrite if exists)
    # Mode: 'a' - Read/write (create if doesn't exist)
    if mode in ['w', 'w-', 'a']:
        if path_exists and mode == 'w-':
            raise IOError(f'Zarr store already exists at path: {path}')
        elif path_exists and mode == 'a':
            # Open existing store for read-write
            spec = {
                'driver': 'zarr',
                'kvstore': {'driver': 'file', 'path': path}
            }
            return ts.open(spec, dtype=dtype).result()

        # Validate required parameters for creation
        if shape is None:
            raise ValueError(f"shape is required when mode='{mode}'")
        if chunks is None:
            raise ValueError(f"chunks is required when mode='{mode}'")
        if len(shape) != len(chunks):
            raise ValueError(f'shape and chunks must have same length, got {len(shape)} and {len(chunks)}')

        # Auto-infer axis_labels if not provided
        if axis_labels is None:
            ndim = len(shape)
            if ndim == 3:
                axis_labels = ['z', 'y', 'x']
            elif ndim == 4:
                axis_labels = ['z', 'c', 'y', 'x']

        # Build spec for creating new store
        spec = {
            'driver': 'zarr',
            'kvstore': {'driver': 'file', 'path': path},
            'metadata': {'zarr_format': 2, 'shape': shape, 'chunks': chunks},
            'key_encoding': '/',
        }
        if axis_labels is not None:
            spec['transform'] = {'input_labels': axis_labels}

        kwargs = {'dtype': dtype, 'create': True, 'delete_existing': mode == 'w'}
        if fill_value is not None:
            kwargs['fill_value'] = fill_value

        return ts.open(spec, **kwargs).result()
    

def set_store_attributes(store, attrs):
    with open(os.path.join(store.kvstore.path, '.zattrs'), 'w') as f:
        json.dump(attrs, f, indent=2)
    return True 


def get_store_attributes(store):
    with open(os.path.join(store.kvstore.path, '.zattrs'), 'r') as f:
        attrs = json.load(f)
    return attrs


# WRITE
def write_slice(dataset, arr, z, x_offset=0, y_offset=0):

    y,x = arr.shape
    new_max = np.array([z+1, y+y_offset, x+x_offset], dtype=int)
    current_max = np.array(dataset.domain.exclusive_max, dtype=int)
    if np.any(current_max < new_max):
        new_max = np.max([current_max, new_max], axis=0)
        dataset = dataset.resize(exclusive_max=new_max, expand_only=True).result()
    try:
        return dataset, dataset[z:z+1, y_offset:y+y_offset, x_offset:x+x_offset].write(arr).result()
    except Exception as e:
        raise e


# READ
def find_ref_slice(dataset, z=None, reverse=False):

    '''Find first or last non-black slice of an image stack.

    Args:
        dataset (tensorstore.TensorStore): A dataset containing the image data.
        z (int or None): Z index to start from (axis=0). If None, will pick a start based on reverse. Default: None.
        reverse (bool): Whether to look for images at indices higher than z (False) or lower than z (True). Default: False.
    
    Returns:
        tuple: tuple of image np.ndarray and corresponding z index.
    '''
    
    increment = -1 if reverse else 1

    if z is None and reverse:
        z = dataset.domain.exclusive_max[0] - 1 
    elif z is None:
        z = dataset.domain.inclusive_min[0]

    img = dataset[z].read().result()

    while not img.any():
        z += increment
        img = dataset[z].read().result()
    return img, z


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