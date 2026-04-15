import os
import cv2
import numpy as np
import tensorstore as ts
from typing import List, Optional, Union
import json

from emalign.arrays.utils import resample

def open_store(
    path: str,
    mode: str = 'a',
    dtype: ts.dtype = ts.uint8,
    shape: Optional[List[int]] = None,
    chunks: Optional[List[int]] = None,
    axis_labels: Optional[List[str]] = None,
    fill_value: Optional[Union[float, int, bool]] = None,
    allow_missing: Optional[bool] = False
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
        allow_missing (bool): Whether to allow the store to be missing. If False, will raise an IO error. If True, will return None. Default: False.

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

    if not path_exists and mode in ['r', 'r+']:
        if not allow_missing:
            raise IOError(f'Zarr store not found at path: {path}')
        else:
            return None

    # Spec for existing datasets
    spec = {
        'driver': 'zarr',
        'kvstore': {'driver': 'file', 'path': path}
    }

    # Mode: 'r' - Read only (must exist)
    if mode == 'r':
        return ts.open(spec, read=True).result()

    # Mode: 'r+' - Read/write (must exist)
    if mode == 'r+':
        return ts.open(spec).result()

    # Mode: 'w-' - Create (fail if exists)
    # Mode: 'w' - Create (overwrite if exists)
    # Mode: 'a' - Read/write (create if doesn't exist)
    if mode in ['w', 'w-', 'a']:
        if path_exists and mode == 'w-':
            raise IOError(f'Zarr store already exists at path: {path}')
        elif path_exists and mode == 'a':
            # Open existing store for read-write
            return ts.open(spec).result()

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
    

def set_store_attributes(store: ts.TensorStore | str, attrs: dict) -> bool:
    '''Set attributes for a Zarr store.

    Args:
        store (tensorstore.TensorStore | str): The store or its kvstore path.
        attrs (dict): Dictionary of attributes to store.

    Returns:
        bool: True if successful.

    Raises:
        IOError: If the .zattrs file cannot be written.
    '''
    path = store if isinstance(store, str) else store.kvstore.path
    attrs_path = os.path.join(path, '.zattrs')
    with open(attrs_path, 'w') as f:
        json.dump(attrs, f, indent=2)
    return True


def get_store_attributes(store: ts.TensorStore | str) -> dict:
    '''Get attributes from a Zarr store.

    Args:
        store (tensorstore.TensorStore | str): The store or its kvstore path.

    Returns:
        dict: Dictionary of stored attributes.

    Raises:
        IOError: If the .zattrs file cannot be read.
        json.JSONDecodeError: If the .zattrs file contains invalid JSON.
    '''
    path = store if isinstance(store, str) else store.kvstore.path
    attrs_path = os.path.join(path, '.zattrs')
    with open(attrs_path, 'r') as f:
        attrs = json.load(f)
    return attrs


def write_ndarray(
    dataset: ts.TensorStore,
    arr: np.ndarray,
    z: int,
    xy_offset: Optional[List[int]] = None,
    resolve: bool = True
) -> tuple:
    '''Write an N-dimensional array to a tensor store dataset at a specific z-index.

    This is the core write function that handles all dimensionalities and indexing patterns.
    Automatically resizes the dataset if needed.

    Args:
        dataset (tensorstore.TensorStore): The tensorstore to write to.
        arr (np.ndarray): N-dimensional array to write. Supported shapes:
            - 2D [y, x] for writing to 3D [z, y, x] datasets
            - 3D [c, y, x] for writing to 4D [z, c, y, x] datasets (e.g., flow fields)
            - 2D [a, b] for writing to 3D [z, a, b] datasets (e.g., transformation matrices)
        z (int): Z-index where to write the data (first dimension).
        xy_offset (list of int or None): Optional offsets for non-z spatial dimensions, for 2D arrays.
            Default: None (all offsets = 0).
        resolve (bool): If True, calls dataset.resolve().result() to refresh metadata
            before checking bounds. Recommended for correctness. Default: True.

    Returns:
        tuple: (updated_dataset, write_result) where updated_dataset is the potentially
            resized dataset and write_result is the tensorstore write operation result.

    Raises:
        ValueError: If validation fails (negative offsets, etc.).

    Examples:
        Write 2D slice to 3D dataset at z=10 with offsets:
        >>> ds, result = write_ndarray(dataset, img, 10, offsets=[50, 100])

        Write 4D flow [4, y, x] to [z, c, y, x] at z=5:
        >>> ds, result = write_ndarray(flow_ds, flow, 5)

        Write 2x4 matrix to [z, a, b] at z=3:
        >>> ds, result = write_ndarray(trsf_ds, matrix, 3)
    '''
    # Resolve dataset to get fresh metadata
    if resolve:
        dataset = dataset.resolve().result()

    # Handle offsets
    if xy_offset is not None and any(o < 0 for o in xy_offset):
        raise ValueError(f'Offsets must be non-negative, got {xy_offset}')

    # Determine indexing based on array dimensionality
    if arr.ndim == 2:
        # Case 1: 2D array -> 3D dataset [z, y, x] OR matrix dataset [z, a, b]
        y, x = arr.shape
        x_off, y_off = xy_offset if xy_offset is not None else [0,0]

        # Calculate required bounds
        new_max = np.array([z+1, y+y_off, x+x_off], dtype=int)
        current_max = np.array(dataset.domain.exclusive_max, dtype=int)

        if np.any(current_max < new_max):
            new_max = np.max([current_max, new_max], axis=0)
            dataset = dataset.resize(exclusive_max=new_max, expand_only=True).result()

        # Write with slice notation for 3D indexing
        write_result = dataset[z:z+1, y_off:y+y_off, x_off:x+x_off].write(arr).result()

    elif arr.ndim == 3:
        # Case 2: 3D array [c, y, x] -> 4D dataset [z, c, y, x]
        c, y, x = arr.shape

        # Calculate required bounds
        new_max = np.array([z+1, c, y, x], dtype=int)
        current_max = np.array(dataset.domain.exclusive_max, dtype=int)

        if np.any(current_max < new_max):
            new_max = np.max([current_max, new_max], axis=0)
            dataset = dataset.resize(exclusive_max=new_max, expand_only=True).result()

        # Write directly at z-index (no slice notation for first dim)
        write_result = dataset[z, :, :y, :x].write(arr).result()

    else:
        raise ValueError(f'Unsupported array dimensionality: {arr.ndim}. Expected 2D or 3D.')

    return dataset, write_result


def write_ndarray_with_mask(
    dataset: ts.TensorStore,
    arr: np.ndarray,
    z: int,
    mask: Optional[np.ndarray] = None,
    xy_offset: Optional[List[int]] = None
) -> tuple:
    '''Write array to dataset with optional masking to preserve existing data.

    This function reads existing data from the dataset, applies the mask to merge new
    and existing values, then writes back. Useful when writing to a slice that was processed 
    with an overlapping dataset.

    Args:
        dataset (tensorstore.TensorStore): Tensorstore dataset to write to.
        arr (np.ndarray): Array to write (typically 2D [y, x]).
        z (int): Z-index where to write.
        mask (np.ndarray or None): Boolean mask indicating which pixels to update
            (True = write new value, False = preserve existing value). Must have
            same shape as arr. If None, writes entire array (equivalent to write_ndarray).
        xy_offset (list of int or None): Offsets for non-z spatial dimensions.

    Returns:
        tuple: (dataset, write_result) where dataset is the potentially resized dataset
            and write_result is the tensorstore write operation result.

    Raises:
        ValueError: If mask shape doesn't match arr shape.

    Examples:
        Only write pixels where mask is True:
        >>> ds, result = write_with_mask(dataset, aligned, z=10, mask=valid_mask,
        ...                               offsets=[50, 100])

        Write entire array (mask=None, equivalent to write_ndarray):
        >>> ds, result = write_with_mask(dataset, img, z=5, offsets=[0, 0])
    '''

    # If no mask, just use write_ndarray
    if mask is None:
        return write_ndarray(dataset, arr, z, xy_offset=xy_offset, resolve=True)

    # Validate mask shape
    if mask.shape != arr.shape:
        raise ValueError(f'Mask shape {mask.shape} must match array shape {arr.shape}')

    # Only supports 2D arrays currently
    if arr.ndim != 2:
        raise ValueError(f'write_with_mask currently only supports 2D arrays, got {arr.ndim}D')

    # Resolve dataset to get fresh metadata
    dataset = dataset.resolve().result()

    y, x = arr.shape
    x_off, y_off = xy_offset if xy_offset is not None else [0,0]

    # Ensure dataset is large enough (resize if needed, but don't write yet)
    new_max = np.array([z+1, y+y_off, x+x_off], dtype=int)
    current_max = np.array(dataset.domain.exclusive_max, dtype=int)

    if np.any(current_max < new_max):
        new_max = np.max([current_max, new_max], axis=0)
        dataset = dataset.resize(exclusive_max=new_max, expand_only=True).result()

    # Read existing data (this will be zeros if slice doesn't exist yet)
    og_data = dataset[z, y_off:y+y_off, x_off:x+x_off].read().result()

    # Merge: keep existing data where mask is False, use new data where mask is True
    og_data[mask] = arr[mask]

    # Write merged data
    write_result = dataset[z, y_off:y+y_off, x_off:x+x_off].write(og_data).result()

    return dataset, write_result


def write_data(
    dataset: ts.TensorStore,
    arr: np.ndarray,
    z: int,
    xy_offset: Optional[np.ndarray] = None,
    preserve_mask: np.ndarray = None,
    downsample_factor: Optional[float] = 1.0,
    resolve: bool = True
    ) -> tuple:
    '''Write array data to a TensorStore dataset with optional downsampling and masking.

    Parameters
    ----------
    dataset : ts.TensorStore
        Target TensorStore dataset to write to.
    arr : np.ndarray
        Array data to write. Must be 2D for image data.
    z : int
        Z-index (slice number) where data should be written.
    xy_offset : Optional[np.ndarray], default=None
        [x, y] offset for positioning data within the slice. If None, writes at [0, 0].
    preserve_mask : np.ndarray, default=None
        Boolean mask indicating which pixels to write. If provided, existing data
        in the dataset is preserved where mask is False. If None, entire array is written.
    downsample_factor : Optional[float], default=1.0
        Factor to downsample array before writing (must be <= 1.0). Values < 1.0
        will resample the array and offset accordingly.
    resolve : bool, default=True
        Whether to resolve the dataset before writing (only used when preserve_mask is None).

    Returns
    -------
    tuple
        (dataset, write_result) - The updated dataset and TensorStore write result.

    Raises
    ------
    ValueError
        If downsample_factor > 1.0 (upsampling not supported).
    '''
    if downsample_factor > 1:
        raise ValueError('Downsample factor cannot be higher than 1 (upsampling).')
    elif downsample_factor < 1:
        arr = resample(arr, downsample_factor)
        xy_offset = np.round(xy_offset * downsample_factor).astype(int)

        if preserve_mask is not None:
            preserve_mask = resample(preserve_mask, downsample_factor)

    if preserve_mask is not None:
        return write_ndarray_with_mask(dataset, arr, z, preserve_mask, xy_offset)

    return write_ndarray(dataset, arr, z, xy_offset, resolve)


# READ
def find_ref_slice(dataset: ts.TensorStore, z: Optional[int] = None,
                   reverse: bool = False, max_depth: Optional[int] = float('inf')) -> tuple:
    '''Find first or last non-empty slice of an image stack.

    Searches for a slice that contains non-zero values, starting from a given z-index
    or from the beginning/end of the stack.

    Args:
        dataset (tensorstore.TensorStore): A dataset containing the image data.
        z (int or None): Z index to start from (axis=0). If None, will start from the
            beginning (if reverse=False) or end (if reverse=True). Default: None.
        reverse (bool): Search direction. If False, searches forward (increasing z).
            If True, searches backward (decreasing z). Default: False.
        max_depth (int or float('inf')): Maximum number of slices to visit before 
            giving up the search. Default: float('inf') (no max depth)

    Returns:
        tuple: (image, z_index) where image is a 2D np.ndarray and z_index is the
            corresponding z coordinate.

    Raises:
        IndexError: If no non-empty slice is found within dataset bounds.

    Note:
        This function may hang if the entire dataset contains only zeros.
    '''
    increment = -1 if reverse else 1

    if z is None:
        z = dataset.domain.exclusive_max[0] - 1 if reverse else dataset.domain.inclusive_min[0]

    z_min = dataset.domain.inclusive_min[0]
    z_max = dataset.domain.exclusive_max[0] - 1

    img = dataset[z].read().result()

    # Add bounds checking to prevent infinite loop
    count = 0
    while not img.any():
        if count >= max_depth:
            raise IndexError(f'No non-empty slice found in dataset before reaching max search depth (searched z range: {z_min} to {z_max})')
        z += increment
        if z < z_min or z > z_max:
            raise IndexError(f'No non-empty slice found in dataset (searched z range: {z_min} to {z_max})')
        img = dataset[z].read().result()
        count += 1

    return img, z


def get_data_samples(dataset: ts.TensorStore, step_slices: int,
                     yx_target_resolution: Union[List[float], np.ndarray]) -> np.ndarray:
    '''Sample slices from a dataset at regular intervals with optional resampling to a target resolution.

    Extracts slices at regular z-intervals, skipping empty slices, and resamples them
    to match a target resolution if needed.

    Args:
        dataset (tensorstore.TensorStore): 3D dataset to sample from.
        step_slices (int): Number of slices to skip between samples. A value of 1
            means every slice, 2 means every other slice, etc.
        yx_target_resolution (list or np.ndarray): Target YX resolution in nanometers.
            Should be a 1D array or list of length 2: [y_resolution, x_resolution].

    Returns:
        np.ndarray: 3D array of sampled slices with shape [n_samples, y, x], where
            spatial dimensions match the target resolution.

    Raises:
        RuntimeError: If dataset pixel size is higher (worse) than target resolution,
            as upsampling would be required.
        KeyError: If dataset attributes don't contain 'resolution' field.
        IndexError: If unable to find non-empty slices.

    Note:
        - Empty slices (all zeros) are automatically skipped by advancing to the next slice
        - Resolution downsampling uses cv2.resize for interpolation
        - The last slice is always included in the sample
    '''
    resolution = np.array(get_store_attributes(dataset)['resolution'])[1:]
    yx_target_resolution = np.asarray(yx_target_resolution)

    z_min = dataset.domain.inclusive_min[0]
    z_max = dataset.domain.exclusive_max[0] - 1

    z_list = np.arange(z_min, z_max + 1, step_slices)
    # Ensure last slice is included
    if z_max not in z_list:
        z_list = np.append(z_list, z_max)

    data = []
    for z in z_list:
        arr = dataset[z].read().result()

        # Skip empty slices
        while not arr.any():
            z += 1
            if z > z_max:
                raise IndexError(f'No non-empty slice found starting from z={z_list[len(data)]}')
            arr = dataset[z].read().result()

        # Resample if needed
        if np.any(resolution < yx_target_resolution):
            # Downsample (resolution is better than target)
            ratio, _ = resolution / yx_target_resolution
            arr = resample(arr, ratio)
        elif np.any(resolution > yx_target_resolution):
            # Would require upsampling - not supported
            raise RuntimeError(
                f'Dataset resolution ({resolution.tolist()}) is lower quality than '
                f'target resolution ({yx_target_resolution.tolist()}). Upsampling not supported.'
            )

        data.append(arr)

    return np.array(data)