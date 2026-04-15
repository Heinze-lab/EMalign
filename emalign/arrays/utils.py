import cv2
import numpy as np

def resample(array, ratio):
    '''
    Resize an array by a scaling ratio using OpenCV interpolation.

    Parameters:
    -----------
    array : np.ndarray
        Input array to resize
    ratio : float
        Scaling factor. Values < 1 downsample, values > 1 upsample.
        If 1, returns original array without change.

    Returns:
    --------
    np.ndarray
        Resized array

    Raises:
    -------
    ValueError
        If ratio is not positive

    Notes:
    ------
    - Uses cv2.resize with default interpolation (bilinear)
    - Boolean arrays are converted to uint8, resized, then converted back to bool
    - Returns original array without copying if ratio == 1
    '''
    if ratio <= 0:
        raise ValueError(f"Ratio must be positive, got {ratio}")

    if ratio == 1:
        return array

    if array.dtype == bool:
        # cv2.resize doesn't want boolean
        array = cv2.resize(array.astype(np.uint8), None, fx=ratio, fy=ratio).astype(bool)
    else:
        array = cv2.resize(array, None, fx=ratio, fy=ratio)
    return array

# PAD
def xy_offset_to_pad(offset):
    '''
    Convert (x, y) offset values to numpy padding specification.

    Parameters:
    -----------
    offset : tuple or array-like
        (x, y) offset values. Can be positive or negative.

    Returns:
    --------
    np.ndarray
        2x2 padding array suitable for np.pad, with format [[top, bottom], [left, right]]:
        - Positive y offset: pads bottom (pad[0][1])
        - Negative y offset: pads top (pad[0][0])
        - Positive x offset: pads right (pad[1][1])
        - Negative x offset: pads left (pad[1][0])

    Examples:
    ---------
    >>> xy_offset_to_pad((10, 5))
    array([[0, 5],
           [0, 10]])

    >>> xy_offset_to_pad((-10, -5))
    array([[5, 0],
           [10, 0]])
    '''
    pad = np.zeros([2, 2], dtype=int)
    x, y = [int(i) for i in offset]

    if y > 0:
        pad[0][1] = y
    else:
        pad[0][0] = abs(y)

    if x > 0:
        pad[1][1] = x
    else:
        pad[1][0] = abs(x)

    return pad


def pad_to_shape(array, target_shape, direction=None, axis=None, pad_value=0):
    '''
    Pad an array to match a shape along specified axes. If the target shape is smaller 
    than the array's shape in a dimension, no padding is added to that dimension.
    
    Parameters:
    -----------
    array : np.ndarray
        Input array to pad
    target_shape : list or tuple
        Target shape for the specified axes
    direction : list
        Direction to pad for each axis (0=left/start, 1=right/end)
    axis : list
        Axes to apply padding to (supports negative indexing)
    pad_value : 
        Value to use for padding
        
    Returns:
    --------
    np.ndarray
        Padded array
    '''
    # Convert axis to positive indices and validate
    if axis is not None:
        axis = np.array(axis)
        axis = np.where(axis < 0, array.ndim + axis, axis)
    else:
        axis=np.arange(0, array.ndim)

    if direction is None:
        direction = np.ones_like(axis)

    # Validate inputs
    if len(target_shape) != len(axis):
        raise ValueError(f'Target_shape length ({len(target_shape)}) must match axis length ({len(axis)})')
    if len(direction) != len(axis):
        raise ValueError(f'Direction length ({len(direction)}) must match axis length ({len(axis)})')
    if not all(0 <= ax < array.ndim for ax in axis):
        raise ValueError(f'Axis indices out of bounds. Valid range: [0, {array.ndim}), got: {axis}')
    
    # Calculate padding needed for each specified axis
    pad_sizes = np.array(target_shape) - np.array(array.shape)[axis]
    pad_sizes = np.max([pad_sizes, np.zeros_like(pad_sizes)], axis=0)  # No negative padding
    
    # Create padding specification for np.pad
    pad_width = np.zeros((array.ndim, 2), dtype=int)
    for ax, pad_size, dir_val in zip(axis, pad_sizes, direction):
        if pad_size > 0:
            if dir_val == 0:  # Pad at start
                pad_width[ax, 0] = pad_size
            else:  # Pad at end (default)
                pad_width[ax, 1] = pad_size
    
    return np.pad(array, pad_width, constant_values=pad_value)


def homogenize_arrays_shape(arrs, pad_value=0):
    '''
    Pad multiple arrays to have the same shape by padding each to the maximum shape.

    Parameters:
    -----------
    arrs : list of np.ndarray
        List of arrays to homogenize. All arrays must have the same number of dimensions.
    pad_value : scalar, optional
        Value to use for padding. Default is 0.

    Returns:
    --------
    np.ndarray
        Array containing all padded arrays stacked along axis 0.
        Shape: (len(arrs), *max_shape)

    Notes:
    ------
    - All input arrays must have the same dtype or compatible dtypes
    - Padding is applied to the end (right/bottom) of each dimension
    - The returned array has dtype matching the input arrays

    Examples:
    ---------
    >>> arr1 = np.ones((10, 20))
    >>> arr2 = np.ones((15, 25))
    >>> result = homogenize_arrays_shape([arr1, arr2])
    >>> result.shape
    (2, 15, 25)
    '''
    max_shape = np.max([a.shape for a in arrs], axis=0)
    return np.array([pad_to_shape(a, max_shape, pad_value=pad_value) for a in arrs])


# ASSESS QUALITY
def compute_laplacian_var(arr, mask=None):
    '''
    Compute Laplacian variance as a sharpness metric.

    The Laplacian operator detects edges by computing second derivatives.
    Higher variance indicates sharper, more distinct edges.

    Parameters:
    -----------
    arr : np.ndarray
        Input image array
    mask : np.ndarray (bool), optional
        Boolean mask to restrict computation to specific regions.
        If provided, only masked pixels are used. Default: None (use all pixels).

    Returns:
    --------
    float
        Variance of the Laplacian-filtered image. Higher values indicate sharper images.

    Notes:
    ------
    - Sensitive to contrast - images with higher contrast may have higher variance
    - Uses cv2.Laplacian with CV_64F for numerical precision
    - Commonly used for autofocus and blur detection
    - Optimized: Crops to mask bounding box before filtering to reduce computation
    '''
    if mask is None:
        l = cv2.Laplacian(arr, cv2.CV_64F)
        return np.var(l)

    # cv2 needs 2D array so masking needs to happen after computation
    # Crop to relevant bbox to limit computations
    ymin, ymax, xmin, xmax = mask_to_bbox(mask)
    arr_crop = arr[ymin:ymax+1, xmin:xmax+1]
    mask_crop = mask[ymin:ymax+1, xmin:xmax+1]

    l = cv2.Laplacian(arr_crop, cv2.CV_64F)[mask_crop]
    return np.var(l)


def compute_sobel_mean(arr, mask=None):
    '''
    Compute mean Sobel gradient magnitude as a sharpness metric.

    The Sobel operator computes first derivatives in x and y directions,
    providing edge detection and sharpness estimation.

    Parameters:
    -----------
    arr : np.ndarray
        Input image array
    mask : np.ndarray (bool), optional
        Boolean mask to restrict computation to specific regions.
        If provided, only masked pixels are used. Default: None.

    Returns:
    --------
    float
        Mean gradient magnitude. Higher values indicate sharper images with stronger edges.

    Notes:
    ------
    - Uses 5x5 Sobel kernels for derivative computation
    - Combines x and y gradients via Euclidean norm: sqrt(sobel_x^2 + sobel_y^2)
    - Less sensitive to noise than Laplacian variance
    - Optimized: Crops to mask bounding box before filtering to reduce computation
    '''
    if mask is None:
        sobel_x = cv2.Sobel(arr, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(arr, cv2.CV_64F, 0, 1, ksize=5)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        return np.mean(sobel)

    # cv2 needs 2D array so masking needs to happen after computation
    # Crop to relevant bbox to limit computations
    ymin, ymax, xmin, xmax = mask_to_bbox(mask)
    arr_crop = arr[ymin:ymax+1, xmin:xmax+1]
    mask_crop = mask[ymin:ymax+1, xmin:xmax+1]

    sobel_x = cv2.Sobel(arr_crop, cv2.CV_64F, 1, 0, ksize=5)[mask_crop]
    sobel_y = cv2.Sobel(arr_crop, cv2.CV_64F, 0, 1, ksize=5)[mask_crop]
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    return np.mean(sobel)


def compute_grad_mag(arr, mask=None):
    '''
    Compute mean gradient magnitude using numpy gradients.

    This function calculates pixel-to-pixel variation as a measure of
    image sharpness and quality.

    Parameters:
    -----------
    arr : np.ndarray
        Input image array
    mask : np.ndarray (bool), optional
        Boolean mask to restrict computation to specific regions.
        If provided, only masked pixels are used. Default: None.

    Returns:
    --------
    float
        Average gradient magnitude. Higher values indicate more pixel variation
        and potentially sharper images.

    Notes:
    ------
    - Uses numpy.gradient for derivative computation (central differences)
    - Combines x and y gradients via Euclidean norm: sqrt(gx^2 + gy^2)
    - Faster than Sobel but may be more sensitive to noise
    - Optimized: Crops to mask bounding box before computing gradients
    '''
    if mask is None:
        gy, gx = np.gradient(arr)
        gnorm = np.sqrt(gx**2 + gy**2)
        return np.average(gnorm)

    # cv2 needs 2D array so masking needs to happen after computation
    # Crop to relevant bbox to limit computations
    ymin, ymax, xmin, xmax = mask_to_bbox(mask)
    arr_crop = arr[ymin:ymax+1, xmin:xmax+1]
    mask_crop = mask[ymin:ymax+1, xmin:xmax+1]

    gy, gx = np.gradient(arr_crop)
    gnorm = np.sqrt(gx**2 + gy**2)[mask_crop]
    return np.average(gnorm)


def compute_laplacian_var_diff(overlap1,
                               overlap2,
                               mask=None):

    '''
    Compute similarity metric between two arrays based on Laplacian variance.

    This function measures how well two image regions overlap by comparing their
    edge content. Regions with similar edge characteristics receive higher scores.

    Parameters:
    -----------
    overlap1 : np.ndarray
        First image region to compare
    overlap2 : np.ndarray
        Second image region to compare
    mask : np.ndarray (bool), optional
        Boolean mask to restrict comparison to specific regions. Default: None.

    Returns:
    --------
    float
        Similarity metric in range [0, 1]:
        - 1.0: Identical edge content (perfect overlap)
        - 0.0: Completely different edge content or no overlap
        - Intermediate values indicate partial similarity

    Notes:
    ------
    - Uses Laplacian variance to quantify edge content in each region
    - Computes normalized difference: 1 - |var1 - var2| / max(var1, var2)
    - Returns 1.0 if both regions have zero variance (e.g., uniform regions)
    - Returns 0.0 if one variance is zero but the other is not
    - Useful for validating image alignment and registration quality
    '''

    lap_var1 = compute_laplacian_var(overlap1, mask)
    lap_var2 = compute_laplacian_var(overlap2, mask)

    # Calculate an index of difference in edge content (variance of laplacian)
    # Between 0 and 1, low means exact same content, 1 means different
    max_var = max(lap_var1, lap_var2)
    if max_var == 0:
        return 1.0 if lap_var1 == lap_var2 else 0.0

    return 1 - abs(lap_var1 - lap_var2) / max_var