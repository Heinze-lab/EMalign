import numpy as np
from scipy import ndimage

from ...arrays.utils import resample

def compute_range_mask(data, filter_size, range_limit):

    '''
    Compute a mask keeping in regions of the data with enough range of variation.
    '''

    mask = (ndimage.maximum_filter(data, filter_size) 
            - ndimage.minimum_filter(data, filter_size)
            ) < range_limit
    return mask


def compute_greyscale_mask(data, background_value=0, downsample_factor=10):
    ratio = 1 / downsample_factor

    # Downsample greyscale, then threshold
    small = resample(data, ratio)
    mask_small = small > background_value
    del small

    structure = ndimage.generate_binary_structure(2, 1)
    labels, num_labels = ndimage.label(mask_small, structure=structure)

    if num_labels == 0:
        return np.zeros(data.shape, dtype=bool)

    component_sizes = np.bincount(labels.ravel())[1:]
    largest_component = np.argmax(component_sizes) + 1
    mask_small = labels == largest_component
    del labels

    mask_small = ndimage.binary_fill_holes(mask_small)
    struct_elem = ndimage.generate_binary_structure(2, 1)
    struct_elem = ndimage.iterate_structure(struct_elem, 2)
    mask_small = ndimage.binary_opening(mask_small, structure=struct_elem)
    mask_small = ndimage.binary_closing(mask_small, structure=struct_elem)

    # Upsample back — nearest-neighbour preserves hard edges cleanly
    return resample(mask_small, downsample_factor)

def mask_to_bbox(mask):
    y = np.any(mask, axis=1)
    x = np.any(mask, axis=0)
    ymin, ymax = np.where(y)[0][[0, -1]]
    xmin, xmax = np.where(x)[0][[0, -1]]
    return ymin, ymax + 1, xmin, xmax + 1