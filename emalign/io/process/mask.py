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


def compute_greyscale_mask(data, background_value=0):

    '''
    Compute a mask of the region that contains greyscale data.
    '''
   
    mask = data>background_value

    structure = ndimage.generate_binary_structure(data.ndim, 1)
    labels, num_labels = ndimage.label(mask, structure=structure)

    if num_labels != 0:
        component_sizes = np.bincount(labels.ravel())[1:]
        largest_component = np.argmax(component_sizes) + 1

        mask = labels == largest_component

        # Fill holes
        mask = ndimage.binary_fill_holes(mask)

        # Close smaller holes
        struct_elem = ndimage.generate_binary_structure(data.ndim, 1)
        struct_elem = ndimage.iterate_structure(struct_elem, 2)
        mask = ndimage.binary_opening(mask, structure=struct_elem)
        mask = ndimage.binary_closing(mask, structure=struct_elem)

    return mask


def mask_to_bbox(mask):
    y = np.any(mask, axis=1)
    x = np.any(mask, axis=0)
    ymin, ymax = np.where(y)[0][[0, -1]]
    xmin, xmax = np.where(x)[0][[0, -1]]
    return ymin, ymax + 1, xmin, xmax + 1