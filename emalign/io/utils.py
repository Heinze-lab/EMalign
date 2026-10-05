import os

from emalign.arrays.utils import resample
from emalign.io.process.mask import compute_greyscale_mask
from emalign.io.store import find_ref_slice
import tensorstore as ts


def get_dataset_name(dataset):
    '''Get dataset name from a path string or TensorStore object.'''
    if isinstance(dataset, str):
        return os.path.basename(os.path.abspath(dataset))
    elif isinstance(dataset, ts.TensorStore):
        return os.path.basename(os.path.abspath(dataset.kvstore.path))
    
def get_zarr_root(dataset):
    # CHANGED HERE: basename removed, the walk up needs the full path
    if isinstance(dataset, str):
        zarr_path = os.path.abspath(dataset)
    elif isinstance(dataset, ts.TensorStore):
        zarr_path = os.path.abspath(dataset.kvstore.path)
    
    while not zarr_path.endswith('.zarr'):
        if zarr_path == os.path.dirname(zarr_path):
            raise ValueError('No zarr container found in provided destination path.')
        # Get the zarr container
        zarr_path = os.path.dirname(zarr_path)
    return zarr_path 

def load_slice(dataset, dataset_mask, z, scale, compute_mask=True):
    '''Load one slice + mask and resample.'''
    img = dataset[z].read().result()
    if not img.any():
        return {'z': z, 'img': None, 'mask': None, 'empty': True}

    img = resample(img, scale)
    if dataset_mask is not None:
        mask = resample(dataset_mask[z].read().result(), scale)
    elif compute_mask:
        mask = compute_greyscale_mask(img, downsample_factor=10)
    else:
        mask = None
    return {'z': z, 'img': img, 'mask': mask, 'empty': False}


def load_reference_slice(reference, reference_mask, z, scale, reverse=False):
    '''Load the closest non-empty slice of the reference + its mask, and resample.

    Searches from z onwards (or backwards if reverse), so a reference with a
    different z resolution still yields an image.'''
    img, z_ref = find_ref_slice(reference, z, reverse=reverse)
    img = resample(img, scale)
    if reference_mask is not None:
        mask = resample(reference_mask[z_ref].read().result(), scale)
    else:
        mask = compute_greyscale_mask(img, downsample_factor=10)
    return {'z': z_ref, 'img': img, 'mask': mask, 'empty': False}