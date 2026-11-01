import os
import numpy as np

from emalign.arrays.utils import resample
from emalign.io.process.mask import compute_greyscale_mask
from emalign.io.store import find_ref_slice, get_store_attributes
from emalign.arrays.warp_tiled import warp_affine_auto
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

def warp_slice(s, M, canvas_shape):
    '''Warp a loaded slice + mask into the flow canvas.'''
    if s['empty']:
        return s
    img = warp_affine_auto(s['img'], M[:2], canvas_shape)
    mask = warp_affine_auto(s['mask'].astype(np.uint8), M[:2], canvas_shape).astype(bool)
    return s | {'img': img, 'mask': mask}

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

def has_valid_transform(dataset_config):
    '''Check that a dataset has ran through the transform step and has valid transform'''
    zarr_path = get_zarr_root(dataset_config['destination_path'])
    transform_path = os.path.join(zarr_path, 'z_intermediate', 'transform', dataset_config['dataset_name'])
    trsf_attrs = get_store_attributes(transform_path)
    if trsf_attrs is None:
        return False
    return bool(trsf_attrs.get('complete', False) and trsf_attrs.get('any_valid', False))