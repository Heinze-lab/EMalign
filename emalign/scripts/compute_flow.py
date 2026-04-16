import argparse
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'cuda_async'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
# os.environ['OMP_NUM_THREADS'] = '4'
# os.environ['MKL_NUM_THREADS'] = '4'

import cv2
import json
import numpy as np
import logging
import tensorstore as ts

from connectomics.common import bounding_box
from tqdm import tqdm

from sofima import mesh
from sofima.warp import ndimage_warp

from ..align_z.align_z import compute_flow_dataset, get_inv_map
from ..io.store import find_ref_slice, open_store, set_store_attributes, get_store_attributes, write_data
from ..arrays.utils import resample, pad_to_shape
from ..io.progress import get_mongo_client, get_mongo_db, wipe_progress, check_progress, log_progress
from ..io.process.mask import compute_greyscale_mask, mask_to_bbox


def fun(destination_path,
        dataset_path,
        dataset_name,
        z_offset,
        scale, 
        flow_config,
        warp_config,
        ref_slice,
        ref_slice_mask,
        yx_target_resolution,
        reference_path=None,
        reference_offset=0,
        bbox_ref = None,
        mesh_config={},
        project_name='OV',
        mongodb_config_filepath=None,
        local_z_min=None,
        local_z_max=None,
        xy_offset=[0,0],
        ignore_slices_flow=[],
        save_downsampled=1,
        overwrite=False,
        wipe_progress_flag=False,
        reverse_order=False,
        num_workers=10):
    
    if reverse_order:
        raise NotImplementedError('Processing a stack in reverse is not implemented yet.')
    
    if isinstance(yx_target_resolution, list):
        assert yx_target_resolution[0] == yx_target_resolution[1], 'Only supports equal resolution in X and Y'
        yx_target_resolution = yx_target_resolution[0]
    
    client = get_mongo_client(mongodb_config_filepath)
    db = get_mongo_db(client, project_name)

    # Flow parameters
    patch_size    = flow_config['patch_size'] 
    stride        = flow_config['stride'] 
    max_deviation = flow_config['max_deviation']
    max_magnitude = flow_config['max_magnitude']

    # Open input dataset
    dataset_path = os.path.abspath(dataset_path)
    dataset = open_store(dataset_path, mode='r', dtype=ts.uint8)

    # Keep within bounds
    original_shape = dataset.shape
    if local_z_min is not None and local_z_max is not None:
        dataset = dataset[local_z_min: local_z_max]

    dataset_mask = open_store(os.path.abspath(dataset_path) + '_mask', mode='r', dtype=ts.bool, allow_missing=True)
    dataset_mask = dataset_mask[dataset.domain] if dataset_mask is not None else dataset_mask

    # Make the resolution match the target
    attrs = get_store_attributes(dataset)
    res = attrs['resolution'][-1]
    if yx_target_resolution is None:
        target_scale = 1
    else:
        target_scale = res/yx_target_resolution
        assert target_scale >= 1, f'Target resolution ({yx_target_resolution}) must be lower than or equal to current dataset resolution ({res}) to avoid data loss.'
    
    #---------- Open reference stack if relevant ----------#
    if reference_path is not None:
        reference = open_store(reference_path, mode='r', dtype=ts.uint8)
        ref_res = get_store_attributes(reference)['resolution']
        ref_scale = ref_res[-1] / yx_target_resolution
    else:
        # No reference is provided, we will use the destination and the dataset itself
        reference = None
        ref_scale = 1

    #---------- First slice ----------#
    if ref_slice is None and reference is None and dataset.shape[0] == 1:
        # Very first image, no reference, only one image -> no flow computed, early exit
        return True
    
    # Compute flow
    flow, ds_flow, transform, bbox_ref = compute_flow_dataset(dataset=dataset,
                                                              original_shape=original_shape, 
                                                              ignore_slices=ignore_slices_flow,
                                                              scale=scale, 
                                                              patch_size=patch_size, 
                                                              stride=stride, 
                                                              db=db,
                                                              destination_path=os.path.dirname(os.path.abspath(destination_path)),
                                                              dataset_mask=dataset_mask,
                                                              reference_dataset=reference,
                                                              reference_offset=reference_offset,
                                                              bbox_ref=bbox_ref,
                                                              ref_slice=ref_slice,
                                                              ref_slice_mask=ref_slice_mask,
                                                              target_scale=target_scale,
                                                              ref_scale=ref_scale,
                                                              z_offset=z_offset)
    
    
    # Clean and combine flow fields
    # TODO: save this to file?
    clean_flow = combine_flow(flow=flow, 
                              ds_flow=ds_flow,
                              stride=stride,
                              patch_size=patch_size,
                              max_magnitude=max_magnitude,
                              max_deviation=max_deviation,
                              ds_scale=scale,
                              dataset_name=dataset_name)
    return clean_flow