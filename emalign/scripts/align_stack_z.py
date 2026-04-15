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


logging.basicConfig(level=logging.INFO)
logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('jax._src.xla_bridge').setLevel(logging.WARNING)

# TODO: save PR metric for optic flow to highlight slices where alignment might not be good

def align_stack_z(destination_path,
                  dataset_path,
                  dataset_name,
                  z_offset,
                  scale, 
                  flow_config,
                  warp_config,
                  first_slice,
                  yx_target_resolution,
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

    if wipe_progress_flag:
        logging.info(f'Wiping progress for stack: {dataset_name}')
        wipe_progress(db, dataset_name)
        
        # Since we wipe progress, we also want to make sure that all old data will be overwritten properly
        overwrite = True

    #---------- Prepare variables ----------#
    # Flow parameters
    patch_size    = flow_config['patch_size'] 
    stride        = flow_config['stride'] 
    max_deviation = flow_config['max_deviation']
    max_magnitude = flow_config['max_magnitude']

    # Mesh opti parameters
    mesh_config_args = { # Default config
        'dt': 0.001,
        'gamma': 0.5,
        'k0': 0.01,
        'k': 0.4,
        'num_iters': 1000,
        'max_iters': 100000,
        'stop_v_max': 0.01,
        'dt_max': 1000,
        'start_cap': 0.1,
        'final_cap': 1.0,
        'prefer_orig_order': True,
        'remove_drift': False # for some reason, setting this to True actually introduces drift
    }
    mesh_config_args.update(mesh_config) # Update with user-defined values if they exist
    mesh_config_args['stride'] = [stride, stride]

    # Warp parameters
    work_size = warp_config['work_size']   
    overlap   = warp_config['overlap'] 
    
    # Paths
    destination_path = os.path.abspath(destination_path)
    dataset_path = os.path.abspath(dataset_path)
    
    # Open input dataset
    dataset = open_store(dataset_path, mode='r', dtype=ts.uint8)

    # Check whether stack was processed
    attrs = get_store_attributes(dataset)
    if attrs.get('z_aligned', False) == True and not overwrite:
        logging.info(f'Dataset {dataset_name} was already processed and will be skipped.')
        return False
    
    # Keep within bounds
    original_shape = dataset.shape
    if local_z_min is not None and local_z_max is not None:
        dataset = dataset[local_z_min: local_z_max]

    dataset_mask = open_store(os.path.abspath(dataset_path) + '_mask', mode='r', dtype=ts.bool, allow_missing=True)
    dataset_mask = dataset_mask[dataset.domain] if dataset_mask is not None else dataset_mask

    # Make the resolution match the target
    res = attrs['resolution'][-1]
    if yx_target_resolution is None:
        target_scale = 1
    else:
        target_scale = res/yx_target_resolution
        assert target_scale >= 1, f'Target resolution ({yx_target_resolution}) must be lower than or equal to current dataset resolution ({res}) to avoid data loss.'
    logging.info(f'Target scale ({dataset_name}): {target_scale}')
    
    #---------- Open destination(s) ----------#
    destination = open_store(destination_path, mode='r+', dtype=ts.uint8)
    destination_mask = open_store(destination_path + '_mask', mode='r+', dtype=ts.bool)

    ds_destination = None
    if save_downsampled > 1:
        ds_output_path, project_name_from_path = destination_path.rsplit('/', maxsplit=1)
        ds_output_path = os.path.join(ds_output_path, f'{save_downsampled}x_' + project_name_from_path)
        ds_destination = open_store(ds_output_path, mode='r+', dtype=ts.uint8)
                
    #---------- Compute flow ----------#
    if first_slice is not None:
        first_slice, z = find_ref_slice(destination, 
                                        int(first_slice), 
                                        reverse=True)
        first_slice_mask = destination_mask[z].read().result()
    elif dataset.shape[0] > 1:
        # More than one image
        first_slice_mask = None
    else:
        # No need to compute flow because we only have one image and it is the first one
        data = dataset[dataset.domain.inclusive_min[0]].read().result() # First slice within bounds
        data = resample(data, target_scale)
        
        if dataset_mask is not None:
            data_mask = dataset_mask[dataset_mask.domain.inclusive_min[0]].read().result()
            data_mask = resample(data_mask, target_scale)
        else:
            data_mask = compute_greyscale_mask(data)

        # To full resolution destination
        write_data(destination,
                   data,
                   z_offset,
                   np.abs(xy_offset),
                   preserve_mask = None,
                   resolve = True)
        # To destination mask
        write_data(destination_mask,
                   data_mask,
                   z_offset,
                   np.abs(xy_offset),
                   preserve_mask = None,
                   resolve = True)
        # To downsampled destination
        if save_downsampled != 1:
            write_data(ds_destination,
                       data,
                       z_offset,
                       np.abs(xy_offset),
                       preserve_mask = None,
                       downsample_factor = 1/save_downsampled,
                       resolve = True)

        attrs['z_aligned'] = True
        set_store_attributes(dataset, attrs)
        return True
        
    # Compute flow and save to file
    flow, transform = compute_flow_dataset(dataset=dataset,
                                           original_shape=original_shape, 
                                           ignore_slices=ignore_slices_flow,
                                           scale=scale, 
                                           patch_size=patch_size, 
                                           stride=stride, 
                                           max_deviation=max_deviation,
                                           max_magnitude=max_magnitude,
                                           db=db,
                                           destination_path=os.path.dirname(os.path.abspath(destination_path)),
                                           ref_slice=first_slice,
                                           ref_slice_mask=first_slice_mask,
                                           target_scale=target_scale,
                                           z_offset=z_offset)

    #---------- Compute mesh ----------#
    out_path_meshes = os.path.join(os.path.dirname(destination_path), 
                                   'z_intermediate', 
                                   'inverse_map')
    os.makedirs(out_path_meshes, exist_ok=True)
    inv_map_path = os.path.join(out_path_meshes, dataset_name)
    mesh_config = mesh.IntegrationConfig(**mesh_config_args)

    if not os.path.exists(inv_map_path):
        inv_map, _ = get_inv_map(flow, stride, dataset_name, mesh_config)

        # Create and write inv_map tensorstore
        inv_map_store = open_store(
            inv_map_path,
            mode='a',
            dtype=ts.float64,
            shape=list(inv_map.shape),  # [2, z, y, x]
            chunks=[2, 1, min(512, inv_map.shape[2]), min(512, inv_map.shape[3])],
            axis_labels=['c', 'z', 'y', 'x']
        )
        inv_map_store[:].write(inv_map).result()

        # Save mesh integration config as attributes
        set_store_attributes(inv_map_store, {
            'mesh_config': mesh_config_args,
            'stride': stride,
            'dataset_name': dataset_name,
            'description': 'Inverse displacement map from mesh relaxation'
        })
    else:
        # Load from existing tensorstore, but first validate parameters
        inv_map_store = open_store(inv_map_path, mode='r')
        stored_attrs = get_store_attributes(inv_map_store)

        # Check if mesh_config and stride match current settings
        params_match = stored_attrs.get('mesh_config') == mesh_config_args

        if not params_match and overwrite:
            logging.warning(f'Stored mesh parameters do not match current settings. Recomputing inverse map.')
            logging.info(f'Stored mesh_config: {stored_attrs.get("mesh_config")}')
            logging.info(f'Current mesh_config: {mesh_config_args}')
            logging.info(f'Stored stride: {stored_attrs.get("stride")}, Current stride: {stride}')

            # Recompute with current parameters
            inv_map, _ = get_inv_map(flow, stride, dataset_name, mesh_config)

            # Overwrite existing store
            inv_map_store = open_store(
                inv_map_path,
                mode='w',
                dtype=ts.float64,
                shape=list(inv_map.shape),
                chunks=[2, 1, min(512, inv_map.shape[2]), min(512, inv_map.shape[3])],
                axis_labels=['c', 'z', 'y', 'x']
            )
            inv_map_store[:].write(inv_map).result()

            # Update attributes with current settings
            set_store_attributes(inv_map_store, {
                'mesh_config': mesh_config_args,
                'stride': stride,
                'dataset_name': dataset_name,
                'description': 'Inverse displacement map from mesh relaxation'
            })
        elif params_match:
            logging.info('Loading existing mesh')
            # Parameters match, safe to use cached inverse map
            inv_map = inv_map_store[:].read().result()
        else:
            raise RuntimeError('Stored mesh parameters do not match current settings but overwrite is set to False.')

    #---------- Render data ----------#
    # Get first slice
    if first_slice is None:
        # First slice to write is the first slice of the dataset, untouched but padded
        # Then we warp the rest from the next slice
        first, z = find_ref_slice(dataset, 
                                  dataset.domain.inclusive_min[0], 
                                  reverse=False)
        first = resample(first, target_scale)
        if dataset_mask is not None:
            first_mask = dataset_mask[z].read().result()
            first_mask = resample(first_mask, target_scale)
        else:
            first_mask = compute_greyscale_mask(first)

        # To full resolution destination
        write_data(destination,
                   first,
                   z + z_offset - dataset.domain.inclusive_min[0],
                   np.abs(xy_offset),
                   preserve_mask = None,
                   resolve = True)
        # To destination mask
        write_data(destination_mask,
                   first_mask,
                   z + z_offset - dataset.domain.inclusive_min[0],
                   np.abs(xy_offset),
                   preserve_mask = None,
                   resolve = True)
        # To downsampled destination
        if save_downsampled != 1:
            write_data(ds_destination,
                       first,
                       z + z_offset - dataset.domain.inclusive_min[0],
                       np.abs(xy_offset),
                       preserve_mask = None,
                       downsample_factor = 1/save_downsampled,
                       resolve = True)
        
        start = z + 1
    else:
        # All slices have to be warped to match the last slice of the previous stack
        start = dataset.domain.inclusive_min[0]

    # Start alignment
    output_shape = np.max(transform[:,:,-1], axis=0).astype(int)
    skipped = 0
    empty = 0
    step_name = 'render_z'
    for z in tqdm(range(start, dataset.domain.exclusive_max[0]),
                    position=0,
                    desc=f'{dataset_name}: Rendering aligned slices',
                    dynamic_ncols=True,
                    leave=True):

        if check_progress(db, dataset_name, step_name, z):
            skipped += 1 # Assume skipped slices are logged correctly
            continue

        # Load data
        data = dataset[z].read().result()
        global_z = z + z_offset - dataset.domain.inclusive_min[0]

        if not data.any():
            # If empty slice, skip and go to next z
            empty += 1
            skipped += 1
            metadata = {'empty_slice': True}
            log_progress(db, dataset_name, step_name, z, global_z, metadata)
            continue

        # Resample if needed (target_scale != 1)
        data = resample(data, target_scale)

        # Load or compute mask
        if dataset_mask is not None:
            data_mask = dataset_mask[z].read().result()
            data_mask = resample(data_mask, target_scale)
        else:
            data_mask = compute_greyscale_mask(data)

        # Transform data
        M = transform[z,:,:-1]
        data = cv2.warpAffine(data, M, output_shape[::-1])
        data_mask = cv2.warpAffine(data_mask.astype(np.uint8), M, output_shape[::-1]).astype(bool)
        data_bbox = bounding_box.BoundingBox(start=(0, 0, 0), 
                                             size=(data.shape[-1], data.shape[-2], 1))
        
        # Warp data in parallel. warp_subvolume uses one thread per image so we use ndimage_wrap instead
        inv_z = z - start + 1
        aligned = ndimage_warp(
                        data, 
                        inv_map[:, inv_z, ...], 
                        stride=(stride, stride),
                        work_size=(work_size, work_size),
                        overlap=(overlap,overlap),
                        image_box=data_bbox,
                        parallelism=num_workers
                    )
        aligned_mask = ndimage_warp(
                        data_mask, 
                        inv_map[:, inv_z, ...], 
                        stride=(stride, stride),
                        work_size=(work_size, work_size),
                        overlap=(overlap,overlap),
                        image_box=data_bbox,
                        parallelism=num_workers
                    )
        # Mask gets full of holes because of warping
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        aligned_mask = cv2.morphologyEx(aligned_mask.astype(np.uint8),cv2.MORPH_CLOSE,kernel).astype(bool)

        if overwrite:
            # There may be data written to this slice so let's make sure it is overwritten
            y1 = x1 = 0
            y2, x2 = destination.shape[1:]
            aligned = pad_to_shape(aligned, destination.shape[1:])
            aligned_mask = pad_to_shape(aligned_mask, destination.shape[1:])
            write_mask = None # This means we write everything even black space
        else:
            # Writing bounding box
            y1, y2, x1, x2 = mask_to_bbox(aligned_mask)
            write_mask = aligned_mask[y1:y2, x1:x2]
        
        # Write data
        offset = np.abs(xy_offset) + np.array([x1, y1])
        
        # To full resolution destination
        write_data(destination,
                   aligned[y1:y2, x1:x2], # Only write in the bounding box where the data is
                   global_z, # z_offset relates to original minimum
                   offset, # Only write in the bounding box where the data is
                   preserve_mask = write_mask, # Mask where to write the data
                   resolve = True)
        # To destination mask
        write_data(destination_mask,
                   aligned_mask[y1:y2, x1:x2], 
                   global_z, 
                   offset, 
                   preserve_mask = write_mask, 
                   resolve = True)
        # To downsampled destination
        if save_downsampled != 1:
            write_data(ds_destination,
                       aligned[y1:y2, x1:x2], 
                       global_z, 
                       offset, #
                       preserve_mask = write_mask, 
                       downsample_factor = 1/save_downsampled,
                       resolve = True)
        
        # Log progress
        metadata = {
            'empty_slice': False,
            'warp_config': warp_config,
            'mesh_config': mesh_config_args,
            'overwrite': overwrite,
            'bbox': [int(y1), int(y2), int(x1), int(x2)]
        }
        log_progress(db, dataset_name, step_name, global_z, z, metadata)
    logging.info(f'{dataset_name}: Done.')
    logging.info(f'Empty slices: {empty}')
    logging.info(f'Skipped already processed slices: {skipped}')

    # Add an attribute to keep track of what datasets have been aligned already
    attrs['z_aligned'] = True 
    set_store_attributes(dataset, attrs)

    return True


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Align a stack in Z.')
    parser.add_argument('config_file', type=str, help='Path to the JSON configuration file.')
    parser.add_argument('--wipe-progress', action='store_true', help='Wipe progress for the specified stack before starting.')

    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = json.load(f)

    project_name = config.get('project_name')
    if not project_name:
        project_name = os.path.basename(config['destination_path']).rstrip('.zarr')

    mongodb_config_filepath = config.get('mongodb_config_filepath')

    config['project_name'] = project_name
    config['mongodb_config_filepath'] = mongodb_config_filepath
    config['wipe_progress_flag'] = args.wipe_progress

    align_stack_z(**config)
