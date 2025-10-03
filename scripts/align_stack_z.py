import inspect
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'cuda_async'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
# os.environ['OMP_NUM_THREADS'] = '4'
# os.environ['MKL_NUM_THREADS'] = '4'

import datetime
import cv2
import json
import numpy as np
import logging
import sys
import tensorstore as ts

from connectomics.common import bounding_box
from tqdm import tqdm

from sofima import mesh
from sofima.warp import ndimage_warp
from emprocess.utils.io import get_dataset_attributes, set_dataset_attributes
from emprocess.utils.mask import compute_greyscale_mask, mask_to_bbox

from emalign.align_z.align_z import compute_flow_dataset, get_inv_map_mod
from emalign.io.store import find_ref_slice
from emalign.arrays.utils import downsample


logging.basicConfig(level=logging.INFO)
logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('jax._src.xla_bridge').setLevel(logging.WARNING)

# TODO: save PR metric for optic flow to highlight slices where alignment might not be good

def align_stack_z(destination_path,
                  dataset_path, 
                  z_offset,
                  scale, 
                  flow_config,
                  mesh_config,
                  warp_config,
                  first_slice,
                  yx_target_resolution,
                  db_name,
                  local_z_min=None,
                  local_z_max=None,
                  xy_offset=[0,0],
                  ignore_slices_flow=[],
                  save_downsampled=1,
                  overwrite=False,
                  num_workers=10):
    
    if isinstance(yx_target_resolution, list):
        assert yx_target_resolution[0] == yx_target_resolution[1], 'Only supports equal resolution in X and Y'
        yx_target_resolution = yx_target_resolution[0]
    
    #---------- Prepare variables ----------#
    # Flow parameters
    patch_size    = flow_config['patch_size'] 
    stride        = flow_config['stride'] 
    max_deviation = flow_config['max_deviation']
    max_magnitude = flow_config['max_magnitude']

    # Mesh opti parameters
    k0    = mesh_config['k0'] 
    k     = mesh_config['k'] 
    gamma = mesh_config['gamma']

    # Warp parameters
    work_size = warp_config['work_size']   
    overlap   = warp_config['overlap'] 
    
    # Paths
    dataset_name = os.path.basename(os.path.abspath(dataset_path))
    destination_path = os.path.abspath(destination_path)
    dataset_path = os.path.abspath(dataset_path)
    
    # Open input dataset
    dataset = ts.open({'driver': 'zarr',
                       'kvstore': {
                             'driver': 'file',
                             'path': dataset_path,
                                  }
                      },
                      dtype=ts.uint8
                      ).result()
    
    # Keep within bounds
    original_shape = dataset.shape
    if local_z_min is not None and local_z_max is not None:
        dataset = dataset[local_z_min: local_z_max]

    try:
        dataset_mask = ts.open({'driver': 'zarr',
                                'kvstore': {
                                        'driver': 'file',
                                        'path': os.path.abspath(dataset_path) + '_mask',
                                            }
                                },
                                dtype=ts.bool
                                ).result()
        
        dataset_mask = dataset_mask[dataset.domain]
    except ValueError as e:
        if 'NOT_FOUND' in str(e):
            dataset_mask = None
        else:
            raise e
    except Exception as e:
        raise e
        
    # Check whether stack was processed
    attrs = get_dataset_attributes(dataset)
    if attrs.get('z_aligned', False) == True and not overwrite:
        logging.info(f'Dataset {dataset_name} was already processed and will be skipped.')
        return False

    # Make the resolution match the target
    res = attrs['resolution'][-1]
    if yx_target_resolution is None:
        target_scale = 1
    else:
        target_scale = res/yx_target_resolution
        assert target_scale >= 1, f'Target resolution ({yx_target_resolution}) must be lower than or equal to current dataset resolution ({res}) to avoid data loss.'
    logging.info(f'Target scale ({dataset_name}): {target_scale}')
    
    #---------- Open destination(s) ----------#
    destination = ts.open({'driver': 'zarr',
                           'kvstore': {
                                 'driver': 'file',
                                 'path': destination_path,
                                      }
                          },
                          dtype=ts.uint8
                          ).result()
    destination_mask = ts.open({'driver': 'zarr',
                                'kvstore': {
                                        'driver': 'file',
                                        'path': destination_path + '_mask',
                                            }
                                },
                                dtype=ts.bool
                                ).result()
    
    if save_downsampled > 1:
        ds_output_path, project_name = destination_path.rsplit('/', maxsplit=1)
        ds_output_path = os.path.join(ds_output_path, f'{save_downsampled}x_' + project_name)
        ds_destination = ts.open({'driver': 'zarr',
                                  'kvstore': {
                                          'driver': 'file',
                                          'path': ds_output_path,
                                              }
                                  },
                                  dtype=ts.uint8
                                  ).result()
                
    #---------- Compute flow ----------#
    if first_slice is not None:
        first_slice, z = find_ref_slice(destination, 
                                        int(first_slice), 
                                        reverse=True)
        first_slice_mask = destination_mask[z].read().result()
    elif dataset.shape[0] > 1:
        # More than one images
        first_slice_mask = None
    else:
        # No need to compute flow because we only have one image and it is the first one
        data = dataset[dataset.domain.inclusive_min[0]].read().result() # First slice within bounds
        data = downsample(data, target_scale)
        
        if dataset_mask is not None:
            data_mask = dataset_mask[dataset_mask.domain.inclusive_min[0]].read().result()
            data_mask = downsample(data_mask, target_scale)
        else:
            data_mask = compute_greyscale_mask(data)

        write_data(destination, data, z_offset, np.abs(xy_offset), save_downsampled, ds_destination)
        write_data(destination_mask, data_mask, z_offset, np.abs(xy_offset))

        attrs['z_aligned'] = True
        set_dataset_attributes(dataset, attrs)
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
                                           db_name=db_name,
                                           destination_path=os.path.dirname(os.path.abspath(destination_path)),
                                           ref_slice=first_slice,
                                           ref_slice_mask=first_slice_mask,
                                           target_scale=target_scale)

    #---------- Compute mesh ----------#
    # Elasticity ratio = k0/k (the larger the more deformation)
    # The ratio is what matters for how much the data is deformed
    # However smaller numbers might limit the necessary deformation of the mesh
    # Good values:
    # k0 = 0.01 # inter-section springs (elasticity). High k0 results in images that tend to "fold" onto themselves
    # k = 0.4 # intra-section springs (elasticity). Increase if data deforms too much
    # gamma = 0.5 # dampening factor. Increase if data drift over time
    out_path_meshes = os.path.dirname(destination_path) + '/flow_meshes'
    os.makedirs(out_path_meshes, exist_ok=True)
    inv_map_path = os.path.join(out_path_meshes, dataset_name + '.npy')
    vmax_path = os.path.join(out_path_meshes, dataset_name + '_vmax.npy')
    
    mesh_config = mesh.IntegrationConfig(dt=0.001, gamma=0.5, k0=0.1, k=0.4,
                                         stride=(stride, stride),
                                         num_iters=1000,max_iters=100000,
                                         stop_v_max=0.002, dt_max=2, 
                                         start_cap=0.1, final_cap=1.0,
                                         prefer_orig_order=True, remove_drift=True)
    
    if not os.path.exists(inv_map_path):
        inv_map, _, v_max = get_inv_map_mod(flow, stride, dataset_name, mesh_config)
        np.save(inv_map_path, inv_map)
        np.save(vmax_path, v_max)
    else:
        inv_map = np.load(inv_map_path)
    
    #---------- Render data ----------#
    # Get first slice
    if first_slice is None:
        # First slice to write is the first slice of the dataset, untouched but padded
        # Then we warp the rest from the next slice
        first, z = find_ref_slice(dataset, 
                                  dataset.domain.inclusive_min[0], 
                                  reverse=False)
        first = downsample(first, target_scale)
        if dataset_mask is not None:
            first_mask = dataset_mask[z].read().result()
            first_mask = downsample(first_mask, target_scale)
        else:
            first_mask = compute_greyscale_mask(first)

        write_data(destination, first, 
                   z + z_offset - dataset.domain.inclusive_min[0], # z_offset relates to the original minimum
                   np.abs(xy_offset), save_downsampled, ds_destination)
        write_data(destination_mask, first_mask, 
                   z + z_offset - dataset.domain.inclusive_min[0], 
                   np.abs(xy_offset))
        
        start = z + 1
    else:
        # All slices have to be warped to match the last slice of the previous stack
        start = 0 + dataset.domain.inclusive_min[0]

    # Start alignment
    output_shape = np.max(transform[:,:,-1], axis=0).astype(int)
    skipped = 0
    for z in tqdm(range(start, dataset.shape[0]), 
                    position=0,
                    desc=f'{dataset_name}: Rendering aligned slices'):
        # Load data
        data = dataset[z].read().result()

        if not data.any():
            # If empty slice, skip and go to next z
            skipped += 1
            continue

        # Resample if needed (target_scale != 1)
        data = downsample(data, target_scale)

        # Load or compute mask
        if dataset_mask is not None:
            data_mask = dataset_mask[z].read().result()
            data_mask = downsample(data_mask, target_scale)
        else:
            data_mask = compute_greyscale_mask(data)

        # Transform data
        M = transform[z,:,:-1]
        data = cv2.warpAffine(data, M, output_shape[::-1])
        data_mask = cv2.warpAffine(data_mask.astype(np.uint8), M, output_shape[::-1]).astype(bool)
        data_bbox = bounding_box.BoundingBox(start=(0, 0, 0), 
                                             size=(data.shape[-1], data.shape[-2], 1))
        
        # Warp data in parallel. warp_subvolume uses one thread per image so we use ndimage_wrap instead
        if first_slice is None:
            inv_z = z - skipped
        else:
            inv_z = z + 1 - skipped

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

        # Writing bounding box
        y1, y2, x1, x2 = mask_to_bbox(aligned_mask)
        
        write_data(destination, 
                   aligned[y1:y2, x1:x2], # Only write in the bounding box where the data is
                   z + z_offset - dataset.domain.inclusive_min[0], # z_offset relates to original minimum
                   aligned_mask[y1:y2, x1:x2], # Mask where to write the data
                   np.abs(xy_offset) + np.array([x1, y1]), # Only write in the bounding box where the data is
                   save_downsampled, ds_destination)
        write_data(destination_mask, 
                   aligned_mask[y1:y2, x1:x2], 
                   z + z_offset - dataset.domain.inclusive_min[0], 
                   aligned_mask[y1:y2, x1:x2],
                   np.abs(xy_offset) + np.array([x1, y1]), 
                   1, None)
    logging.info(f'{dataset_name}: Done. ({skipped} empty slices)')

    # Add an attribute to keep track of what datasets have been aligned already
    attrs['z_aligned'] = True
    set_dataset_attributes(dataset, attrs)

    return True


def write_data(destination, data, z, mask=None, xy_offset=[0,0], save_downsampled=1, ds_destination=None):
    tasks = []
    x_off, y_off = xy_offset

    # Write to destination
    y,x = data.shape
    if np.any(destination.domain.exclusive_max < np.array([z+1, y+y_off, x+x_off])):
        new_max = np.max([destination.domain.exclusive_max, [z+1, y+y_off, x+x_off]], axis=0)
        destination = destination.resize(exclusive_max=new_max, expand_only=True).result()

    if mask is not None:
        # We assume that data already exists and we need to preserve it
        og_data = destination[z, y_off:y+y_off, x_off:x+x_off].read().result()
        og_data[mask] = data[mask]
        data = og_data
    
    tasks.append(destination[z, y_off:y+y_off, x_off:x+x_off].write(data).result())

    # Write downsampled data for inspection
    if save_downsampled > 1 and ds_destination is not None:
        ds_data = cv2.resize(data, None, fx=1/save_downsampled, fy=1/save_downsampled)
        y,x = ds_data.shape
        x_off, y_off = xy_offset // save_downsampled
        if np.any(ds_destination.domain.exclusive_max < np.array([z+1, y+y_off, x+x_off])):
            new_max = np.max([ds_destination.domain.exclusive_max, [z+1, y+y_off, x+x_off]], axis=0)
            ds_destination = ds_destination.resize(exclusive_max=new_max, expand_only=True).result()

        tasks.append(ds_destination[z, y_off:y+y_off, x_off:x+x_off].write(ds_data).result())
    
    return tasks

if __name__ == '__main__':

    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        config = json.load(f)

    params = inspect.signature(align_stack_z).parameters
    relevant_args = {k: v for k, v in config.items() if k in params}

    align_stack_z(**relevant_args)
