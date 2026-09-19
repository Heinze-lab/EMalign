import logging
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import tensorstore as ts

from tqdm import tqdm

from emalign.arrays.sift import SIFT_SCALES, estimate_transform_sift_fast
from emalign.arrays.utils import resample
from emalign.io.process.mask import compute_greyscale_mask
from emalign.io.progress import get_mongo_client, get_mongo_db, log_progress, check_progress
from emalign.io.store import (write_ndarray, open_store, find_ref_slice,
                              get_store_attributes, set_store_attributes)


IDENTITY = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)


def _load_slice(dataset, dataset_mask, z, scale):
    '''Load one slice + mask and resample.'''
    img = dataset[z].read().result()
    if not img.any():
        return {'z': z, 'img': None, 'mask': None, 'empty': True}

    img = resample(img, scale)
    if dataset_mask is not None:
        mask = resample(dataset_mask[z].read().result(), scale)
    else:
        mask = compute_greyscale_mask(img, downsample_factor=10)
    return {'z': z, 'img': img, 'mask': mask, 'empty': False}


def _load_reference_slice(reference, reference_mask, z, scale, reverse=False):
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


def _pairwise_transform(ref_img, ref_mask, mov_img, mov_mask):
    '''Compute pairwise raw homology matrix mapping mov to ref, or None.'''
    for sift_scale in SIFT_SCALES:
        M, _, _, valid, _ = estimate_transform_sift_fast(
            ref_img, mov_img, scale=sift_scale,
            ref_mask=ref_mask, mov_mask=mov_mask,
            refine_estimate=False, return_raw_homology=True)
        if valid and M is not None:
            return M.astype(np.float64)
    return None


def compute_transforms_stack(dataset_path,
                             dataset_name,
                             destination_path,
                             z_offset,
                             yx_target_resolution,
                             local_z_min=None,
                             local_z_max=None,
                             reference_dataset=None,
                             reference_offset=0,
                             align_to_reference=False,
                             ignore_slices=(),
                             project_name='OV',
                             mongodb_config_filepath=None,
                             num_workers=0
                             ):
    '''Compute and store the pairwise affine for every slice of one stack.

    The anchor each slice is aligned to depends on the reference:
      - no reference_dataset: the anchor is the first valid slice of this stack,
        it gets the identity transform and every slice follows the previous one.
      - reference_dataset: the anchor is the last valid slice of the reference,
        then every slice still follows the previous one of this stack.
      - reference_dataset + align_to_reference: every slice is aligned to its
        matching slice in the reference, found at
        global_z + reference_offset.
    '''

    if num_workers > 0:
        # Set the number of workers used by cv2 to match keypoints
        cv2.setNumThreads(max(1, num_workers - 1))

    # Progress logging
    db = get_mongo_db(get_mongo_client(mongodb_config_filepath), project_name)
    step_name = 'transform_z'

    # ---------- Prepare stores ----------
    # Get input store
    dataset_path = os.path.abspath(dataset_path)
    dataset = open_store(dataset_path, mode='r', dtype=ts.uint8)
    dataset_mask = open_store(dataset_path + '_mask', mode='r', dtype=ts.bool, allow_missing=True)               

    # Find container root for destination
    zarr_path = os.path.abspath(destination_path)
    while not zarr_path.endswith('.zarr'):
        if zarr_path == os.path.dirname(zarr_path):
            raise ValueError('No zarr container found in provided destination path.')
        # Get the zarr container
        zarr_path = os.path.dirname(zarr_path)

    res = get_store_attributes(dataset)['resolution'][-1]
    target_scale = 1 if yx_target_resolution is None else res / yx_target_resolution

    # Get reference store, it may have its own resolution
    if reference_dataset is None:
        if align_to_reference:
            raise ValueError('align_to_reference requires a reference_dataset.')
        reference = reference_mask = None
        ref_scale = target_scale
    else:
        reference_dataset = os.path.abspath(reference_dataset)
        reference = open_store(reference_dataset, mode='r', dtype=ts.uint8)
        reference_mask = open_store(reference_dataset + '_mask', mode='r',
                                    dtype=ts.bool, allow_missing=True)
        ref_res = get_store_attributes(reference)['resolution'][-1]
        ref_scale = 1 if yx_target_resolution is None else ref_res / yx_target_resolution

    z_min = local_z_min if local_z_min is not None else dataset.domain.inclusive_min[0]
    z_max = local_z_max if local_z_max is not None else dataset.domain.exclusive_max[0]
    ignore_slices = set(ignore_slices)  

    # Create transform store at the root
    # These will contain the homography matrix and offset
    trsf_path = os.path.join(zarr_path, 'z_intermediate', 'transform', dataset_name)
    dataset_trsf = open_store(
        trsf_path, 
        mode='a', dtype=ts.float32,
        shape=[z_max, 2, 3], chunks=[1, 2, 3], axis_labels=['z', 'a', 'b'],
        fill_value=np.nan)
    
    def write_t(z, M):
        nonlocal dataset_trsf
        dataset_trsf, _ = write_ndarray(dataset_trsf, np.asarray(M, np.float32), 
                                        z, resolve=False)

    # ---------- Check progress ----------
    first_z = z_min
    resume = False
    for z in range(z_min, z_max):
        if not check_progress(db, dataset_name, step_name, z):
            if z != z_min:
                if align_to_reference:
                    # Every slice has its own reference, nothing to carry over
                    first_z = z
                    resume = True
                    break

                # Walk back to the last slice holding a valid transform
                for z_ref in range(z - 1, z_min - 1, -1):
                    M_ref = dataset_trsf[z_ref].read().result()
                    if not np.isnan(M_ref).any():
                        break
                else:
                    # Nothing valid behind us, back to fresh start
                    break   

                ref_slice = _load_slice(dataset, dataset_mask, z_ref, target_scale)
                first_z = z
                resume = True
            break
    else:
        # Everything was processed already
        logging.info(f'{dataset_name}: All transforms were already computed.')
        return
    
    # ---------- First slice to process ----------
    if not resume:
        first_z = z_min

        # Find first slice to use
        if reference_dataset is None:
            # Find the first non-empty slice in this dataset
            ref_slice = _load_slice(dataset, dataset_mask, first_z, target_scale)
            while ref_slice['empty']:
                global_z = first_z + z_offset - z_min
                log_progress(db, dataset_name, step_name, global_z, first_z, {'empty_slice': True, 'skipped': True})

                # Get next one
                first_z += 1
                ref_slice = _load_slice(dataset, dataset_mask, first_z, target_scale)

            # Very first stack, the first slice is not transformed
            write_t(first_z, IDENTITY)
            log_progress(db, dataset_name, step_name, first_z + z_offset - z_min, first_z,
                        {'valid_estimate': True, 'anchor': True, 'scale': target_scale, 'skipped': False})
            first_z += 1 # Start the loop with the next slice
        elif not align_to_reference:
            # Get the last valid slice of the reference as anchor
            ref_slice = _load_reference_slice(reference, reference_mask, None,
                                              ref_scale, reverse=True)
        # else: each slice gets its own reference slice in the loop below

        # Set attributes
        set_store_attributes(
            dataset_trsf, 
            {
                'dataset_path': dataset_path,
                'scale': target_scale,
                'external_first_slice': reference_dataset is not None,
                'reference_path': reference_dataset,
                'reference_offset': int(reference_offset),
                'align_to_reference': bool(align_to_reference),
                'ref_scale': ref_scale,
                'raw_homography': True,  # Pairwise, not cumulative transforms
                'first_z': int(first_z)
        })
        

    # ---------- Pools and IO functions ----------
    # Cache slices for processing
    read_pool = ThreadPoolExecutor(max_workers=1)   

    # Only fetch data within a window to bound memory and keep pairing in order
    # In theory it's not necessary but it will facilitate inspection if necessary
    # We can also keep a slice in memory for the two operations it contributes to
    window = 4
    read_futures = {}
    next_read = first_z

    def prefetch_data(up_to):
        nonlocal next_read
        up_to = min(up_to, z_max - 1)
        while next_read <= up_to:
            read_futures[next_read] = read_pool.submit(
                _load_slice, dataset, dataset_mask, next_read, target_scale)
            next_read += 1

    # ---------- Stage 3 consumer: persist one pairwise result ----------
    # Prefetch the first slice
    prefetch_data(first_z + window)

    n_gap = 0
    n_failed = 0
    n_ignored = 0
    for z in tqdm(
        range(first_z, z_max), 
        total=z_max - first_z, 
        desc=f'{dataset_name}: Computing transforms', 
        dynamic_ncols=True
        ):
        # Prefetch more data to consume later
        prefetch_data(z + window)

        # Get current slice
        mov_slice = read_futures.pop(z).result()
        global_z = z + z_offset - z_min

        if mov_slice['empty']: 
            # Empty slice. Skip this and compute transform across the gap
            n_gap += 1
            log_progress(db, dataset_name, step_name, global_z, z,
                         {'empty_slice': bool(mov_slice['empty']), 'skipped': True})
            continue
        if z in ignore_slices:
            # Skipped by user
            n_ignored += 1
            log_progress(db, dataset_name, step_name, global_z, z,
                         {'skipped': True})
            continue

        if align_to_reference:
            # Matching slice in the reference, in global coordinates
            ref_slice = _load_reference_slice(
                reference, reference_mask, global_z + reference_offset, ref_scale)

        # Compute SIFT transform
        M = _pairwise_transform(
            ref_slice['img'], ref_slice['mask'], 
            mov_slice['img'], mov_slice['mask']
            )

        # Write to file
        if M is not None:
            write_t(z, M)
            if not align_to_reference:
                ref_slice = mov_slice   # Reference for next slice
        else:
            # Failed to find transform, so we keep the reference
            n_failed += 1

        log_progress(db, dataset_name, step_name, global_z, z,
                     {'valid_estimate': M is not None, 'scale': target_scale,
                      'z_ref': int(ref_slice['z']), 'ref_scale': ref_scale,
                      'skipped': M is None})
        

    # Shutdown the read pool
    read_pool.shutdown()

    logging.info(f'{dataset_name}: Pairwise transform done. | Ignored slices: {n_ignored}. | Empty slices: {n_gap}. | Failed slices: {n_failed}.')


def chain_transforms_path(
        path, 
        dataset_configs,
        anchor_inv=None,
        carry_in=None,
        bbox=None
        ):
    if anchor_inv is None: 
        anchor_inv = np.eye(3)

    chained_paths = []
    # First pass, chain transforms
    for dataset_name in path:
        if dataset_name not in dataset_configs:
            raise RuntimeError(f'No configuration found for dataset: {dataset_name}')

        config = dataset_configs[dataset_name].copy()
        trsf_chained_path, carry_in, bbox = chain_transforms_stack(
            dataset_name,
            config['dataset_path'],
            config['destination_path'],
            carry_in,
            bbox,
            anchor_inv
        )
        chained_paths.append(trsf_chained_path)
    return bbox

def shift_transforms(
        trsf_chained_path,
        shift_yx,
        output_shape,
        anchor_inv=None
):
    
    dataset_trsf_chained = open_store(trsf_chained_path, mode='r+')
    attrs = get_store_attributes(dataset_trsf_chained)
    
    if attrs.get('final', False):
        logging.info('Final transforms already written to file. Use --start-over to recompute them.')
        return

    if anchor_inv is None:
        anchor_inv = np.eye(3)
    
    shift_y, shift_x = shift_yx
    shift = shift_matrix(shift_x, shift_y)

    chained_transforms = dataset_trsf_chained.read().result() 
    chained_transforms = to_3x3(chained_transforms)
    
    shifted_chained_transforms = shift @ anchor_inv @ chained_transforms
    dataset_trsf_chained.write(shifted_chained_transforms[:, :2, :].astype(np.float32)).result()

    attrs |= {'final': True, 'output_shape': output_shape, 'shift_x': shift_x, 'shift_y': shift_y}
    set_store_attributes(dataset_trsf_chained, attrs)

def chain_transforms_stack(
        dataset_name,
        dataset_path,
        destination_path,
        carry_in=None,
        bbox=None,
        anchor_inv=None,
        return_transform=False
    ):

    if anchor_inv is None:
        anchor_inv = np.eye(3)

    dataset_path = os.path.abspath(dataset_path)
    dataset = open_store(dataset_path, 'r')

    # Find transforms
    destination_path = os.path.abspath(destination_path)
    zarr_path = os.path.abspath(destination_path)
    while not zarr_path.endswith('.zarr'):
        if zarr_path == os.path.dirname(zarr_path):
            raise ValueError('No zarr container found in provided destination path.')
        # Get the zarr container
        zarr_path = os.path.dirname(zarr_path)

    trsf_path = os.path.join(zarr_path, 'z_intermediate', 'transform', dataset_name)
    dataset_trsf = open_store(trsf_path, mode='r')
    trsf_chained_path = os.path.join(zarr_path, 'z_intermediate', 'transform_chained', dataset_name)
    dataset_trsf_chained = open_store(
        trsf_chained_path, 
        mode='a', dtype=ts.float32,
        shape=dataset_trsf.shape, chunks=[1, 2, 3], axis_labels=['z', 'a', 'b'],
        fill_value=np.nan)
    
    attrs = get_store_attributes(dataset_trsf_chained)
    if attrs is not None and not attrs['raw_homography']:
        chained_transforms = dataset_trsf_chained.read().result()
        chained_transforms = to_3x3(chained_transforms)
        
        merged_bbox = merge_bbox(bbox, attrs['bbox'])
        if return_transform:
            return chained_transforms, chained_transforms[-1], merged_bbox
        return trsf_chained_path, chained_transforms[-1], merged_bbox
    
    # Get everything in memory, should be tiny
    transforms = dataset_trsf.read().result()
    
    # Turn transforms from [z, 2, 3] to [z, 3, 3] so they can be multiplied with each other
    transforms = to_3x3(transforms)

    # Chain matrices starting with the first item
    # Ignore NaNs or they will break the chain
    valid = ~np.isnan(transforms).any(axis=(1, 2))
    chained_transforms = np.empty_like(transforms)
    prev = carry_in if carry_in is not None else np.eye(3)
    for i in range(len(transforms)):
        if not valid[i]:
            # Just inherit the last valid transform and jump to the next
            chained_transforms[i] = prev
            continue
        chained_transforms[i] = prev @ transforms[i]
        prev = chained_transforms[i]

    # Update the global bbox
    bbox = update_bbox(bbox, chained_transforms, dataset.shape[1:], anchor_inv)

    # Write to file
    dataset_trsf_chained.write(chained_transforms[:, :2, :].astype(np.float32)).result()

    set_store_attributes(dataset_trsf_chained, 
                            {
                                'dataset_path': dataset_path,
                                'raw_transform_path': trsf_path,
                                'raw_homography': False,
                                'bbox': bbox
    })

    if return_transform:
        return chained_transforms[:, :2, :], chained_transforms[-1], bbox
    return trsf_chained_path, chained_transforms[-1], bbox


def to_3x3(M):
    row = np.tile(np.array([0, 0, 1]), (M.shape[0], 1, 1))  # shape (z, 1, 3)
    return np.concatenate([M, row], axis=1) 


def merge_bbox(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return (
        min(a[0], b[0]), min(a[1], b[1]),
        max(a[2], b[2]), max(a[3], b[3]),
    )

def update_bbox(bbox, chained, input_shape, anchor_inv):
    '''
    Project this stack's chained transforms into the *global* anchor
    frame, warp the image corners through them, and fold the result
    into a running (xmin, ymin, xmax, ymax) bbox.
 
    `anchor_inv` must be the SAME matrix across all stacks in the
    sequence -- it defines a single consistent orientation for the
    whole chain. Pass np.eye(3) if you don't want any re-anchoring.
    '''
    y, x = input_shape
    corners = np.array([
        [0, 0, 1],
        [x, 0, 1],
        [x, y, 1],
        [0, y, 1],
    ], dtype=np.float64).T  # (3, 4)
 
    anchored = anchor_inv @ chained                 # (z, 3, 3)
    warped = anchored @ corners                      # (z, 3, 4)
    warped_xy = warped[:, :2, :].transpose(0, 2, 1).reshape(-1, 2)
 
    xmin, ymin = warped_xy.min(axis=0)
    xmax, ymax = warped_xy.max(axis=0)
 
    if bbox is None:
        return float(xmin), float(ymin), float(xmax), float(ymax)
    return (
        min(bbox[0], xmin), min(bbox[1], ymin),
        max(bbox[2], xmax), max(bbox[3], ymax),
    )


def compute_global_bbox(bbox):
    '''Turn the accumulated global bbox into output shape + shift.'''
    xmin, ymin, xmax, ymax = bbox
    shift_x, shift_y = -xmin, -ymin
    output_shape = (int(np.ceil(ymax - ymin)), int(np.ceil(xmax - xmin)))
    return output_shape, (shift_y, shift_x)
 
 
def shift_matrix(shift_x, shift_y):
    return np.array([
        [1, 0, shift_x],
        [0, 1, shift_y],
        [0, 0, 1],
    ], dtype=np.float64)



if __name__ == '__main__':

    import json
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('absl').setLevel(logging.WARNING)
    logging.getLogger('jax._src.xla_bridge').setLevel(logging.WARNING)

    config_path = '/mnt/hdd1/SRC/EMpipelines/EMalign/output_test/config/z_config/z_01_OV_2.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    compute_transforms_stack(config['dataset_path'],
                             config['dataset_name'],
                             config['destination_path'],
                             config['z_offset'],
                             config['yx_target_resolution'],
                             local_z_min=None,
                             local_z_max=None,
                             ignore_slices=(),
                             project_name='TEST_NEW_TRSFM',
                             mongodb_config_filepath=None,
                             cv2_threads=16,
                             cache_workers=2
                             )
