import logging
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import tensorstore as ts

from tqdm import tqdm

from emalign.arrays.sift import SIFT_SCALES, estimate_transform_sift
from emalign.arrays.utils import resample
from emalign.io.process.mask import compute_greyscale_mask
from emalign.io.progress import get_mongo_client, get_mongo_db, log_progress, check_progress
from emalign.io.store import write_ndarray, open_store, get_store_attributes, set_store_attributes


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


def _pairwise_transform(ref_img, ref_mask, mov_img, mov_mask):
    '''Compute pairwise raw homology matrix mapping mov to ref, or None.'''
    for sift_scale in SIFT_SCALES:
        M, _, _, valid, _ = estimate_transform_sift(
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
                             first_slice=None,
                             first_slice_mask=None,
                             ignore_slices=(),
                             project_name='OV',
                             mongodb_config_filepath=None,
                             num_workers=4,
                             cache_workers=4,
                             cv2_threads=1
                             ):
    '''Compute and store the pairwise affine for every slice of one stack.'''

    # Cap number of parallel workers available to OpenCV
    cv2.setNumThreads(cv2_threads)

    # For logging
    external_first_slice = first_slice is not None

    # Progress logging
    db = get_mongo_db(get_mongo_client(mongodb_config_filepath), project_name)
    step_name = 'transform_z'

    # Get image stores
    dataset_path = os.path.abspath(dataset_path)
    dataset = open_store(dataset_path, mode='r', dtype=ts.uint8)
    dataset_mask = open_store(dataset_path + '_mask', mode='r', dtype=ts.bool, allow_missing=True)

    res = get_store_attributes(dataset)['resolution'][-1]
    target_scale = 1 if yx_target_resolution is None else res / yx_target_resolution

    z_min = local_z_min if local_z_min is not None else dataset.domain.inclusive_min[0]
    z_max = local_z_max if local_z_max is not None else dataset.domain.exclusive_max[0]
    ignore_slices = set(ignore_slices)                 

    # Find container root
    zarr_path = os.path.abspath(destination_path)
    while not zarr_path.endswith('.zarr'):
        if zarr_path == os.path.dirname(zarr_path):
            raise ValueError('No zarr container found in provided destination path.')
        # Get the zarr container
        zarr_path = os.path.dirname(zarr_path)

    # Create transform store.
    trsf_path = os.path.join(zarr_path, 'z_intermediate', 'transform', dataset_name)
    dataset_trsf = open_store(
        trsf_path, mode='a', dtype=ts.float32,
        shape=[z_max, 2, 3], chunks=[1, 2, 3], axis_labels=['z', 'a', 'b'],
        fill_value=np.nan)
    
    def write_t(z, M):
        nonlocal dataset_trsf
        dataset_trsf, _ = write_ndarray(dataset_trsf, np.asarray(M, np.float32), 
                                        z, resolve=False)
    
    # ---------- First slice to process ----------
    first_z = z_min
    if first_slice is None:
        # Find the first non-empty slice
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
        first_z += 1 # Skip the first slice
    else:
        # Use the provided slice as anchor for this dataset
        ref_slice = {'z': -1, 'img': first_slice, 'mask': first_slice_mask, 'empty': False}

    # ---------- Check progress ----------
    for z in range(first_z, z_max):
        if not check_progress(db, dataset_name, step_name, z):
            if z != first_z:
                # We pick up progress if there was any
                first_z = z
                i = 1
                ref_slice = _load_slice(dataset, dataset_mask, first_z-i, target_scale)
                while ref_slice['empty']:
                    i += 1
                    ref_slice = _load_slice(dataset, dataset_mask, first_z-i, target_scale)
            break
    else:
        # Everything was processed already
        logging.info(f'{dataset_name}: All transforms were already computed.')
        return
    
    # ---------- Pools and IO functions ----------
    read_pool = ThreadPoolExecutor(max_workers=cache_workers)   # Cache slices for processing
    proc_pool = ThreadPoolExecutor(max_workers=num_workers)     # Compute transform

    # Only fetch data within a window to bound memory and keep pairing in order
    window = max(num_workers, cache_workers) * 2 + 2
    read_futures = {}
    next_read = first_z

    def prefetch_data(up_to):
        nonlocal next_read
        up_to = min(up_to, z_max - 1)
        while next_read <= up_to:
            read_futures[next_read] = read_pool.submit(
                _load_slice, dataset, dataset_mask, next_read, target_scale)
            next_read += 1

    prefetch_data(first_z + window)

    # ---------- Stage 3 consumer: persist one pairwise result ----------
    proc_futs = deque()   # (z, future)
    n_gap = 0

    def drain_one():
        nonlocal n_gap
        z, M_fut = proc_futs.popleft()
        global_z = z + z_offset - z_min
        res = M_fut.result() # May be None (failed SIFT)
        
        if res is not None:
            write_t(z, res)
        n_gap += res is None
        log_progress(db, dataset_name, step_name, global_z, z,
                     {'valid_estimate': res is not None, 'scale': target_scale, 'skipped': res is None})

    # ---------- Main loop: pair, dispatch, drain ----------
    pbar = tqdm(total=z_max - first_z, 
                desc=f'{dataset_name}: Computing transforms', 
                dynamic_ncols=True)
    for z in range(first_z, z_max):
        prefetch_data(z + window)
        mov_slice = read_futures.pop(z).result()
        global_z = z + z_offset - z_min

        if mov_slice['empty'] or z in ignore_slices:
            # Skip this, compute transform across the gap
            log_progress(db, dataset_name, step_name, global_z, z,
                         {'empty_slice': bool(mov_slice['empty']), 'skipped': True})
            pbar.update(1)
            continue

        # Compute SIFT transform
        M_fut = proc_pool.submit(_pairwise_transform,
                               ref_slice['img'], ref_slice['mask'], mov_slice['img'], mov_slice['mask'])
        proc_futs.append((z, M_fut))
        ref_slice = mov_slice   # Reference for next slice
        
        pbar.update(1)
        while len(proc_futs) >= window:
            pbar.set_description(f'{dataset_name}: Writing...')
            drain_one()
            pbar.set_description(f'{dataset_name}: Computing transforms')

    # Empty the queue and shutdown
    while proc_futs:
        drain_one()
    pbar.close()
    read_pool.shutdown()
    proc_pool.shutdown()

    set_store_attributes(dataset_trsf, {
        'dataset_path': dataset_path,
        'scale': target_scale,
        'external_first_slice': external_first_slice,
        'pairwise': True,  # Pairwise, not cumulative transforms
        'first_z': int(first_z)
    })

    logging.info(f'{dataset_name}: Pairwise transform done.')
    logging.info(f'{dataset_name}: Empty/skipped slices: {n_gap}.')


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
                             num_workers=4,
                             cache_workers=4,
                             cv2_threads=1
                             )
