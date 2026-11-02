import logging
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import tensorstore as ts

from tqdm import tqdm

from emalign.arrays.sift import SIFT_SCALES, estimate_transform_sift_fast
from emalign.io.progress import get_mongo_client, get_mongo_db, log_progress, check_progress
from emalign.io.store import write_ndarray, open_store, get_store_attributes, set_store_attributes
from emalign.io.utils import load_reference_slice, load_slice


IDENTITY = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)

# What each slice was aligned to, with align_to_reference. Otherwise it is the z of the
# moving slice it was chained to.
NO_REF = -2             # No transform: empty, ignored, failed or too far from the reference
TO_REFERENCE = -1       # Matching slice of the reference
FROM_OTHER_STACK = -3   # Slice of the previous stack, carried over (see read_carried_from)


def read_carried_from(zarr_path, dataset_name):
    '''Slice of another stack the first slices of this stack were chained to (FROM_OTHER_STACK),
    as {'dataset_name', 'dataset_path', 'z', 'global_z', 'depth'}, or None.'''
    attrs = get_store_attributes(os.path.join(zarr_path, 'z_intermediate', 'transform_ref', dataset_name))
    return (attrs or {}).get('carried_from')


def last_valid_slice(zarr_path, dataset_name, z_min, z_max):
    '''Last slice of a stack holding a valid transform, and its number of hops to the reference,
    as (z, depth), or None.'''
    z_refs = read_transform_refs(zarr_path, dataset_name, z_max)
    if z_refs is None:
        return None
    carried = read_carried_from(zarr_path, dataset_name)
    for z in range(z_max - 1, z_min - 1, -1):
        if z_refs[z] != NO_REF:
            return z, chain_depth(z_refs, z, carried['depth'] if carried else 0)
    return None


def read_transform_refs(zarr_path, dataset_name, n):
    '''What each slice was aligned to (see NO_REF), as an int array of length n, or None if
    it was not recorded (stack computed before slices could chain across reference gaps).'''
    store = open_store(os.path.join(zarr_path, 'z_intermediate', 'transform_ref', dataset_name), 
                       mode='r', dtype=ts.int32, allow_missing=True)
    if store is None:
        return None
    z_refs = np.full(n, NO_REF, dtype=np.int64)
    stored = store.read().result()[:n, 0, 0]
    z_refs[:len(stored)] = stored
    return z_refs


def chain_depth(z_refs, z, carried_depth=0):
    '''Number of slice to slice hops between slice z and the reference.
    carried_depth: depth of the slice of another stack chains may start from (FROM_OTHER_STACK).'''
    depth = 0
    while z_refs[z] >= 0:
        z = z_refs[z]
        depth += 1
        if depth > len(z_refs):
            raise RuntimeError(f'Slice {z}: circular chain of transforms.')
    if z_refs[z] == FROM_OTHER_STACK:
        depth += 1 + carried_depth
    return depth


def load_matching_reference(reference, reference_mask, z, scale):
    '''Reference slice at z, empty if z is outside of the reference or blank.
    There is no search: a slice from another depth would misplace the moving slice.'''
    if not reference.domain.inclusive_min[0] <= z < reference.domain.exclusive_max[0]:
        return {'z': z, 'img': None, 'mask': None, 'empty': True}
    return load_slice(reference, reference_mask, z, scale)


def _pairwise_transform(ref_img, ref_mask, mov_img, mov_mask, match_scale=1):
    '''Compute pairwise raw homology matrix mapping mov to ref, or None.'''
    for sift_scale in SIFT_SCALES:
        M, _, _, valid, _ = estimate_transform_sift_fast(
            ref_img, mov_img, scale=sift_scale,
            ref_mask=ref_mask, mov_mask=mov_mask,
            refine_estimate=False, return_raw_homology=True)
        if valid and M is not None:
            return M.astype(np.float64), sift_scale
    return None


def compute_transforms_stack(dataset_path,
                             dataset_name,
                             destination_path,
                             z_offset,
                             yx_target_resolution,
                             local_z_min=None,
                             local_z_max=None,
                             reference_path=None,
                             reference_offset=0,
                             align_to_reference=False,
                             ignore_slices=(),
                             project_name='OV',
                             mongodb_config_filepath=None,
                             num_workers=0
                             ):
    '''Compute and store the pairwise affine for every slice of one stack.

    The anchor each slice is aligned to depends on the reference:
      - no reference_path: the anchor is the first valid slice of this stack,
        it gets the identity transform and every slice follows the previous one.
      - reference_path: the anchor is the last valid slice of the reference,
        then every slice still follows the previous one of this stack.
      - reference_path + align_to_reference: every slice is aligned to its
        matching slice in the reference, found at z + reference_offset. If that
        slice is missing or blank, the slice is aligned to the previous valid slice
        of this stack instead (to the next one before the first slice that reached the
        reference), and chained to it. A slice is at most max_gap hops away from
        the reference, it gets no transform beyond that. What each slice was aligned
        to is stored in z_intermediate/transform_ref, for chain_transforms_stack.
        carry_slice is the last valid slice of the previous stack ({'dataset_name',
        'dataset_path', 'z', 'global_z', 'depth'}, see last_valid_slice): if it is at most
        max_gap slices before this stack, it is the previous slice of the first slices, so
        that a stack starting without reference (e.g. a single slice) can still be placed.
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
    if reference_path is None:
        if align_to_reference:
            raise ValueError('align_to_reference requires a reference_path.')
        reference = reference_mask = None
        ref_scale = target_scale
    else:
        reference_path = os.path.abspath(reference_path)
        reference = open_store(reference_path, mode='r', dtype=ts.uint8)
        reference_mask = open_store(reference_path + '_mask', mode='r',
                                    dtype=ts.bool, allow_missing=True)
        ref_res = get_store_attributes(reference)['resolution'][-1]
        ref_scale = 1 if yx_target_resolution is None else ref_res / yx_target_resolution

    # When every slice is matched to the reference, match at the coarser of the two resolutions:
    # upsampling a coarse reference adds nothing for SIFT and multiplies its size (50 -> 12 nm is
    # ~17x the pixels). Both images are loaded at match_scale relative to the target resolution,
    # and the translation of each transform is scaled back to target pixels afterwards.
    match_scale = min(1, 1 / ref_scale) if align_to_reference else 1

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

    if align_to_reference:
        # What each slice was aligned to (see NO_REF)
        dataset_ref = open_store(
            os.path.join(zarr_path, 'z_intermediate', 'transform_ref', dataset_name),
            mode='a', dtype=ts.int32,
            shape=[z_max, 1, 1], chunks=[1, 1, 1], axis_labels=['z', 'a', 'b'],
            fill_value=NO_REF)

    def write_r(z, z_ref):
        nonlocal dataset_ref
        if align_to_reference:
            dataset_ref, _ = write_ndarray(dataset_ref, np.full((1, 1), z_ref, np.int32),
                                           z, resolve=False)

    # Slices without reference are chained to their neighbour
    prev_slice = None   # Last slice holding a valid transform
    prev_depth = 0      # Its number of hops to the reference

    if align_to_reference and carry_slice is not None and not 0 < z_offset - carry_slice['global_z'] <= max_gap:
        # Too far from this stack (or on the same slices) to stand for its previous slice
        carry_slice = None

    def load_carry_slice():
        '''Carried slice of the previous stack, at its own resolution, or None.'''
        if carry_slice is None:
            return None
        other_path = os.path.abspath(carry_slice['dataset_path'])
        other = open_store(other_path, mode='r', dtype=ts.uint8)
        other_mask = open_store(other_path + '_mask', mode='r', dtype=ts.bool, allow_missing=True)
        other_res = get_store_attributes(other)['resolution'][-1]
        other_scale = 1 if yx_target_resolution is None else other_res / yx_target_resolution
        s = load_slice(other, other_mask, carry_slice['z'], other_scale * match_scale)
        if s['empty']:
            return None
        set_store_attributes(dataset_ref, (get_store_attributes(dataset_ref) or {}) | {'carried_from': carry_slice})
        return s | {'carried': True}

    def prev_ref():
        '''z_ref of a slice aligned to prev_slice.'''
        return FROM_OTHER_STACK if prev_slice.get('carried', False) else prev_slice['z']

    # ---------- Check progress ----------
    first_z = z_min
    resume = False
    timestamp = None
    for z in range(z_min, z_max):
        done, timestamp = check_progress(db, dataset_name, step_name, z, return_timestamp=True)
        if not done:
            if z != z_min:
                if align_to_reference:
                    # Carry over the last slice holding a valid transform, in case
                    # the next slices have no reference
                    first_z = z
                    resume = True
                    z_refs = read_transform_refs(zarr_path, dataset_name, z)
                    if z_refs is not None:
                        carried = read_carried_from(zarr_path, dataset_name)
                        for z_prev in range(z - 1, z_min - 1, -1):
                            if z_refs[z_prev] != NO_REF:
                                prev_slice = load_slice(dataset, dataset_mask, z_prev,
                                                        target_scale * match_scale)
                                prev_depth = chain_depth(z_refs, z_prev, carried['depth'] if carried else 0)
                                break
                    if prev_slice is None:
                        # Nothing valid behind us in this stack, the previous stack may stand for it
                        prev_slice = load_carry_slice()
                        prev_depth = carry_slice['depth'] if prev_slice is not None else 0
                    break

                # Walk back to the last slice holding a valid transform
                for z_ref in range(z - 1, z_min - 1, -1):
                    M_ref = dataset_trsf[z_ref].read().result()
                    if not np.isnan(M_ref).any():
                        break
                else:
                    # Nothing valid behind us, back to fresh start
                    break   

                ref_slice = load_slice(dataset, dataset_mask, z_ref, target_scale)
                first_z = z
                resume = True
            break
    else:
        # Everything was processed already
        logging.info(f'    All transforms were already computed.')
        return
    
    # ---------- First slice to process ----------
    if not resume:
        first_z = z_min

        # Find first slice to use
        if reference_path is None:
            # Find the first non-empty slice in this dataset
            ref_slice = load_slice(dataset, dataset_mask, first_z, target_scale)
            while ref_slice['empty']:
                global_z = first_z + z_offset - z_min
                log_progress(db, dataset_name, step_name, global_z, first_z, {'empty_slice': True, 'skipped': True})

                # Get next one
                first_z += 1
                ref_slice = load_slice(dataset, dataset_mask, first_z, target_scale)

            # Very first stack, the first slice is not transformed
            write_t(first_z, IDENTITY)
            log_progress(db, dataset_name, step_name, first_z + z_offset - z_min, first_z,
                        {'valid_estimate': True, 'anchor': True, 'scale': target_scale, 'skipped': False})
            first_z += 1 # Start the loop with the next slice
        elif not align_to_reference:
            # Get the last valid slice of the reference as anchor
            ref_slice = load_reference_slice(
                reference, reference_mask, None,
                ref_scale, reverse=True
                )
        else:
            # Each slice gets its own reference slice in the loop below. The previous stack
            # stands for the previous slice of the first slices, if close enough.
            set_store_attributes(dataset_ref, (get_store_attributes(dataset_ref) or {}) | {'carried_from': None})
            prev_slice = load_carry_slice()
            prev_depth = carry_slice['depth'] if prev_slice is not None else 0

        # Set attributes
        set_store_attributes(
            dataset_trsf, 
            {
                'dataset_path': dataset_path,
                'scale': target_scale,
                'external_first_slice': reference_path is not None,
                'reference_path': reference_path,
                'reference_offset': int(reference_offset),
                'align_to_reference': bool(align_to_reference),
                'ref_scale': ref_scale,
                'match_scale': match_scale,
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
                load_slice, dataset, dataset_mask, next_read, target_scale * match_scale)
            next_read += 1

    # ---------- Compute pairwise transforms ----------
    # Prefetch the first slice
    prefetch_data(first_z + window)

    n_gap = 0
    n_failed = 0
    n_ignored = 0
    n_chained = 0
    n_no_ref = 0

    def align_slice(ref_slice, mov_slice, z_ref, depth):
        '''Match mov_slice to ref_slice, write and log the result. Returns whether it was found.'''
        nonlocal n_failed
        z = mov_slice['z']
        result = _pairwise_transform(
            ref_slice['img'], ref_slice['mask'],
            mov_slice['img'], mov_slice['mask'],
            match_scale
            )
        if result is not None:
            M, sift_scale = result
            # Back to target resolution pixels, the linear part is scale invariant
            M[:, 2] /= match_scale

            # Write to file
            write_t(z, M)
            write_r(z, z_ref)
        else:
            # Failed to find transform
            sift_scale = None
            write_t(z, np.full((2, 3), np.nan))
            write_r(z, NO_REF)
            n_failed += 1

        metadata = {'valid_estimate': result is not None, 'scale': target_scale,
                    'match_scale': match_scale, 'z_ref': int(ref_slice['z']), 'ref_scale': ref_scale,
                    'skipped': result is None, 'sift_scale': sift_scale}
        if align_to_reference:
            # z_ref is in the reference, in this stack or in the previous stack
            metadata |= {'aligned_to': {TO_REFERENCE: 'reference', FROM_OTHER_STACK: 'other_stack'}.get(z_ref, 'previous'),
                         'chain_depth': depth}
            if z_ref == FROM_OTHER_STACK:
                metadata['carried_from'] = carry_slice['dataset_name']
        log_progress(db, dataset_name, step_name, z + z_offset - z_min, z, metadata)
        return result is not None

    def skip_no_ref(z):
        '''No reference within max_gap slices, the slice gets no transform.'''
        nonlocal n_no_ref
        write_t(z, np.full((2, 3), np.nan))
        write_r(z, NO_REF)
        n_no_ref += 1
        log_progress(db, dataset_name, step_name, z + z_offset - z_min, z,
                     {'no_reference': True, 'skipped': True})

    # Slices without reference met before any slice reached it, aligned
    # to the next valid slice once one does. Bounded by max_gap.
    pending = []

    for z in tqdm(
        range(first_z, z_max), 
        total=z_max - first_z, 
        desc=f'    Computing transforms', 
        dynamic_ncols=True
        ):
        # Prefetch more data to consume later
        prefetch_data(z + window)

        # Get current slice
        mov_slice = read_futures.pop(z).result()
        global_z = z + z_offset - z_min

        if mov_slice['empty']:
            # Empty slice. Skip this and compute transform across the gap
            write_t(z, np.full((2, 3), np.nan))
            write_r(z, NO_REF)
            n_gap += 1
            log_progress(db, dataset_name, step_name, global_z, z,
                         {'empty_slice': bool(mov_slice['empty']), 'skipped': True})
            continue
        if z in ignore_slices:
            # Skipped by user
            write_t(z, np.full((2, 3), np.nan))
            write_r(z, NO_REF)
            n_ignored += 1
            log_progress(db, dataset_name, step_name, global_z, z,
                         {'skipped': True})
            continue

        if align_to_reference:
            # Matching slice in the reference, in global coordinates
            ref_slice = load_reference_slice(
                reference, reference_mask, z + reference_offset, ref_scale * match_scale)

        # Compute SIFT transform
        res = _pairwise_transform(
            ref_slice['img'], ref_slice['mask'], 
            mov_slice['img'], mov_slice['mask']
            )
        if res is not None:
            M, sift_scale = res
            # Back to target resolution pixels, the linear part is scale invariant
            M[:, 2] /= match_scale

            # Write to file
            write_t(z, M)
            if not align_to_reference:
            if align_slice(ref_slice, mov_slice, None, None):
                ref_slice = mov_slice   # Reference for next slice
            # else: failed to find transform, so we keep the reference
            continue

        # Matching slice in the reference, in global coordinates
        ref_slice = load_matching_reference(
            reference, reference_mask, z + reference_offset, ref_scale * match_scale)

        if not ref_slice['empty']:
            if align_slice(ref_slice, mov_slice, TO_REFERENCE, 0):
                prev_slice, prev_depth = mov_slice, 0

                # Slices waiting for the reference are chained to this one, closest first
                next_slice, depth = mov_slice, 0
                for p in pending[::-1]:
                    n_chained += 1
                    if align_slice(next_slice, p, next_slice['z'], depth + 1):
                        next_slice, depth = p, depth + 1
                pending = []
            # else: failed, the previous slice stays the one to chain to
        elif prev_slice is not None:
            # No reference, chain to the previous valid slice if it is close enough to the reference
            if prev_depth < max_gap:
                n_chained += 1
                if align_slice(prev_slice, mov_slice, prev_ref(), prev_depth + 1):
                    prev_slice, prev_depth = mov_slice, prev_depth + 1
            else:
                skip_no_ref(z)
        else:
            # No reference and nothing to chain to yet, wait for a slice that reaches it
            pending.append(mov_slice)
            if len(pending) > max_gap:
                # The first one will be too far from the reference
                skip_no_ref(pending.pop(0)['z'])

    # Slices that never met a slice reaching the reference
    for p in pending:
        skip_no_ref(p['z'])

    # Shutdown the read pool
    read_pool.shutdown()

    # Mark stack transforms as complete
    attrs = get_store_attributes(dataset_trsf)
    attrs['complete'] = True
    set_store_attributes(dataset_trsf, attrs)

    logging.info(f'    Pairwise transform done. | Ignored slices: {n_ignored}. | Empty slices: {n_gap}. | Failed slices: {n_failed}.')


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
            anchor_inv,
            align_to_reference=config.get('align_to_reference', False)
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

    if not attrs['complete']:
        raise RuntimeError('Transform chaining is incomplete.')
    if attrs.get('final', False):
        # Transform were already shifted
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
        align_to_reference=False,
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

    attrs = get_store_attributes(dataset_trsf)
    if not attrs['complete']:
        raise RuntimeError(f'{dataset_name}: Raw transform computation incomplete')

    trsf_chained_path = os.path.join(zarr_path, 'z_intermediate', 'transform_chained', dataset_name)
    dataset_trsf_chained = open_store(
        trsf_chained_path, 
        mode='a', dtype=ts.float32,
        shape=dataset_trsf.shape, chunks=[1, 2, 3], axis_labels=['z', 'a', 'b'],
        fill_value=np.nan)
    
    attrs = get_store_attributes(dataset_trsf_chained)
    if attrs is not None and not attrs['raw_homography'] and attrs['complete']:
        # Chained transform was already performed and written to file, return
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
    
    if not align_to_reference:
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
    else:
        # Slices aligned to the reference need no chaining. Slices without reference were
        # aligned to a neighbour, and are chained to it.
        z_refs = read_transform_refs(zarr_path, dataset_name, len(transforms))
        if z_refs is None:
            z_refs = np.where(valid, TO_REFERENCE, NO_REF)

        # Chains may start from a slice of the previous stack, chained before this one
        carried = read_carried_from(zarr_path, dataset_name)
        carried_M = None
        if carried is not None and (z_refs == FROM_OTHER_STACK).any():
            carried_M = _read_chained_transform(zarr_path, carried['dataset_name'], carried['z'])

        chained_transforms = np.full_like(transforms, np.nan)
        for z in np.flatnonzero(valid):
            # Walk to the reference, then compose back. Chains are at most max_gap long.
            chain = [z]
            while z_refs[chain[-1]] >= 0 and len(chain) <= len(transforms):
                chain.append(z_refs[chain[-1]])
            start = {TO_REFERENCE: np.eye(3), FROM_OTHER_STACK: carried_M}.get(z_refs[chain[-1]])
            if not all(valid[c] for c in chain) or start is None or np.isnan(start).any():
                logging.warning(f'    {dataset_name}: Slice {z} is chained to a slice without transform '
                                f'(ignored after transforms were computed?), it will not be placed.')
                continue
            # As in the chain above: final[z] = final[z_ref] @ raw[z]
            M = start
            for c in chain[::-1]:
                M = M @ transforms[c]
            chained_transforms[z] = M
        valid &= ~np.isnan(chained_transforms).any(axis=(1, 2))

    # Bbox of this stack, from the slices that are rendered only. Invalid slices inherit the
    # previous transform (or are NaN with align_to_reference) and would only inflate it.
    stack_bbox = None
    if valid.any():
        img_shape = np.array(dataset.shape[1:]) * target_scale
        stack_bbox = update_bbox(None, chained_transforms[valid], img_shape, anchor_inv)
    else:
        logging.info(f'    {dataset_name}: No valid transform, the stack does not contribute to the output bbox.')

    # Update the global bbox
    bbox = update_bbox(bbox, chained_transforms, dataset.shape[1:], anchor_inv)

    # Write to file
    dataset_trsf_chained.write(chained_transforms[:, :2, :].astype(np.float32)).result()

    set_store_attributes(dataset_trsf_chained, 
                            {
                                'dataset_path': dataset_path,
                                'raw_transform_path': trsf_path,
                                'raw_homography': False,
                                'bbox': bbox,
                                'complete': True
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