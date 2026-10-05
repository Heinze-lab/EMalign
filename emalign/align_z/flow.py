from concurrent.futures import ThreadPoolExecutor
import logging
import os

import cv2
from scipy import ndimage
import numpy as np
import tensorstore as ts
import jax.numpy as jnp
from tqdm import tqdm
from sofima import flow_field, flow_utils, map_utils
from sofima.mesh import relax_mesh, IntegrationConfig
from connectomics.common import bounding_box

from emalign.align_z.transform import (to_3x3, update_bbox,
                                       compute_global_bbox, shift_matrix)
from emalign.align_z.warp_hotfix import warp_affine_tiled
from emalign.io.utils import load_slice, load_reference_slice
from emalign.arrays.utils import pad_to_shape
from emalign.io.progress import get_mongo_client, get_mongo_db, log_progress, check_progress
from emalign.io.store import write_ndarray, open_store, get_store_attributes, set_store_attributes


SHRT_MAX = 32767  # Largest image cv2.warpAffine can handle


logging.basicConfig(level=logging.INFO)
logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('jax._src.xla_bridge').setLevel(logging.WARNING)


def combine_flow(
        flow, 
        ds_flow,
        stride,
        patch_size,
        max_magnitude,
        max_deviation,
        ds_scale,
        dataset_name
        ):
    pad = patch_size // 2 // stride
    flow = np.pad(flow, [[0, 0], [0, 0], [pad, pad], [pad, pad]], constant_values=np.nan)
    ds_flow = np.pad(ds_flow, [[0, 0], [0, 0], [pad, pad], [pad, pad]], constant_values=np.nan)

    flow = flow_utils.clean_flow(flow,
                                 min_peak_ratio=1.6,
                                 min_peak_sharpness=1.6,
                                 max_magnitude=max_magnitude,
                                 max_deviation=max_deviation)
    ds_flow = flow_utils.clean_flow(ds_flow,
                                    min_peak_ratio=1.6,
                                    min_peak_sharpness=1.6,
                                    max_magnitude=max_magnitude,
                                    max_deviation=max_deviation)
    ds_flow_hires = np.zeros_like(flow)

    bbox = bounding_box.BoundingBox(start=(0, 0, 0),
                                    size=(flow.shape[-1], flow.shape[-2], 1))
    bbox_ds = bounding_box.BoundingBox(start=(0, 0, 0),
                                       size=(ds_flow.shape[-1], ds_flow.shape[-2], 1))

    for z in tqdm(range(ds_flow.shape[1]),
                  desc=f'    Upsampling flow map',
                  dynamic_ncols=True):
        # Upsample and scale spatial components.
        resampled = map_utils.resample_map(
            ds_flow[:, z:z+1, ...],  #
            bbox_ds, bbox,
            1 / ds_scale, 1)
        ds_flow_hires[:, z:z + 1, ...] = resampled / ds_scale

    return flow_utils.reconcile_flows((flow, ds_flow_hires), max_gradient=0, max_deviation=max_deviation, min_patch_size=400)


def get_inv_map(flow,
                stride,
                mesh_config=None,
                relax_xy=False,
                init_map=None,
                start=(0., 0.),
                init_start=(0., 0.)):
    '''Relax the mesh of every slice of one stack, then invert them.

    Follows the same chain as compute_flow_stack: the flow of a slice is relative to the
    slice it was compared to, so its mesh is relaxed towards the state of that slice.
      - flow: [2, z, y, x] final flow of the stack. Slices without flow (all NaN: anchor,
        empty, invalid, or every vector rejected) keep the current state, which remains
        the reference for the next slice.
      - init_map: [2, 1, y', x'] solved map of the slice the first flow of this stack refers to,
        i.e. the anchor slice of the previous stack in the path. None for the identity (root stack).
      - start, init_start: yx origin of the flow grid and of the init_map grid in the output
        frame, in grid units (-shift / stride), so that stacks with different canvases match.
      - relax_xy: every slice was compared to a final reference (align_to_reference),
        no state is carried from slice to slice.
    Returns the inverse maps, the solved maps and the flow bounding box. The solved maps give
    the state to pass as init_map to the next stack of the path.
    '''

    if mesh_config is None:
        mesh_config = IntegrationConfig(dt=0.001, gamma=0.5, k0=0.01, k=0.1, stride=(stride, stride), num_iters=1000,
                                            max_iters=100000, stop_v_max=0.005, dt_max=1000, start_cap=0.01,
                                            final_cap=10, prefer_orig_order=True)

    start = jnp.asarray(start, dtype=jnp.float32)
    identity = np.zeros_like(flow[:, 0:1, ...])
    if init_map is None or relax_xy:
        state = identity
    else:
        # Bring the state of the previous stack onto the grid of this stack, by sampling it at
        # the nodes of this grid. compose_maps_fast is not used here: it ignores the origin of
        # the second map, so it is only correct when init_start <= start.
        offset = np.asarray(start, dtype=np.float64) - np.asarray(init_start, dtype=np.float64)  # yx, grid units
        yy, xx = np.mgrid[:identity.shape[2], :identity.shape[3]].astype(np.float64)
        coords = [yy + offset[0], xx + offset[1]]
        state = np.stack([ndimage.map_coordinates(np.asarray(init_map[c, 0], np.float64), coords,
                                                  order=1, mode='nearest')
                          for c in range(init_map.shape[0])])[:, None].astype(identity.dtype)

    solved = []
    for z in tqdm(range(flow.shape[1]),
                  desc=f'    Relaxing mesh',
                  dynamic_ncols=True):
        f = flow[:, z:z+1, ...]
        if np.isnan(f).all():
            # No flow for this slice, it keeps the current state, which stays the reference
            solved.append(identity if relax_xy else state)
            continue

        if relax_xy:
            # The reference is final, relax towards the flow only
            target = f
        else:
            # Relax towards the state of the slice this flow was computed against
            target = map_utils.compose_maps_fast(f, start, stride, state, start, stride)
        x, _, _ = relax_mesh(identity, target, mesh_config)
        x = np.array(x)
        solved.append(x)

        if not relax_xy:
            state = x

    solved = np.concatenate(solved, axis=1)

    flow_bbox = bounding_box.BoundingBox(start=(0, 0, 0), size=(flow.shape[-1], flow.shape[-2], 1))

    inv_map = map_utils.invert_map(solved, flow_bbox, flow_bbox, stride)

    return inv_map, solved, flow_bbox


def _read_final_transforms(zarr_path, dataset_name, scale):
    '''Read the final transforms of a stack as [z, 3, 3], rescaled to the flow resolution.'''
    trsf_path = os.path.join(zarr_path, 'z_intermediate', 'transform_chained', dataset_name)
    dataset_trsf = open_store(trsf_path, mode='r', dtype=ts.float32)
    attrs = get_store_attributes(dataset_trsf)
    if not attrs or not attrs.get('final', False):
        raise RuntimeError(f'{dataset_name}: Final transforms not found, run compute_transforms first.')

    transforms = to_3x3(dataset_trsf.read().result().astype(np.float64))
    # Transforms are in target resolution pixels, only the translation changes with scale
    transforms[:, :2, 2] *= scale
    return transforms


def _warp_slice(s, M, canvas_shape):
    '''Warp a loaded slice + mask into the flow canvas.'''
    if s['empty']:
        return s

    dsize = (int(canvas_shape[1]), int(canvas_shape[0]))
    warp = warp_affine_tiled if max(*s['img'].shape, *dsize) > SHRT_MAX else cv2.warpAffine
    img = warp(s['img'], M[:2], dsize)
    mask = warp(s['mask'].astype(np.uint8), M[:2], dsize).astype(bool)
    return s | {'img': img, 'mask': mask}


def _homogenise_ref_shape(ref, ref_mask, shape):
    '''Bring ref and its mask to the shape of mov, padding or cropping at the end of the array.'''
    # Different shapes may cause issues so we need to bring ref to the right shape without losing info.
    # Note that we don't want to change the shape of mov if we can avoid it because then we'd have to
    # keep track for the whole pipeline since the flow shape will have changed too.
    if np.any(np.array(shape) > np.array(ref.shape)):
        # If ref is smaller, we pad to shape with zeros to the end of the array.
        # It doesn't affect offset.
        ref = pad_to_shape(ref, shape)
        ref_mask = pad_to_shape(ref_mask, shape)
    if np.any(np.array(ref.shape) > np.array(shape)):
        # If ref is larger, we crop to shape.
        # ref and mov should be roughly overlapping, so we should not be losing relevant info.
        y, x = shape
        ref = ref[:y, :x]
        ref_mask = ref_mask[:y, :x]
    return ref, ref_mask


def _compute_flow_slice(ref, 
                        ref_mask, 
                        mov, 
                        mov_mask, 
                        mfc, 
                        patch_size, 
                        stride,
                        batch_size=128,
                        mask_only_for_patch_selection=False
                        ):
    '''Homogenise shapes between ref and mov, then compute the optical flow field.
    Returns flow array.'''
    ref, ref_mask = _homogenise_ref_shape(ref, ref_mask, mov.shape)

    assert (np.array(ref.shape) == np.array(mov.shape)).all()
    assert (np.array(ref_mask.shape) == np.array(mov_mask.shape)).all()
    assert np.any(ref_mask & mov_mask)

    return mfc.flow_field(ref, mov, (patch_size, patch_size),
                          (stride, stride), batch_size=batch_size,
                          pre_mask=~ref_mask, post_mask=~mov_mask,
                          mask_only_for_patch_selection=mask_only_for_patch_selection)


def _compute_flow_slice_blockwise(ref,
                                  ref_mask,
                                  mov,
                                  mov_mask,
                                  mfc,
                                  patch_size,
                                  stride,
                                  block_size=8192,
                                  batch_size=128,
                                  mask_only_for_patch_selection=False
                                  ):
    '''Same as _compute_flow_slice, but the flow field is computed block by block
    so that large images are not sent to the GPU at once.

    Each flow entry only depends on its own patch pair, so the output is identical to
    _compute_flow_slice as long as blocks start on the stride grid and overlap by
    patch_size - stride. ref and mov are assumed to be roughly aligned already.
    block_size (pixels) is rounded down to a multiple of stride.
    Returns flow array.'''
    ref, ref_mask = _homogenise_ref_shape(ref, ref_mask, mov.shape)

    assert (np.array(ref.shape) == np.array(mov.shape)).all()
    assert (np.array(ref_mask.shape) == np.array(mov_mask.shape)).all()
    assert np.any(ref_mask & mov_mask)

    # Output grid, as computed by sofima: entry i is the patch starting at i * stride
    out_shape = (np.array(mov.shape) - (patch_size - stride)) // stride
    n_channels = mfc.non_spatial_flow_channels + mov.ndim
    flow = np.full([n_channels, *out_shape], np.nan, dtype=np.float32)

    # Number of flow entries per block along each axis
    block_out = max(1, (block_size - patch_size) // stride + 1)

    for oy in range(0, out_shape[0], block_out):
        for ox in range(0, out_shape[1], block_out):
            oy1 = min(oy + block_out, out_shape[0])
            ox1 = min(ox + block_out, out_shape[1])
            # Pixels needed for the patches of this block
            y0, y1 = oy * stride, (oy1 - 1) * stride + patch_size
            x0, x1 = ox * stride, (ox1 - 1) * stride + patch_size
            block_ref_mask = ref_mask[y0:y1, x0:x1]
            block_mov_mask = mov_mask[y0:y1, x0:x1]
            if not block_ref_mask.any() or not block_mov_mask.any():
                # All patches would be rejected by sofima, leave NaN
                continue

            flow[:, oy:oy1, ox:ox1] = mfc.flow_field(
                ref[y0:y1, x0:x1], mov[y0:y1, x0:x1],
                (patch_size, patch_size), (stride, stride),
                batch_size=batch_size,
                pre_mask=~block_ref_mask, post_mask=~block_mov_mask,
                mask_only_for_patch_selection=mask_only_for_patch_selection)

    return flow


def compute_flow_stack(dataset_path,
                       dataset_name,
                       destination_path,
                       z_offset,
                       yx_target_resolution,
                       flow_config,
                       scale=1,
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
    '''Compute and store the optical flow for every slice of one stack.

    Slices are warped with the final transforms of compute_transforms_stack, onto
    a canvas covering the stack's footprint in the output frame. Each slice is
    compared to the same anchor it was aligned to for the transforms:
      - no reference_dataset: the anchor is the first valid slice of this stack,
        it gets no flow and every slice follows the previous valid one.
      - reference_dataset: the anchor is the last non-empty slice of the reference,
        then every slice still follows the previous valid one of this stack.
      - reference_dataset + align_to_reference: every slice is compared to its
        matching slice in the reference, found at z + reference_offset.
    Slices without a valid transform (empty, ignored or failed) get no flow.

    scale is applied on top of yx_target_resolution, so the flow can also be
    computed at a lower resolution. Each scale gets its own flow store.
    '''

    if num_workers > 0:
        # Set the number of workers used by cv2 to warp slices
        cv2.setNumThreads(max(1, num_workers - 1))

    # Progress logging
    db = get_mongo_db(get_mongo_client(mongodb_config_filepath), project_name)
    step_name = 'flow_z'

    patch_size = flow_config['patch_size']
    stride = flow_config['stride']

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
    mov_flow_scale = scale * target_scale

    # Get reference store, it may have its own resolution
    if reference_dataset is None:
        # Align to self, no reference provided
        if align_to_reference:
            raise ValueError('align_to_reference requires a reference_dataset.')
        reference = reference_mask = None
        ref_flow_scale = mov_flow_scale
    else:
        # Align to external reference dataset
        reference_dataset = os.path.abspath(reference_dataset)
        reference = open_store(reference_dataset, mode='r', dtype=ts.uint8)
        reference_mask = open_store(reference_dataset + '_mask', mode='r',
                                    dtype=ts.bool, allow_missing=True)
        ref_res = get_store_attributes(reference)['resolution'][-1]
        ref_scale = 1 if yx_target_resolution is None else ref_res / yx_target_resolution
        ref_flow_scale = scale * ref_scale

        if not align_to_reference:
            # We only align to the last slice of the reference, as an anchor
            # Transforms are already at target scale so we only need to scale them for the flow
            ref_transforms = _read_final_transforms(zarr_path, os.path.basename(reference_dataset), scale)
        else:
            # We assume that the reference is otherwise final
            # Transforms are already at target scale so we only need to scale them for the flow
            ref_transforms = np.eye(3, dtype=np.float32)
            ref_transforms[:2, 2] *= scale
            ref_transforms = np.repeat(ref_transforms[None, ...], reference.shape[0], axis=0)

    z_min = local_z_min if local_z_min is not None else dataset.domain.inclusive_min[0]
    z_max = local_z_max if local_z_max is not None else dataset.domain.exclusive_max[0]
    ignore_slices = set(ignore_slices)

    # Get transforms
    # First check raw transforms for valid slices (non-Nan)
    raw_trsf_path = os.path.join(zarr_path, 'z_intermediate', 'transform', dataset_name)
    raw_transforms = open_store(raw_trsf_path, mode='r', dtype=ts.float32).read().result()
    valid = ~np.isnan(raw_transforms).any(axis=(1, 2))
    valid[[z for z in ignore_slices if z < len(valid)]] = False
    if not valid[z_min:z_max].any():
        raise RuntimeError(f'{dataset_name}: No valid transform found, run compute_transforms first.')
    # Then read final transforms
    transforms = _read_final_transforms(zarr_path, dataset_name, scale)

    # Flow is computed only within a bbox containing the relevant image
    # Get the bounding box of the warped image from the transform
    img_shape = np.round(np.array(dataset.shape[1:]) * mov_flow_scale)
    bbox = update_bbox(None, transforms[valid], img_shape, np.eye(3))

    # Shift the transforms so that they place the cropped image in the correct area
    canvas_shape, (shift_y, shift_x) = compute_global_bbox(bbox)
    shift = shift_matrix(shift_x, shift_y)
    transforms = shift @ transforms
    if reference is not None:
        ref_transforms = shift @ ref_transforms

    # Create flow store at the root, one per scale
    scale_str = str(round(scale, 2)).replace('.', '_')
    flow_path = os.path.join(zarr_path, 'z_intermediate', f'flow{scale_str}x', dataset_name)
    flow_shape = (np.array(canvas_shape) - (patch_size - stride)) // stride  # Same as sofima's flow_field
    
    dataset_flow = open_store(
        flow_path,
        mode='a', dtype=ts.float32,
        shape=[z_max, 4, *flow_shape.tolist()], chunks=[1, 4, 128, 128], axis_labels=['z', 'c', 'y', 'x'],
        fill_value=np.nan)

    def write_f(z, flow):
        nonlocal dataset_flow
        dataset_flow, _ = write_ndarray(dataset_flow, np.asarray(flow, np.float32),
                                        z, resolve=False)
    
    def _validate_flow_attrs(attrs):
        if stride != attrs['stride']:
            raise ValueError('{}: Stride ({}) does not correspond with existing flow ({})'.format(dataset_name, stride, attrs['stride']))
        elif patch_size != attrs['patch_size']:
            raise ValueError('{}: Patch size ({}) does not correspond with existing flow ({})'.format(dataset_name, patch_size, attrs['patch_size']))
        elif shift_y != attrs['shift_y']:
            raise ValueError('{}: Shift y ({}) does not correspond with existing flow ({})'.format(dataset_name, shift_y, attrs['shift_y']))
        elif shift_x != attrs['shift_x']:
            raise ValueError('{}: Shift x ({}) does not correspond with existing flow ({})'.format(dataset_name, shift_x, attrs['shift_x']))
        elif list(canvas_shape) != attrs['canvas_shape']:
            raise ValueError('{}: Canvas shape ({}) does not correspond with existing flow ({})'.format(dataset_name, canvas_shape, attrs['canvas_shape']))


    # ---------- Check progress ----------
    first_z = None
    for z in range(z_min, z_max):
        if not check_progress(db, dataset_name, step_name, z, doc_filter={'scale': scale}):
            first_z = z
            break
    if first_z is None:
        # Everything was processed already
        logging.info(f'    All flows were already computed (scale={scale}).')
        return

    # Resume from the last slice holding a valid transform
    previous = [z for z in range(z_min, first_z) if valid[z]]
    resume = len(previous) > 0
    if resume:
        # Check parameters
        attrs = get_store_attributes(dataset_flow)
        if attrs is None:
            raise RuntimeError(f'No attribute file for {flow_path}')
        _validate_flow_attrs(attrs)

        # Get the correct ref slice
        z_ref = previous[-1]
        if not align_to_reference:
            # Load the corresponding slice within this dataset
            ref_slice = _warp_slice(load_slice(dataset, dataset_mask, z_ref, mov_flow_scale),
                                    transforms[z_ref], canvas_shape)
        else:
            # Load the corresponding slice in the reference dataset
            ref_slice = load_reference_slice(reference, reference_mask, first_z + reference_offset,
                                              ref_flow_scale)
            ref_slice = _warp_slice(ref_slice, ref_transforms[ref_slice['z']], canvas_shape)

    # ---------- First slice to process ----------
    if not resume:
        # Skip until the first valid slice
        while not valid[first_z]:
            log_progress(db, dataset_name, step_name, first_z + z_offset - z_min, first_z,
                            {'scale': scale, 'skipped': True})
            first_z += 1

        if reference_dataset is None:
            # No reference, the first slice is the anchor and gets no flow
            ref_slice = _warp_slice(load_slice(dataset, dataset_mask, first_z, mov_flow_scale),
                                    transforms[first_z], canvas_shape)
            log_progress(db, dataset_name, step_name, first_z + z_offset - z_min, first_z,
                        {'anchor': True, 'scale': scale, 'skipped': False})
            first_z += 1  # Start the loop with the next slice
        elif not align_to_reference:
            # One slice for anchor and that's it
            # Get the last non-empty slice of the reference as anchor, as for the transforms
            ref_slice = load_reference_slice(reference, reference_mask, None,
                                              ref_flow_scale, reverse=True)
            ref_slice = _warp_slice(ref_slice, ref_transforms[ref_slice['z']], canvas_shape)
        else:
            # The whole dataset will be aligned to the reference
            # We look for the slice at the right offset
            ref_slice = load_reference_slice(reference, reference_mask, first_z + reference_offset,
                                              ref_flow_scale)
            ref_slice = _warp_slice(ref_slice, ref_transforms[ref_slice['z']], canvas_shape)

        # Set attributes
        attrs = get_store_attributes(dataset_flow)
        if attrs is not None:
            _validate_flow_attrs(attrs)
        else:
            set_store_attributes(
                dataset_flow,
                {
                    'dataset_path': dataset_path,
                    'scale': scale,
                    'patch_size': patch_size,
                    'stride': stride,
                    'canvas_shape': list(canvas_shape),
                    'shift_y': shift_y,  # Output frame -> canvas, in flow resolution pixels
                    'shift_x': shift_x,
                    'external_first_slice': reference_dataset is not None,
                    'reference_path': reference_dataset,
                    'reference_offset': int(reference_offset),
                    'align_to_reference': bool(align_to_reference),
                    'ref_scale': ref_flow_scale,
                    'first_z': int(first_z)
            })

    # ---------- Start processing ----------
    mfc = flow_field.JAXMaskedXCorrWithStatsCalculator()

    # Cache slices for processing
    read_pool = ThreadPoolExecutor(max_workers=1)   
    window = 4
    read_futures = {}
    next_read = first_z

    def prefetch_data(up_to):
        nonlocal next_read
        up_to = min(up_to, z_max - 1)
        while next_read <= up_to:
            read_futures[next_read] = read_pool.submit(
                load_slice, dataset, dataset_mask, next_read, mov_flow_scale)
            next_read += 1

    # Prefetch the first slice
    prefetch_data(first_z + window)

    n_gap = 0
    n_ignored = 0
    for z in tqdm(
        range(first_z, z_max), 
        total=z_max - first_z, 
        desc=f'    Computing flow ({scale})', 
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
                         {'scale': scale, 'empty_slice': bool(mov_slice['empty']), 'skipped': True})
        elif not valid[z]:
            # Invalid slice, or ignored by user
            n_ignored += 1
            log_progress(db, dataset_name, step_name, global_z, z,
                         {'scale': scale, 'valid': False, 'skipped': True})
        else:
            # Process the data
            mov_slice = _warp_slice(mov_slice, transforms[z], canvas_shape)

            # Compute flow
            flow = _compute_flow_slice_blockwise(
                ref_slice['img'], ref_slice['mask'], 
                mov_slice['img'], mov_slice['mask'],
                mfc, patch_size, stride, 
                block_size=8192, batch_size=128, mask_only_for_patch_selection=False)

            # Write data
            write_f(z, flow)            
            log_progress(db, dataset_name, step_name, global_z, z,
                     {'scale': scale, 'mov_scale': mov_flow_scale,
                      'z_ref': int(ref_slice['z']), 'ref_scale': ref_flow_scale,
                      'skipped': False})
            
            if not align_to_reference:
                # Align to self, only use the mov_slice as ref if it was valid
                ref_slice = mov_slice
        
        if align_to_reference and z + 1 < z_max:
            # Get ref for the next iteration, regardless of whether this slice was valid
            # We assume a one-to-one correspondence along Z
            ref_slice = load_reference_slice(reference, reference_mask, z + reference_offset + 1,
                                              ref_flow_scale)
            ref_slice = _warp_slice(ref_slice, ref_transforms[ref_slice['z']], canvas_shape)
        
     # Shutdown the read pool
    read_pool.shutdown()

    # Mark stack transforms as complete
    attrs = get_store_attributes(dataset_flow)
    attrs['complete'] = True
    set_store_attributes(dataset_flow, attrs)

    logging.info(f'    Flow computation done (scale={scale}). | Ignored slices: {n_ignored}. | Empty slices: {n_gap}.')
