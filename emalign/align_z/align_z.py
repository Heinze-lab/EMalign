import cv2
import logging
import jax
import jax.numpy as jnp
import numpy as np
import os
import tensorstore as ts

from connectomics.common import bounding_box
from sofima import flow_field, flow_utils, map_utils
from sofima.mesh import relax_mesh, IntegrationConfig
from tqdm import tqdm

from ..io.progress import check_progress, log_progress
from ..io.process.mask import compute_greyscale_mask
from ..io.store import find_ref_slice, open_store, write_ndarray, get_store_attributes, set_store_attributes
from ..arrays.utils import resample, homogenize_arrays_shape, pad_to_shape
from ..arrays.sift import estimate_transform_sift
from ..arrays.overlap import get_overlap_ref


PAD_OVERLAP = 1000


def _get_flow_stores(dataset_path, dataset_name, destination_path,
                    original_shape, scale, patch_size, stride,
                    ref_slice, transformations):
    '''Open existing flow/transform stores (with parameter validation) or create new ones.
    Returns (dataset_flow, dataset_trsf). dataset_trsf is None when transformations are provided.'''
    scale_str = str(round(scale, 2)).replace('.', '_')
    ds_flow_path = os.path.join(destination_path, 'z_intermediate', f'flow{scale_str}x', dataset_name)
    ds_trsf_path = os.path.join(destination_path, 'z_intermediate', 'transform', dataset_name)

    # Get flow dataset
    if os.path.exists(ds_flow_path):
        dataset_flow = open_store(ds_flow_path, mode='r+', dtype=ts.float32)
        attrs = get_store_attributes(dataset_flow)
        assert stride == attrs['stride'], 'stride does not correspond with existing flow'
        assert patch_size == attrs['patch_size'], 'patch_size does not correspond with existing flow'
        assert (ref_slice is not None) == attrs['external_first_slice'], 'ref slice does not correspond with existing flow'
    else:
        dataset_flow = open_store(
            ds_flow_path,
            mode='w',
            dtype=ts.float32,
            shape=[original_shape[0], 4, 1, 1],
            chunks=[1, 4, 128, 128],
            axis_labels=['z', 'c', 'y', 'x'],
            fill_value=np.nan
        )
        set_store_attributes(dataset_flow, {
            'dataset_path': dataset_path,
            'patch_size': patch_size,
            'stride': stride,
            'scale': scale,
            'external_first_slice': ref_slice is not None
        })

    # Get transformation dataset
    if transformations is not None:
        # Transformations may be provided already if this is a downsampled flow computation
        dataset_trsf = None
    elif os.path.exists(ds_trsf_path):
        dataset_trsf = open_store(ds_trsf_path, mode='r+', dtype=ts.float32)
    else:
        dataset_trsf = open_store(
            ds_trsf_path,
            mode='w',
            dtype=ts.float32,
            shape=[original_shape[0], 2, 4],
            chunks=[1, 2, 4],
            axis_labels=['z', 'a', 'b']
        )
        set_store_attributes(dataset_trsf, {
            'dataset_path': dataset_path,
            'scale': scale,
            'external_first_slice': ref_slice is not None
        })

    return dataset_flow, dataset_trsf



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
    # Different shapes may cause issues so we need to bring ref to the right shape without losing info.
    # Note that we don't want to change the shape of mov if we can avoid it because then we'd have to
    # keep track for the whole pipeline since the flow shape will have changed too.
    if np.any(np.array(mov.shape) > np.array(ref.shape)):
        # If ref is smaller, we pad to shape with zeros to the end of the array.
        # It doesn't affect offset.
        ref = pad_to_shape(ref, mov.shape)
        ref_mask = pad_to_shape(ref_mask, mov.shape)
    if np.any(np.array(ref.shape) > np.array(mov.shape)):
        # If ref is larger, we crop to shape.
        # ref and mov should be roughly overlapping, so we should not be losing relevant info.
        y, x = mov.shape
        ref = ref[:y, :x]
        ref_mask = ref_mask[:y, :x]

    assert (np.array(ref.shape) == np.array(mov.shape)).all()
    assert (np.array(ref_mask.shape) == np.array(mov_mask.shape)).all()
    assert np.any(ref_mask & mov_mask)

    return mfc.flow_field(ref, mov, (patch_size, patch_size),
                          (stride, stride), batch_size=batch_size,
                          pre_mask=~ref_mask, post_mask=~mov_mask,
                          mask_only_for_patch_selection=mask_only_for_patch_selection)


def _compute_flow(dataset,
                  patch_size,
                  stride,
                  scale,
                  db,
                  reference_dataset=None,
                  reference_offset=0,
                  original_shape=None,
                  ignore_slices=[],
                  destination_path=None,
                  dataset_mask=None,
                  ref_slice=None,
                  ref_slice_mask=None,
                  ref_scale=1,
                  transformations=None,
                  bbox_ref=None,
                  z_offset=0):

    original_shape = dataset.shape if original_shape is None else original_shape

    #---------- Resolve paths and mask ----------#
    dataset_path = os.path.abspath(dataset.kvstore.path)
    dataset_name = os.path.basename(dataset_path)
    if dataset_mask is None:
        ds_mask_path = dataset_path + '_mask'
        if os.path.exists(ds_mask_path):
            dataset_mask = open_store(ds_mask_path, mode='r')
    if destination_path is None:
        destination_path = os.path.dirname(dataset_path)
        while not destination_path.endswith('.zarr'):
            destination_path = os.path.dirname(destination_path)
    
    if reference_dataset is None:
        # Align to self if no reference is provided
        reference_dataset = dataset
        reference_dataset_mask = dataset_mask
        # No point in cropping the reference here, images should be roughly the same shape
        bbox_ref = [0, reference_dataset.shape[1],   
                    0, reference_dataset.shape[2]]
    else:
        ref_mask_path = os.path.abspath(reference_dataset.kvstore.path) + '_mask'
        if os.path.exists(ref_mask_path):
            reference_dataset_mask = open_store(ref_mask_path, mode='r')
        else:
            reference_dataset_mask = None
    ref_dataset_name = os.path.basename(os.path.abspath(reference_dataset.kvstore.path))

    #---------- Prepare destinations ----------#
    # Both transformations and flow are saved to file. They don't take much space but are slow to compute.
    dataset_flow, dataset_trsf = _get_flow_stores(
        dataset_path, dataset_name, destination_path,
        original_shape, scale, patch_size, stride, ref_slice, transformations)
    
    #---------- Determine where to start ----------#
    if ref_slice is not None and reference_dataset is not dataset:
        raise ValueError('An external reference slice was provided with a reference dataset (incompatible).')
    if reference_dataset is not dataset or ref_slice is not None:
        # All slices are aligned to a slice from an external reference
        start = dataset.domain.inclusive_min[0]
    else:
        # No external reference: first slice is the reference and will not be warped
        start = dataset.domain.inclusive_min[0] + 1
    
    #---------- Check Progress ----------#
    step_name = 'flow_z'
    flows = []
    transform = np.zeros([original_shape[0], 2, 4], dtype=np.float32)

    for z in range(start, dataset.domain.exclusive_max[0]):
        if check_progress(db, dataset_name, step_name, z, doc_filter={'scale': scale}):
            flows.append(dataset_flow[z].read().result())
            transform[z] = dataset_trsf[z].read().result() if transformations is None else transformations[z]

            if bbox_ref is None:
                # Get bbox from previous slices
                bbox_ref = db[dataset_name].find_one({'step_name': step_name, 'local_slice': z, 'scale': scale}, 
                                                     {'bbox_ref': 1})['bbox_ref']
                
    if len(flows) == (dataset.domain.exclusive_max[0] - start):
        # Everything appears to have been processed, early exit
        flows = homogenize_arrays_shape(flows, pad_value=np.nan)
        flows = np.transpose(flows, [1, 0, 2, 3])  # [channels, z, y, x]
        return flows, transform, bbox_ref
    
    if flows:
        # Resuming: start from after the last checkpointed slice
        start += len(flows)
        if reference_dataset is dataset:
            # Align using the last slice used for computing flow
            ref, z_ref = find_ref_slice(dataset, start - 1, reverse=True)
            ref = resample(ref, scale)
            if dataset_mask is not None:
                ref_mask = dataset_mask[z_ref].read().result()
                ref_mask = resample(ref_mask, scale)
            else:
                ref_mask = compute_greyscale_mask(ref, downsample_factor=16)
            t = transform[z_ref]
            ref = cv2.warpAffine(ref, t[:, :-1], t[:, -1].astype(int)[::-1])
            ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), t[:, :-1], t[:, -1].astype(int)[::-1]).astype(bool)
        logging.info(f'{dataset_name}: Skipping {len(flows)} already-processed slices')
    else:
        # Fresh start
        if ref_slice is None and reference_dataset is dataset:
            ref, z_ref = find_ref_slice(reference_dataset,
                                        reference_dataset.domain.inclusive_min[0],
                                        reverse=False)
            ref = resample(ref, ref_scale)
            if reference_dataset_mask is not None:
                ref_mask = reference_dataset_mask[z_ref].read().result()
                ref_mask = resample(ref_mask, ref_scale)
            else:
                ref_mask = compute_greyscale_mask(ref, downsample_factor=16)
            if reference_dataset is dataset:
                start = z_ref + 1  # start from after the actual reference slice
        else:
            # External reference slice provided (reference_dataset is dataset, validated above)
            z_ref = start
            ref = ref_slice
            ref_mask = ref_slice_mask

    #---------- Start processing ----------#
    mfc = flow_field.JAXMaskedXCorrWithStatsCalculator()

    pbar = tqdm(range(start, dataset.domain.exclusive_max[0]), position=0, dynamic_ncols=True)
    for z in pbar:
        if z in ignore_slices:
            pbar.set_description(f'{dataset_name}: Ignoring slice...')
            # Slice is to be ignored for flow computation based on user input.
            # These should not be used for mesh relaxation or they will bias the result, so we set them as invalid.
            flows.append(np.ones_like(flows[-1]) * np.nan)
            
            metadata = {
                'ref_dataset': ref_dataset_name,
                'scale': scale,
                'ref_scale': ref_scale,
                'skipped': True,
                'empty_slice': False
            }
            global_z = z + z_offset - dataset.domain.inclusive_min[0]
            log_progress(db, dataset_name, step_name, global_z, z, metadata)
            continue

        ##### MAIN LOOP #####
        pbar.set_description(f'{dataset_name}: Computing flow (scale={scale})')
        mov = dataset[z].read().result()

        # If empty slice, skip and compare to next one
        if not mov.any():
            # We should be starting with a non-empty slice, so by the time we hit this, flow should exist
            flows.append(np.ones_like(flows[-1]) * np.nan)
            metadata = {
                'ref_dataset': ref_dataset_name,
                'scale': scale,
                'ref_scale': ref_scale,
                'skipped': True,
                'empty_slice': True
            }
            global_z = z + z_offset - dataset.domain.inclusive_min[0]
            log_progress(db, dataset_name, step_name, global_z, z, metadata)
            continue

        mov = resample(mov, scale)
        if dataset_mask is not None:
            mov_mask = dataset_mask[z].read().result()
            mov_mask = resample(mov_mask, scale)
        else:
            mov_mask = compute_greyscale_mask(mov, downsample_factor=10)

        if reference_dataset is not dataset:
            # Prepare reference slice (global coordinates)
            z_ref = z + reference_offset + z_offset - dataset.domain.inclusive_min[0]

            # Find the first non-black image: in case z resolution is not the same in reference
            ref, z_ref = find_ref_slice(reference_dataset,
                                        z_ref, 
                                        reverse=False)
            ref = resample(ref, ref_scale)
            if reference_dataset_mask is not None:
                ref_mask = reference_dataset_mask[z_ref].read().result()
                ref_mask = resample(ref_mask, ref_scale)
            else:
                ref_mask = compute_greyscale_mask(ref, downsample_factor=16)

        # Transform mov to match ref
        if transformations is None:
            overlap_ref, overlap_ref_mask, bbox_ref = get_overlap_ref(ref, 
                                                                      mov, 
                                                                      ref_mask=ref_mask, 
                                                                      mov_mask=mov_mask,
                                                                      bbox_ref=bbox_ref,
                                                                      pad_overlap=PAD_OVERLAP)
            
            # Refine alignment
            # ref and mov have been resampled already so scale does not need to be accounted for
            M, output_shape, ref_xy_offset, valid_estimate, _ = estimate_transform_sift(overlap_ref, mov, 0.1, refine_estimate=True)
            if not valid_estimate:
                M, output_shape, ref_xy_offset, valid_estimate, _ = estimate_transform_sift(overlap_ref, mov, 0.3, refine_estimate=True)
            ref_xy_offset = ref_xy_offset.tolist()
            valid_estimate = bool(valid_estimate)

            # Adding patch size so we don't crop the corners if there is a rotation
            # This gets added at the end of the array
            output_shape = np.array(output_shape) + patch_size
        else:
            overlap_ref, overlap_ref_mask, bbox_ref = get_overlap_ref(ref, 
                                                                      mov, 
                                                                      ref_mask=ref_mask, 
                                                                      mov_mask=mov_mask,
                                                                      bbox_ref=bbox_ref,
                                                                      pad_overlap=PAD_OVERLAP)
            # Just get the overlap
            M = transformations[z][:, :-1]
            output_shape = transformations[z][:, -1].astype(int)
            ref_xy_offset = None
            valid_estimate = None
                
        t = np.concatenate([M, output_shape[None].T], axis=1).astype(np.float32)
        transform[z] = t

        # Warp data
        mov = cv2.warpAffine(mov, M, output_shape[::-1])
        mov_mask = cv2.warpAffine(mov_mask.astype(np.uint8), M, output_shape[::-1]).astype(bool)

        # Compute flow
        flow = _compute_flow_slice(overlap_ref, overlap_ref_mask, 
                                   mov, mov_mask, mfc, patch_size, stride)
        flows.append(flow)

        # Save to file + database
        dataset_flow, _ = write_ndarray(dataset_flow, flow, z, resolve=True)
        if transformations is None:
            dataset_trsf, _ = write_ndarray(dataset_trsf, t, z, resolve=False)

        # Log progress
        metadata = {
            'z_ref': z_ref,
            'reference_offset': reference_offset,
            'ref_dataset': ref_dataset_name,
            'flow_parameters': {
                'stride': stride,
                'patch_size': patch_size
            },
            'sift_xy_offset': ref_xy_offset,
            'valid_estimate': valid_estimate,
            'scale': scale,
            'ref_scale': ref_scale,
            'bbox_ref': bbox_ref,
            'pad_overlap': PAD_OVERLAP,
            'skipped': False,
            'empty_slice': False,
        }
        global_z = z + z_offset - dataset.domain.inclusive_min[0]
        log_progress(db, dataset_name, step_name, global_z, z, metadata)

        if reference_dataset is dataset:
            # Use this slice as ref for the next iteration
            ref = mov.copy()
            ref_mask = mov_mask.copy()
            z_ref = z

    jax.clear_caches()

    flows = homogenize_arrays_shape(flows, pad_value=np.nan)
    flows = np.transpose(flows, [1, 0, 2, 3])  # [channels, z, y, x]
    return flows, transform, bbox_ref


def compute_flow_dataset(dataset,
                         scale,
                         patch_size,
                         stride,
                         max_deviation,
                         max_magnitude,
                         db,
                         original_shape=None,
                         ignore_slices=[],
                         destination_path=None,
                         dataset_mask=None,
                         reference_dataset=None,
                         reference_offset=0,
                         bbox_ref=None,
                         ref_slice=None,
                         ref_slice_mask=None,
                         target_scale=1,
                         ref_scale=1,
                         z_offset=0):

    dataset_name = os.path.basename(os.path.abspath(dataset.kvstore.path))
    flow, transform, bbox_ref = _compute_flow(dataset=dataset,
                                              original_shape=original_shape,
                                              ignore_slices=ignore_slices,
                                              dataset_mask=dataset_mask,
                                              reference_dataset=reference_dataset,
                                              reference_offset=reference_offset,
                                              destination_path=destination_path,
                                              patch_size=patch_size,
                                              stride=stride,
                                              scale=target_scale,
                                              ref_scale=ref_scale,
                                              ref_slice=ref_slice,
                                              ref_slice_mask=ref_slice_mask,
                                              bbox_ref=bbox_ref,
                                              db=db,
                                              z_offset=z_offset)
    assert not np.isnan(flow).all()

    ds_transform = transform*np.array([[1,1,scale,scale], [1,1,scale,scale]])
    ds_bbox_ref = (np.array(bbox_ref) * scale).astype(int).tolist()
    ds_ref_slice = resample(ref_slice, scale) if ref_slice is not None else ref_slice
    ds_ref_slice_mask = resample(ref_slice_mask, scale) if ref_slice_mask is not None else ref_slice_mask
    ds_flow, _, _ = _compute_flow(dataset=dataset,
                                  original_shape=original_shape,
                                  ignore_slices=ignore_slices,
                                  dataset_mask=dataset_mask,
                                  reference_dataset=reference_dataset,
                                  reference_offset=reference_offset,
                                  destination_path=destination_path,
                                  patch_size=patch_size,
                                  stride=stride,
                                  scale=scale*target_scale,
                                  ref_scale=scale*ref_scale,
                                  ref_slice=ds_ref_slice,
                                  ref_slice_mask=ds_ref_slice_mask,
                                  transformations=ds_transform,
                                  bbox_ref=ds_bbox_ref,
                                  db=db,
                                  z_offset=z_offset)
    assert not np.isnan(ds_flow).all()

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
                  desc=f'{dataset_name}: Upsampling flow map',
                  dynamic_ncols=True):
        # Upsample and scale spatial components.
        resampled = map_utils.resample_map(
            ds_flow[:, z:z+1, ...],  #
            bbox_ds, bbox,
            1 / scale, 1)
        ds_flow_hires[:, z:z + 1, ...] = resampled / scale

    final_flow = flow_utils.reconcile_flows((flow, ds_flow_hires), max_gradient=0, max_deviation=max_deviation, min_patch_size=400)
    return final_flow, transform, bbox_ref


def get_inv_map(flow, stride, dataset_name, mesh_config=None, relax_xy=False):

    if mesh_config is None:
        mesh_config = IntegrationConfig(dt=0.001, gamma=0.5, k0=0.01, k=0.1, stride=(stride, stride), num_iters=1000,
                                            max_iters=100000, stop_v_max=0.005, dt_max=1000, start_cap=0.01,
                                            final_cap=10, prefer_orig_order=True)

    solved = [np.zeros_like(flow[:, 0:1, ...])]
    ref = solved[-1]
    origin = jnp.array([0., 0.])
    for z in tqdm(range(0, flow.shape[1]),
                  desc=f'{dataset_name}: Relaxing mesh',
                  dynamic_ncols=True):
        f = flow[:, z:z+1, ...]
        if np.isnan(f).all():
            # No flow was computed for this slice, ignore it for mesh relaxation
            # We keep the latest good slice as reference (ref)
            solved.append(np.zeros_like(f))
            continue
        
        if not relax_xy:
            # Use the previous iteration
            ref = map_utils.compose_maps_fast(f, origin, stride,
                                              ref, origin, stride)
            x, _, _ = relax_mesh(np.zeros_like(solved[0]), ref, mesh_config)
        else:
            # Use the previous iteration
            x, _, _ = relax_mesh(np.zeros_like(solved[0]), f, mesh_config)
        solved.append(np.array(x))
        ref = solved[-1]

    solved = np.concatenate(solved, axis=1)

    flow_bbox = bounding_box.BoundingBox(start=(0, 0, 0), size=(flow.shape[-1], flow.shape[-2], 1))

    inv_map = map_utils.invert_map(solved, flow_bbox, flow_bbox, stride)

    return inv_map, flow_bbox
