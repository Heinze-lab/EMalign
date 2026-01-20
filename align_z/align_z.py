from time import sleep
import cv2
from emalign.io.progress import check_progress, log_progress
import logging
import jax
import jax.numpy as jnp
import numpy as np
import os
import tensorstore as ts

from connectomics.common import bounding_box
from sofima import flow_field, flow_utils, map_utils, mesh
from sofima.mesh import velocity_verlet, inplane_force, IntegrationConfig
from tqdm import tqdm

from emprocess.utils.mask import compute_greyscale_mask
from ..io.store import find_ref_slice, open_store, write_ndarray, get_store_attributes, set_store_attributes
from ..arrays.utils import resample, homogenize_arrays_shape, pad_to_shape
from ..arrays.sift import estimate_transform_sift


def _compute_flow(dataset,
                  patch_size,
                  stride,
                  scale,
                  db,
                  original_shape=None,
                  ignore_slices=[],
                  destination_path=None,
                  dataset_mask=None,
                  ref_slice=None,
                  ref_slice_mask=None,
                  transformations=None,
                  z_offset=0):
    
    mfc = flow_field.JAXMaskedXCorrWithStatsCalculator()
    original_shape = dataset.shape if original_shape is None else original_shape

    #---------- Open dataset ----------#
    dataset_name = os.path.basename(os.path.abspath(dataset.kvstore.path))
    if dataset_mask is None:
        ds_mask_path = os.path.abspath(dataset.kvstore.path) + '_mask'
        if os.path.exists(ds_mask_path):
            dataset_mask = open_store(ds_mask_path, mode='r')
    if destination_path is None:
        destination_path = os.path.dirname(os.path.abspath(dataset.kvstore.path))
            
    #---------- Prepare destinations ----------#
    # Both transformations and flow are saved to file. They don't take much space but are slow to compute.
    scale_str = str(round(scale, 2)).replace('.', '_')
    ds_flow_path = os.path.join(destination_path,
                                f'flow{scale_str}x',
                                dataset_name)
    ds_trsf_path = os.path.join(destination_path,
                                'transform',
                                dataset_name)
    if os.path.exists(ds_flow_path):
        # Flow + Transformations exist
        dataset_flow = open_store(ds_flow_path, mode='r+', dtype=ts.float32)
        attrs = get_store_attributes(dataset_flow)
        assert stride == attrs['stride'], 'stride does not correspond with existing flow'
        assert patch_size == attrs['patch_size'], 'patch_size does not correspond with existing flow'
        assert (ref_slice is not None) == attrs['external_first_slice'], 'ref slice does not correspond with existing flow'
        
        # If flow dataset exists but transformations is None, we assume we can find it in a dataset
        if transformations is None:
            dataset_trsf = open_store(ds_trsf_path, mode='r+', dtype=ts.float32)
    else:
        # Flow + Transformations are to be created from scratch
        dataset_flow = open_store(
            ds_flow_path,
            mode='w',
            dtype=ts.float32,
            shape=[original_shape[0], 4, 1, 1],
            chunks=[1, 4, 128, 128],
            axis_labels=['z', 'c', 'y', 'x'],
            fill_value=np.nan
        )
        attrs = {
            'dataset_path': os.path.abspath(dataset.kvstore.path),
            'patch_size': patch_size,
            'stride': stride,
            'scale': scale,
            'external_first_slice': ref_slice is not None
                }
        set_store_attributes(dataset_flow, attrs)
        
        # No transformation exist, we will compute it
        if transformations is None:
            dataset_trsf = open_store(
                ds_trsf_path,
                mode='w',
                dtype=ts.float32,
                shape=[original_shape[0], 2, 4],
                chunks=[1, 2, 4],
                axis_labels=['z', 'a', 'b']
            )
            attrs = {
                'dataset_path': os.path.abspath(dataset.kvstore.path),
                'scale': scale,
                'external_first_slice': ref_slice is not None
                    }
            set_store_attributes(dataset_trsf, attrs)
    
    #---------- Check Progress ----------#
    step_name = f'flow_z'
    collection = db[dataset_name]
    n_docs = dataset.shape[0] - int(ref_slice is None)
    if collection.count_documents({'step_name': step_name, 'scale': scale}) >= n_docs:
        flows = np.transpose(dataset_flow.read().result(), [1, 0, 2, 3])
        if transformations is None:
            transform = dataset_trsf.read().result()
        else:
            transform = transformations
        return flows, transform
        
    #---------- Start processing ----------#
    # Get reference slice
    if ref_slice is None:
        # Use dataset's first slice to compute flow from.
        prev, z = find_ref_slice(dataset, 
                                  dataset.domain.inclusive_min[0], 
                                  reverse=False)
        if dataset_mask is not None:
            prev_mask = dataset_mask[z].read().result()
        else:
            prev_mask = compute_greyscale_mask(prev)  
        # Downsample if needed
        prev = resample(prev, scale)
        prev_mask = resample(prev_mask, scale)
        z_prev = z
        start = z + 1      
    else:
        # Use provided first slice to compute flow from. Could be slice of a previous dataset
        # We assume that the reference slice is already at the right scale
        start = dataset.domain.inclusive_min[0]
        prev = ref_slice
        prev_mask = ref_slice_mask
        z_prev = start

    # Iterate through slices 
    flows = []
    transform = np.zeros([original_shape[0],2,4], dtype=np.float32)
    pickup_progress = False
    pbar = tqdm(range(start, dataset.domain.exclusive_max[0]), position=0, dynamic_ncols=True)
    for z in pbar:
        ##### CHECKPOINT #####
        if check_progress(db, dataset_name, step_name, z, doc_filter = {'scale': scale}):
            # If slice was processed, we read the flow and transform, or get transform from input
            pbar.set_description(f'{dataset_name}: Skipping...')
            pickup_progress = True # Get ref for first valid slice
            flow = dataset_flow[z].read().result()
            flows.append(flow)
            if transformations is None:
                t = dataset_trsf[z].read().result()
                transform[z] = t
            else:
                t = transformations[z]
                transform[z] = t
            continue
        elif z != start and pickup_progress:
            # First unprocessed slice after checkpoint. We need a reference slice.
            pickup_progress = False # No need to visit this segment anymore

            prev, z_prev = find_ref_slice(dataset, 
                                          z-1, 
                                          reverse=True)
            prev = resample(prev, scale)
            pbar.set_description(f'Preparing previous slice ({z_prev})')
            if dataset_mask is not None:
                prev_mask = dataset_mask[z_prev].read().result()
                prev_mask = resample(prev_mask, scale)
            else:
                prev_mask = compute_greyscale_mask(prev)

            if transformations is None:
                t = dataset_trsf[z_prev].read().result()                
            else:
                t = transformations[z_prev]

            M = t[:, :-1]
            output_shape = t[:, -1].astype(int)
            prev = cv2.warpAffine(prev, 
                                  M, output_shape[::-1])
            prev_mask = cv2.warpAffine(prev_mask.astype(np.uint8), 
                                       M, output_shape[::-1]).astype(bool)
            sleep(3) # To make sure that we see the message (mostly for debug)
        
        if z in ignore_slices:
            pbar.set_description(f'{dataset_name}: Ignoring slice...')
            # Slice is to be ignored for flow computation based on user input
            # These should not be used for mesh relaxation or they will bias the result, so we set them as invalid
            if z != start:
                flows.append(np.ones_like(flow)*np.nan)

            # Log progress
            metadata = {
                'z_prev': z_prev,
                'scale': scale,
                'skipped': True,
                'empty_slice': False
            }
            global_z = z + z_offset - dataset.domain.inclusive_min[0]
            log_progress(db, dataset_name, step_name, global_z, z, metadata)
            continue

        ##### MAIN LOOP #####
        pbar.set_description(f'{dataset_name}: Computing flow (scale={scale})')
        curr = dataset[z].read().result()
        
        # If empty slice, skip and compare to next one
        if not curr.any():
            # We should be starting with a non-empty slice, so by the time we hit this, flow should exist
            flows.append(np.empty_like(flow) * np.nan)
            # Log progress
            metadata = {
                'z_prev': z_prev,
                'scale': scale,
                'skipped': True,
                'empty_slice': True
            }
            global_z = z + z_offset - dataset.domain.inclusive_min[0]
            log_progress(db, dataset_name, step_name, global_z, z, metadata)
            continue
        
        curr = resample(curr, scale)

        # If no mask exists, compute it
        if dataset_mask is not None:
            curr_mask = dataset_mask[z].read().result()
            curr_mask = resample(curr_mask, scale)
        else:
            curr_mask = compute_greyscale_mask(curr)
        
        # Transform curr to match prev
        if transformations is None:
            M, output_shape, ref_offset, valid_estimate, _ = estimate_transform_sift(prev, curr, 0.1/scale, refine_estimate=True)
            if not valid_estimate:
                M, output_shape, ref_offset, valid_estimate, _ = estimate_transform_sift(prev, curr, 0.3/scale, refine_estimate=True)
            ref_offset = ref_offset.tolist()
            valid_estimate = bool(valid_estimate)
        else:
            M = transformations[z][:, :-1]
            output_shape = transformations[z][:, -1].astype(int)
            ref_offset = None 
            valid_estimate = None
        t = np.concatenate([M, output_shape[None].T], axis=1).astype(np.float32)
        transform[z] = t

        # Warp data
        curr = cv2.warpAffine(curr, M, output_shape[::-1])
        curr_mask = cv2.warpAffine(curr_mask.astype(np.uint8), M, output_shape[::-1]).astype(bool)

        try:
            # Different shapes may cause issues so we need to bring prev to the right shape without losing info
            # Note that we don't want to change the shape of curr if we can avoid it because then we'd have to keep track for 
            # the whole pipeline since the flow shape will have changed too.
            if np.any(np.array(curr.shape) > np.array(prev.shape)):
                # If prev is smaller, we pad to shape with zeros to the end of the data
                # It doesn't affect offset
                prev = pad_to_shape(prev, curr.shape)
                prev_mask = pad_to_shape(prev_mask, curr.shape)
            if np.any(np.array(prev.shape) > np.array(curr.shape)):
                # If prev is larger, we crop to shape
                # Prev and curr should be roughly overlapping, so we should not be losing relevant info
                y,x = curr.shape
                prev = prev[:y, :x]
                prev_mask = prev_mask[:y, :x]
                
            assert (np.array(prev.shape) == np.array(curr.shape)).all()
            assert (np.array(prev_mask.shape) == np.array(curr_mask.shape)).all()
            assert np.any(prev_mask & curr_mask)

            # Compute flow
            flow = mfc.flow_field(prev, curr, (patch_size, patch_size),
                                  (stride, stride), batch_size=128,
                                  pre_mask=~prev_mask, post_mask=~curr_mask)
            flows.append(flow)

            # Save to file + database
            dataset_flow, _ = write_ndarray(dataset_flow, flow, z, resolve=True)
            if transformations is None:
                dataset_trsf, _ = write_ndarray(dataset_trsf, t, z, resolve=False)

            # Log progress
            metadata = {
                'z_prev': z_prev,
                'flow_parameters':{
                                'stride':stride,
                                'patch_size':patch_size
                                },
                'ref_offset': ref_offset, 
                'valid_estimate': valid_estimate,
                'scale': scale,
                'skipped': False,
                'empty_slice': False,
            }
            global_z = z + z_offset - dataset.domain.inclusive_min[0]
            log_progress(db, dataset_name, step_name, global_z, z, metadata)
            
            prev = curr.copy()
            prev_mask = curr_mask.copy()
            z_prev = z
        except Exception as e:
            raise RuntimeError(e)
        
    jax.clear_caches()

    flows = homogenize_arrays_shape(flows, pad_value=np.nan)
    flows = np.transpose(flows, [1, 0, 2, 3]) # [channels, z, y, x]
    return flows, transform


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
                         ref_slice=None,
                         ref_slice_mask=None,
                         target_scale=1,
                         z_offset=0):
    
    dataset_name = os.path.basename(os.path.abspath(dataset.kvstore.path))
    flow, transform = _compute_flow(dataset=dataset,
                                    original_shape=original_shape,
                                    ignore_slices=ignore_slices,
                                    dataset_mask=dataset_mask, 
                                    destination_path=destination_path,
                                    patch_size=patch_size, 
                                    stride=stride, 
                                    scale=1*target_scale, 
                                    ref_slice=ref_slice, 
                                    ref_slice_mask=ref_slice_mask,
                                    db=db,
                                    z_offset=z_offset)
    assert not np.isnan(flow).all()

    ds_transform = transform*np.array([[1,1,scale,scale], [1,1,scale,scale]])
    ds_ref_slice = resample(ref_slice, scale) if ref_slice is not None else ref_slice
    ds_ref_slice_mask = resample(ref_slice_mask, scale) if ref_slice_mask is not None else ref_slice_mask
    ds_flow, _ = _compute_flow(dataset=dataset,
                               original_shape=original_shape,
                               ignore_slices=ignore_slices,
                               dataset_mask=dataset_mask, 
                               destination_path=destination_path,
                               patch_size=patch_size, 
                               stride=stride, 
                               scale=scale*target_scale, 
                               ref_slice=ds_ref_slice, 
                               ref_slice_mask=ds_ref_slice_mask,
                               transformations=ds_transform,
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
    return final_flow, transform


def get_inv_map(flow, stride, dataset_name, mesh_config=None):

    if mesh_config is None:
        mesh_config = mesh.IntegrationConfig(dt=0.001, gamma=0.5, k0=0.01, k=0.1, stride=(stride, stride), num_iters=1000,
                                            max_iters=100000, stop_v_max=0.005, dt_max=1000, start_cap=0.01,
                                            final_cap=10, prefer_orig_order=True)

    solved = [np.zeros_like(flow[:, 0:1, ...])]
    origin = jnp.array([0., 0.])

    for z in tqdm(range(0, flow.shape[1]),
                  desc=f'{dataset_name}: Relaxing mesh',
                  dynamic_ncols=True):
        prev = map_utils.compose_maps_fast(flow[:, z:z+1, ...], origin, stride,
                                           solved[-1], origin, stride)
        x, _, _ = mesh.relax_mesh(np.zeros_like(solved[0]), prev, mesh_config)
        solved.append(np.array(x))

    solved = np.concatenate(solved, axis=1)

    flow_bbox = bounding_box.BoundingBox(start=(0, 0, 0), size=(flow.shape[-1], flow.shape[-2], 1))

    inv_map = map_utils.invert_map(solved, flow_bbox, flow_bbox, stride)

    return inv_map, flow_bbox



def relax_mesh_mod(
    x: jnp.ndarray,
    prev: jnp.ndarray | None,
    config: IntegrationConfig,
    mesh_force=inplane_force,
    prev_fn=None,
) -> tuple[jnp.ndarray, list[float], int]:
  """Simulates mesh relaxation.

  Args:
    x: [2, z, y, x] array of mesh node positions
    prev: optional [2, z, y, x] array against which to compute the force due to
      0-length springs
    config: simulation parameters
    mesh_force: callable with the signature of `inplane_force` returning a field
      representing internal mesh forces
    prev_fn: callable taking the 'x' mesh array and returning the 'prev' array

  Returns:
    tuple of:
      [2, z, x, y] array of updated mesh positions
      list of kinetic energy history
      number of simulation steps executed
  """

  t = 0
  v = jnp.zeros_like(x)
  dt = config.dt
  alpha = config.alpha
  e_kin = []
  cap = config.start_cap

  if config.start_cap != config.final_cap:
    if not config.fire:
      raise NotImplementedError(
          'Adaptive force capping is only supported with FIRE.'
      )
    if config.cap_scale <= 1:
      raise ValueError(
          'The scaling factor for the force cap has to be larger '
          'than 1 when the initial and final cap are different.'
      )

  if prev is not None and prev_fn is not None:
    raise ValueError('Only one of: "prev" and "prev_fn" can be specified.')
  
  while t < config.max_iters:
    state = velocity_verlet(
        x,
        v,
        prev,
        config,
        fire_dt=dt,
        fire_alpha=alpha,
        force_cap=cap,
        mesh_force=mesh_force,
        prev_fn=prev_fn,
    )
    t += config.num_iters
    x, v = state[:2]
    v_mag = jnp.linalg.norm(v, axis=0)
    e_kin.append(float(jnp.sum(v_mag**2)))
    v_max = jnp.max(v_mag)

    if config.fire:
      dt, alpha, n_pos, cap = state[-4:]
    #   logging.info(
    #       't=%r: dt=%f, alpha=%f, n_pos=%d, cap=%f, v_max=%f, e_kin=%f',
    #       t,
    #       dt,
    #       alpha,
    #       n_pos,
    #       cap,
    #       v_max,
    #       e_kin[-1],
    #   )

    if v_max < config.stop_v_max:
      if cap >= config.final_cap:
        break

      # Increase cap to ensure progress towards the termination condition.
      cap = min(cap * config.cap_scale, config.final_cap)

  return x, e_kin, t, v_max # mod



def get_inv_map_mod(flow, stride, dataset_name, mesh_config=None):

    if mesh_config is None:
        mesh_config = mesh.IntegrationConfig(dt=0.001, gamma=0.5, k0=0.01, k=0.1, stride=(stride, stride), num_iters=1000,
                                            max_iters=100000, stop_v_max=0.005, dt_max=1000, start_cap=0.01,
                                            final_cap=10, prefer_orig_order=True)

    solved = [np.zeros_like(flow[:, 0:1, ...])]
    ref = solved[-1]
    origin = jnp.array([0., 0.])
    all_vmax = []
    for z in tqdm(range(0, flow.shape[1]),
                  desc=f'{dataset_name}: Relaxing mesh',
                  dynamic_ncols=True):
        f = flow[:, z:z+1, ...]
        if np.isnan(f).all():
            # No flow was computed for this slice, ignore it for mesh relaxation
            # We keep the latest good slice as reference (ref)
            solved.append(np.zeros_like(f))
            continue

        ref = map_utils.compose_maps_fast(f, origin, stride,
                                           ref, origin, stride)
        x, _, _, v_max = relax_mesh_mod(np.zeros_like(solved[0]), ref, mesh_config)
        solved.append(np.array(x))
        all_vmax.append(v_max)
        ref = solved[-1]

    solved = np.concatenate(solved, axis=1)

    flow_bbox = bounding_box.BoundingBox(start=(0, 0, 0), size=(flow.shape[-1], flow.shape[-2], 1))

    inv_map = map_utils.invert_map(solved, flow_bbox, flow_bbox, stride)

    return inv_map, flow_bbox, np.array(all_vmax)
