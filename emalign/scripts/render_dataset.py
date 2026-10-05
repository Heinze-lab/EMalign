
import argparse
import cv2
import logging
import numpy as np
import os
import sys 
import tensorstore as ts
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor
from connectomics.common import bounding_box
from inspect import signature 
from sofima.warp import ndimage_warp

from emalign.align_dataset_z import load_and_validate_configs
from emalign.align_z.warp_hotfix import warp_affine_tiled
from emalign.align_z.transform import to_3x3, shift_matrix 
from emalign.io.progress import get_mongo_client, get_mongo_db, log_progress, wipe_progress, check_progress 
from emalign.io.process.mask import mask_to_bbox
from emalign.io.store import open_store, get_store_attributes, set_store_attributes, write_data
from emalign.io.utils import get_dataset_name, get_zarr_root, load_slice
from emalign.utils.logging_utils import setup_logging


SHRT_MAX = 32767
CHUNK_SIZE = [1, 1024, 1024]  # For store creation
DOWNSAMPLE_SCALE = 10  # For creation of the downsampled inspection store
Z_RESOLUTION = 50 # nm

logging.basicConfig(level=logging.INFO)
setup_logging(logging.INFO)

def render_slice(
        image,
        M,
        output_shape,
        inv_map,
        stride,
        work_size,
        overlap,
        mask=None,
        num_workers=0
):
    '''Warp a slice into the flow canvas with M, then apply the inverse map of the mesh.'''

    if num_workers > 0:
        # Set the number of workers used by cv2 to match keypoints
        cv2.setNumThreads(max(1, num_workers - 1))
    
    # Transform image
    if max(*image.shape, *output_shape) > SHRT_MAX:
        warp_fun = warp_affine_tiled
    else:
        warp_fun = cv2.warpAffine
    image = warp_fun(image, M, output_shape[::-1])
    data_bbox = bounding_box.BoundingBox(
        start=(0, 0, 0), 
        size=(image.shape[-1], image.shape[-2], 1)
        )
    
    # Warp image in parallel. warp_subvolume uses one thread per image so we use ndimage_wrap instead
    aligned = ndimage_warp(
                    image, 
                    inv_map, 
                    stride=(stride, stride),
                    work_size=(work_size, work_size),
                    overlap=(overlap,overlap),
                    image_box=data_bbox,
                    parallelism=max(1, num_workers)
                )
    if mask is not None:
        mask = warp_fun(mask.astype(np.uint8), M, output_shape[::-1]).astype(bool)
        aligned_mask = ndimage_warp(
                        mask, 
                        inv_map, 
                        stride=(stride, stride),
                        work_size=(work_size, work_size),
                        overlap=(overlap,overlap),
                        image_box=data_bbox,
                        parallelism=max(1, num_workers)
                    )
        
        # Mask gets full of holes because of warping so we need to fill them
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        aligned_mask = cv2.morphologyEx(aligned_mask.astype(np.uint8),cv2.MORPH_CLOSE,kernel).astype(bool)
        return aligned, aligned_mask
    return aligned, None


def render_stack(
        dataset,
        destination,
        work_size,
        overlap,
        dataset_mask=None,
        save_downsampled=10,
        target_scale=1,
        z_offset=0,
        local_z_min=None,
        local_z_max=None,
        ignore_slices=(),
        overwrite=False,
        mongodb_config_filepath=None,
        num_workers=0
    ):
    
    # Get destination
    project_name = get_dataset_name(destination)
    destination_path = os.path.abspath(destination.kvstore.path)
    destination_mask = open_store(destination_path + '_mask', mode='r+', dtype=ts.bool)
    ds_destination = None
    if save_downsampled > 1:
        # Save a downsampled version for easy inspection
        ds_output_path = destination_path.rsplit('/', maxsplit=1)[0]
        ds_output_path = os.path.join(ds_output_path, f'{save_downsampled}x_' + project_name)
        ds_destination = open_store(ds_output_path, mode='r+', dtype=ts.uint8)

    db = get_mongo_db(get_mongo_client(mongodb_config_filepath), project_name)
    step_name = 'render_z'

    z_min = local_z_min if local_z_min is not None else dataset.domain.inclusive_min[0]
    z_max = local_z_max if local_z_max is not None else dataset.domain.exclusive_max[0]

    # Prepare transforms
    dataset_name = get_dataset_name(dataset)
    zarr_path = get_zarr_root(destination)
    trsf_path = os.path.join(zarr_path, 'z_intermediate', 'transform_chained', dataset_name)
    mesh_path = os.path.join(zarr_path, 'z_intermediate', 'inverse_map', dataset_name)

    dataset_trsf = open_store(trsf_path, mode='r')
    attrs = get_store_attributes(dataset_trsf)
    if not attrs or not attrs.get('final', False):
        raise RuntimeError(f'{dataset_name}: Final transforms not found, run compute_transforms first.')
    dataset_mesh = open_store(mesh_path, mode='r')
    stride = get_store_attributes(dataset_mesh)['stride']

    # Slices without a valid transform are skipped, as in the flow
    raw_trsf_path = os.path.join(zarr_path, 'z_intermediate', 'transform', dataset_name)
    raw_transforms = open_store(raw_trsf_path, mode='r').read().result()
    valid = ~np.isnan(raw_transforms).any(axis=(1, 2))
    valid[[z for z in ignore_slices if z < len(valid)]] = False

    # The mesh exists on the flow canvas (scale 1), which starts at -shift in the output.
    # We render in that canvas and write it at -shift, rounded so the image is exactly placed.
    mesh_attrs = get_store_attributes(dataset_mesh)
    canvas_shape = mesh_attrs['canvas_shape']
    off_y, off_x = int(round(-mesh_attrs['shift_y'])), int(round(-mesh_attrs['shift_x']))
    to_canvas = shift_matrix(-off_x, -off_y)
    
    def _load_slice_kit(
            dataset, 
            dataset_mask, 
            dataset_trsf,
            dataset_mesh,
            z,
            scale
            ):
        
        M = dataset_trsf[z].read().result()[None].astype(np.float64)
        M = (to_canvas @ to_3x3(M)[0])[:2]
        kit = load_slice(dataset, dataset_mask, z, scale)
        kit['M'] = M
        kit['inv_map'] = dataset_mesh[:, z].read().result()
        return kit
    
    # ---------- Check progress ----------
    # Resume from the first slice not rendered yet, as for the transforms and flow
    first_z = None
    for z in range(z_min, z_max):
        if not check_progress(db, dataset_name, step_name, z):
            first_z = z
            break
    if first_z is None:
        # Everything was processed already
        logging.info(f'{dataset_name}: All slices were already rendered.')
        return

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
                _load_slice_kit, 
                dataset, dataset_mask, dataset_trsf, dataset_mesh, next_read, target_scale)
            next_read += 1

    prefetch_data(first_z + window)  
    n_gap = 0 
    for z in tqdm(
        range(first_z, z_max),  
        total=z_max - first_z,
        position=0,
        desc=f'{dataset_name}: Rendering aligned slices',
        dynamic_ncols=True,
        leave=True
        ):
        # Prefetch more data to consume later
        prefetch_data(z + window)

        mov_slice = read_futures.pop(z).result()
        global_z = z + z_offset - z_min

        if mov_slice['empty']: 
            # Empty slice. Skip this and compute transform across the gap
            n_gap += 1
            log_progress(db, dataset_name, step_name, global_z, z,
                         {'empty_slice': bool(mov_slice['empty']), 'skipped': True})
            continue
        if not valid[z]:
            # Invalid or ignored slice, nothing to render
            log_progress(db, dataset_name, step_name, global_z, z,
                         {'valid': False, 'skipped': True})
            continue

        aligned, aligned_mask = render_slice(
                image=mov_slice['img'],
                M=mov_slice['M'],
                output_shape=canvas_shape,
                inv_map=mov_slice['inv_map'],
                stride=stride,
                work_size=work_size,
                overlap=overlap,
                mask=mov_slice['mask'],
                num_workers=num_workers
        )

        if overwrite:
            # There may be data written to this slice so let's make sure it is overwritten
            y1 = x1 = 0
            y2, x2 = aligned.shape
            write_mask = None # This means we write everything even black space
        else:
            # Write only within bounding box
            y1, y2, x1, x2 = mask_to_bbox(aligned_mask)
            write_mask = aligned_mask[y1:y2, x1:x2]
        
        # To full resolution destination
        write_data(
            destination,
            aligned[y1:y2, x1:x2],              # Only write in the bounding box where the data is
            global_z,                           # z_offset relates to original minimum
            np.array([x1 + off_x, y1 + off_y]), # Canvas to output frame
            preserve_mask=write_mask,           # Mask where to write the data
            resolve=True
            )
        # To destination mask
        write_data(
            destination_mask,
            aligned_mask[y1:y2, x1:x2], 
            global_z, 
            np.array([x1 + off_x, y1 + off_y]),
            preserve_mask=write_mask, 
            resolve=True
            )
        # To downsampled destination
        if ds_destination is not None:
            write_data(
                ds_destination,
                aligned[y1:y2, x1:x2], 
                global_z, 
                np.array([x1 + off_x, y1 + off_y]),
                preserve_mask=write_mask, 
                downsample_factor=1/save_downsampled,
                resolve=True
                )
        
        # Log progress
        metadata = {
            'empty_slice': False,
            'overwrite': overwrite,
            'bbox': [int(y1 + off_y), int(y2 + off_y), int(x1 + off_x), int(x2 + off_x)]
        }
        log_progress(db, dataset_name, step_name, global_z, z, metadata)


def render_dataset(
        project_dir,
        num_workers=0,
        start_over=False,
        wipe_progress_stacks=None
        ):

    config_dir = os.path.join(project_dir, 'config/z_config')
    if not os.path.exists(config_dir) or not os.listdir(config_dir):
        raise FileNotFoundError(f'Configuration directory does not exist or is empty: {config_dir}\nDid you run prep_config_z?')

    # Validate config directory
    logging.info(f'Loading configuration from: {config_dir}')
    align_plan, dataset_configs = load_and_validate_configs(config_dir)

    # Extract key info from align plan
    root_stack = align_plan['root_stack']
    paths = align_plan['paths']
    reverse_order = align_plan['reverse_order']
    project_name = align_plan['project_name']
    destination_path = align_plan['destination_path']
    save_downsampled = align_plan.get('save_downsampled', DOWNSAMPLE_SCALE)
    if any(reverse_order):
        raise NotImplementedError('reverse_order not yet implemented.')

    logging.info(f'Project: {project_name}')
    logging.info(f'Root stack: {root_stack}')
    logging.info(f'Number of alignment paths: {len(paths)}')
    logging.info('ToDo list:')
    mentioned = []
    for path in paths:
        for dataset_name in path:
            if dataset_name in mentioned:
                continue
            logging.info(f'    - {dataset_name}')
            mentioned.append(dataset_name)

    # Handle start_over
    if start_over:
        try:
            input('WARNING: All render progress will be wiped and all datasets will be processed.\n' 
                  'Press ENTER to continue or CTRL+C to abort\n')
        except KeyboardInterrupt:
            logging.info('\nAborted by user')
            sys.exit(0)
        wipe_progress_stacks = list(dataset_configs)

    # Wipe progress for the requested datasets
    wipe_progress_stacks = [s for s in (wipe_progress_stacks or []) if s]
    if wipe_progress_stacks:
        first_config = next(iter(dataset_configs.values()))
        mongodb_config_filepath = first_config.get('mongodb_config_filepath')

        client = get_mongo_client(mongodb_config_filepath)
        db = get_mongo_db(client, project_name)
        for dataset_name in wipe_progress_stacks:
            if dataset_name not in dataset_configs:
                raise RuntimeError(f'No configuration found for dataset: {dataset_name}')
            wipe_progress(db, dataset_name, step_name='render_z') # database progress 
            logging.info(f'Wiped render progress for {dataset_name}')

    # Create or open destination
    zarr_path = get_zarr_root(align_plan['destination_path'])
    any_dataset = next(iter(dataset_configs))
    trsf_attrs = get_store_attributes(open_store(
        os.path.join(zarr_path, 'z_intermediate', 'transform_chained', any_dataset), mode='r'))
    if not trsf_attrs or not trsf_attrs.get('final', False):
        raise RuntimeError('Final transforms not found, run compute_transforms first.')
    max_z = max(c['z_offset'] + c['local_z_max'] - c['local_z_min'] for c in dataset_configs.values())
    dest_shape = [int(max_z), *map(int, trsf_attrs['output_shape'])]

    destination = open_store(
        destination_path, mode='a', dtype=ts.uint8,
        shape=dest_shape, chunks=CHUNK_SIZE
    )
    destination_mask = open_store(
        destination_path + '_mask', mode='a', dtype=ts.bool,
        shape=dest_shape, chunks=CHUNK_SIZE
    )
    if save_downsampled > 1:
        # Save a downsampled version for easy inspection
        ds_output_path = destination_path.rsplit('/', maxsplit=1)[0]
        ds_output_path = os.path.join(ds_output_path, f'{save_downsampled}x_' + project_name)
        ds_destination = open_store(
            ds_output_path, mode='a', dtype=ts.uint8,
            shape=[dest_shape[0], dest_shape[1] // save_downsampled, dest_shape[2] // save_downsampled], 
            chunks=CHUNK_SIZE
            )

    # Resolution and voxel_size are the same, just there for compatibility
    # with different versions of daisy or funlib.persistence
    yx_res = align_plan['yx_target_resolution']
    resolution = [Z_RESOLUTION, yx_res, yx_res]
    attrs = {'voxel_offset': [0, 0, 0], 'offset': [0, 0, 0],
             'resolution': resolution, 'voxel_size': resolution}
    for store in (destination, destination_mask):
        set_store_attributes(store, (get_store_attributes(store) or {}) | attrs)
    if save_downsampled > 1:
        ds_resolution = [Z_RESOLUTION, yx_res * save_downsampled, yx_res * save_downsampled]
        set_store_attributes(ds_destination, (get_store_attributes(ds_destination) or {}) | attrs |
                             {'resolution': ds_resolution, 'voxel_size': ds_resolution})

    logging.info('Starting rendering...')
    logging.info(f'Number of cores used for rendering: {num_workers}')
    params = signature(render_stack).parameters
    rendered = set()
    for path in paths:
        for dataset_name in path:
            if dataset_name not in dataset_configs:
                raise RuntimeError(f'No configuration found for dataset: {dataset_name}')
            if dataset_name in rendered:
                continue

            config = dataset_configs[dataset_name].copy()
            dataset_path = os.path.abspath(config['dataset_path'])
            config['dataset'] = open_store(dataset_path, mode='r', dtype=ts.uint8)
            config['dataset_mask'] = open_store(dataset_path + '_mask', mode='r', dtype=ts.bool, allow_missing=True)
            config['destination'] = destination
            config['work_size'] = config['warp_config']['work_size']
            config['overlap'] = config['warp_config']['overlap']
            config['num_workers'] = num_workers

            # Same resolution as for the transforms and flow
            res = get_store_attributes(config['dataset'])['resolution'][-1]
            yx_target_resolution = config.get('yx_target_resolution')
            config['target_scale'] = 1 if yx_target_resolution is None else res / yx_target_resolution

            relevant_args = {k: v for k, v in config.items() if k in params}
            render_stack(**relevant_args)
            rendered.add(dataset_name)

    logging.info('All the data was rendered and written!')
    logging.info(f'Destination: {destination_path}')


def add_parser_arguments(parser):
    # Required arguments
    parser.add_argument('-p', '--project-dir',
                        metavar='PROJECT_DIR',
                        dest='project_dir',
                        required=True,
                        type=str,
                        help='Project directory containing the configurations created with prep_config_z.')

    # Optional arguments
    parser.add_argument('-c', '--cores',
                        metavar='CORES',
                        dest='num_workers',
                        type=int,
                        default=0,
                        help=f'Number of threads to use for warping slices. Default: 0 (all)')
    parser.add_argument('--start-over',
                        dest='start_over',
                        default=False,
                        action='store_true',
                        help='Wipe all render progress and restart')
    parser.add_argument('--wipe-progress',
                        dest='wipe_progress_stacks',
                        type=str,
                        nargs='+',
                        default=[],
                        help='Wipe render progress for one or more specific stack(s) before starting')
    return parser


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Render a dataset based on pre-computed non-linear transformation.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )


    args = add_parser_arguments(parser).parse_args()

    render_dataset(
        project_dir=args.project_dir,
        num_workers=args.num_workers,
        start_over=args.start_over,
        wipe_progress_stacks=args.wipe_progress_stacks
    )
