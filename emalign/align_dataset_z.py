'''Execute Z alignment using pre-generated configuration files.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python -m emalign.align_dataset_z \\
        -cfg /path/to/z_config_dir/ \\
        -cfg-z /path/to/z_config.json \\
        -c 4 \\
        -ds 10 \\
        --start-over \\
        --wipe-progress stack_name
'''

import os
# To prevent running out of memory because of preallocation
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Influences performance
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import argparse
import json
import logging
import numpy as np
import sys

from inspect import signature
from typing import List, Optional

from emalign.align_z.config import load_align_plan, load_dataset_configs, validate_config_directory
from emalign.scripts.align_stack_z import align_stack_z
from emalign.io.store import open_store
from emalign.io.progress import get_mongo_client, get_mongo_db, wipe_progress


logging.basicConfig(level=logging.INFO)
logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('jax._src.xla_bridge').setLevel(logging.WARNING)

# Constants
CHUNK_SIZE = [1, 1024, 1024]  # For store creation
NUM_WORKERS = 1
DOWNSAMPLE_SCALE = 10  # For creation of the downsampled inspection store
Z_RESOLUTION = 50 # nm


def load_and_validate_configs(config_dir):
    '''Load and validate all configuration from a prepared config directory.

    Args:
        config_dir: Path to directory created by prep_config_z

    Returns:
        tuple: (align_plan, dataset_configs)
            - align_plan: Contents of 00_align_plan.json
            - dataset_configs: Dict of dataset_name -> config

    Raises:
        FileNotFoundError: If required files missing
        ValueError: If configs are invalid or inconsistent
    '''
    # Validate first
    is_valid, errors, warnings_list = validate_config_directory(config_dir)

    for warning in warnings_list:
        logging.warning(warning)

    if not is_valid:
        for error in errors:
            logging.error(error)
        raise ValueError(f'Invalid configuration directory: {config_dir}')

    # Load configs
    align_plan = load_align_plan(config_dir)
    dataset_configs = load_dataset_configs(config_dir)

    return align_plan, dataset_configs


def initialize_destination_stores(destination_path, align_plan, save_downsampled,
                                   project_name, start_over):
    '''Create or open destination zarr stores.

    Args:
        destination_path: Path to main output zarr
        align_plan: Alignment plan dictionary
        save_downsampled: Downsampling factor
        project_name: Name of the project
        start_over: Whether to recreate stores

    Returns:
        tuple: (destination, destination_mask, ds_destination, ds_project_output_path)
    '''
    import tensorstore as ts

    ds_project_output_path = os.path.join(
        os.path.dirname(destination_path),
        f'{int(save_downsampled)}x_{project_name}'
    )
    create_new = not os.path.exists(destination_path) or start_over

    if create_new:
        z_off = 0 # TODO: deal with this
        logging.info(f'Creating project dataset at: \n    {destination_path}\n')
        if start_over:
            logging.info('Previous dataset will be overwritten')
            open_mode = 'w'
        else:
            open_mode = 'a'

        # Estimate shape from align plan
        ds_bounds = align_plan['dataset_local_bounds']
        root_offset = np.array(align_plan['root_offset'])
        pad_offset = np.array(align_plan['pad_offset'])
        bbox = align_plan['ref_global_bbox']
        if bbox is not None:
            y_off = bbox[0]
            x_off = bbox[2]
        else:
            y_off = x_off = 0

        # Calculate total Z range
        max_z = 0
        for _, bounds in ds_bounds.items():
            # bounds is [local_z_min, local_z_max]
            max_z = max(max_z, bounds[1])

        # Estimate YX from root_offset + pad
        # This is a rough estimate; actual shape may need adjustment during alignment
        estimated_yx = root_offset + pad_offset + np.array([10000, 10000])

        dest_shape = [max_z, int(estimated_yx[0]), int(estimated_yx[1])]

        destination = open_store(
            destination_path, mode=open_mode, dtype=ts.uint8,
            shape=dest_shape, chunks=CHUNK_SIZE
        )
        destination_mask = open_store(
            destination_path + '_mask', mode=open_mode, dtype=ts.bool,
            shape=dest_shape, chunks=CHUNK_SIZE
        )

        logging.info(f'Creating downsampled project dataset ({save_downsampled}) at: \n    {ds_project_output_path}\n')
        dest_shape_ds = [
            dest_shape[0],
            int(dest_shape[1] // save_downsampled),
            int(dest_shape[2] // save_downsampled)
        ]

        ds_destination = open_store(
            ds_project_output_path, mode=open_mode, dtype=ts.uint8,
            shape=dest_shape_ds, chunks=CHUNK_SIZE
        )

        # Set some attributes that may be used by downstream pipelines
        # resolution and voxel_size are the same, just there for compatibility
        # with different versions of daisy or funlib.persistence
        attrs = {
                "voxel_offset": [z_off, y_off, x_off],
                "offset": [z_off, y_off, x_off],
                "resolution": [
                    Z_RESOLUTION,
                    align_plan['yx_target_resolution'],
                    align_plan['yx_target_resolution']
                ],
                "voxel_size": [
                    Z_RESOLUTION,
                    align_plan['yx_target_resolution'],
                    align_plan['yx_target_resolution']
                ]
                }
        set_store_attributes(destination, attrs)
        
        attrs = {
                "voxel_offset": [z_off, y_off//save_downsampled, x_off//save_downsampled],
                "offset": [z_off, y_off//save_downsampled, x_off//save_downsampled],
                "resolution": [
                    Z_RESOLUTION,
                    align_plan['yx_target_resolution']*save_downsampled,
                    align_plan['yx_target_resolution']*save_downsampled
                ],
                "voxel_size": [
                    Z_RESOLUTION,
                    align_plan['yx_target_resolution']*save_downsampled,
                    align_plan['yx_target_resolution']*save_downsampled
                ]
                }
        set_store_attributes(ds_destination, attrs)
    else:
        logging.info(f'Opening existing project dataset at: \n    {destination_path}\n')
        import tensorstore as ts
        destination = open_store(destination_path, mode='r+', dtype=ts.uint8)
        destination_mask = open_store(destination_path + '_mask', mode='r+', dtype=ts.bool)

        logging.info(f'Opening existing downsampled project dataset ({save_downsampled}) at: \n    {ds_project_output_path}\n')
        ds_destination = open_store(ds_project_output_path, mode='r+', dtype=ts.uint8)

    return destination, destination_mask, ds_destination, ds_project_output_path


def execute_alignment(paths, dataset_configs, root_stack, num_workers, wipe_progress_stacks):
    '''Execute the alignment for all datasets.

    Args:
        paths: List of alignment paths
        dataset_configs: Dictionary of dataset_name -> config
        root_stack: Name of the root stack
        num_workers: Number of worker threads
        wipe_progress_stacks: Optional stack name to wipe progress for
    '''
    for i, path in enumerate(paths):
        for dataset_name in path:
            if dataset_name not in dataset_configs:
                raise RuntimeError(f'No configuration found for dataset: {dataset_name}')

            config = dataset_configs[dataset_name].copy()

            if dataset_name == path[0] and i == 0:
                assert dataset_name == root_stack, \
                    f'First dataset ({dataset_name}) of the path is not the root stack ({root_stack})'

            config['num_workers'] = num_workers
            config['wipe_progress_flag'] = any([dataset_name == s for s in wipe_progress_stacks])

            # Start alignment
            try:
                params = signature(align_stack_z).parameters
                relevant_args = {k: v for k, v in config.items() if k in params}
                align_stack_z(**relevant_args)
            except KeyError as e:
                raise RuntimeError(f'Missing required parameter for {dataset_name}: {e}')
            except ValueError as e:
                raise RuntimeError(f'Invalid parameter value for {dataset_name}: {e}')
            except IOError as e:
                raise RuntimeError(f'IO error processing {dataset_name}: {e}')
            except Exception as e:
                logging.exception(f'Unexpected error processing {dataset_name}')
                raise RuntimeError(f'Error processing {dataset_name}: {e}')


def align_dataset_z(project_dir: str,
                    num_workers: int = NUM_WORKERS,
                    save_downsampled: float = DOWNSAMPLE_SCALE,
                    start_over: bool = False,
                    wipe_progress_stacks: Optional[list[str]] = None) -> None:
    '''Execute Z alignment using pre-generated configuration files.

    Args:
        project_dir: Directory containing the project: config directory, and output zarr
        num_workers: Number of worker threads
        save_downsampled: Downsampling factor for inspection store
        start_over: Wipe all progress and restart
        wipe_progress_stacks: List of specific stacks to wipe progress for
    '''
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
    destination_path = align_plan['destination_path']
    project_name = align_plan['project_name']

    logging.info(f'Project: {project_name}')
    logging.info(f'Root stack: {root_stack}')
    logging.info(f'Number of alignment paths: {len(paths)}')
    logging.info(f'Destination: {destination_path}')

    # Handle start_over
    if start_over:
        try:
            input('WARNING: All progress will be wiped and all datasets will be processed.\n'
                  'Press ENTER to continue or CTRL+C to abort\n')
        except KeyboardInterrupt:
            logging.info('\nAborted by user')
            sys.exit(0)

        # Wipe progress for all datasets
        first_config = next(iter(dataset_configs.values()))
        mongodb_config_filepath = first_config.get('mongodb_config_filepath')
    
        client = get_mongo_client(mongodb_config_filepath)
        db = get_mongo_db(client, project_name)
        wipe_progress_stacks = []
        steps = ['flow_z', 'mesh_relax_z', 'render_z']
        for dataset_name in dataset_configs:
            for step in steps:
                wipe_progress(db, dataset_name, step_name=step) # database progress
            attrs = get_store_attributes(dataset_configs[dataset_name]['dataset_path'])
            attrs['z_aligned'] = False # attribute flag when data has been processed
            set_store_attributes(dataset_configs[dataset_name]['dataset_path'], attrs)
            logging.info(f'Wiped progress for {dataset_name}')

    # Initialize destination stores
    destination, destination_mask, ds_destination, ds_project_output_path = \
        initialize_destination_stores(
            destination_path, align_plan, save_downsampled, project_name, start_over
        )

    # Execute alignment
    logging.info('Starting Z alignment...')
    execute_alignment(paths, dataset_configs, root_stack, num_workers, wipe_progress_stacks)

    logging.info('Done!')
    logging.info(f'Output: {destination_path}')
    logging.info(f'Downsampled: {ds_project_output_path}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Execute Z alignment using pre-generated configuration files.\n'
                    'Configuration files should be created using prep_config_z first.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

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
                        default=NUM_WORKERS,
                        help=f'Number of threads to use. Default: {NUM_WORKERS}')
    parser.add_argument('-ds', '--downsample-scale',
                        metavar='SCALE',
                        dest='save_downsampled',
                        type=float,
                        default=DOWNSAMPLE_SCALE,
                        help=f'Downsampling factor for inspection store. Default: {DOWNSAMPLE_SCALE}')
    parser.add_argument('--start-over',
                        dest='start_over',
                        default=False,
                        action='store_true',
                        help='Wipe all progress and restart')
    parser.add_argument('--wipe-progress',
                        dest='wipe_progress_stacks',
                        type=str,
                        nargs='+',
                        default=[''],
                        help='Wipe progress for one or more specific stack(s) before starting')

    args = parser.parse_args()

    # Check GPU
    try:
        GPU_ids = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        logging.error('No GPUs specified. Set CUDA_VISIBLE_DEVICES environment variable.')
        logging.error('Example: CUDA_VISIBLE_DEVICES=0,1 python -m emalign.align_dataset_z ...')
        sys.exit(1)
    logging.info(f'Using GPU IDs: {GPU_ids}')

    align_dataset_z(
        project_dir=args.project_dir,
        num_workers=args.num_workers,
        save_downsampled=args.save_downsampled,
        start_over=args.start_over,
        wipe_progress_stacks=args.wipe_progress_stacks
    )
