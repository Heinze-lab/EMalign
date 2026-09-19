
import os
import argparse
import logging
import numpy as np
import sys

from inspect import signature

from emalign.align_z.config import load_align_plan, load_dataset_configs, validate_config_directory
from emalign.align_z.transform import chain_transforms_path, compute_global_bbox, compute_transforms_stack, shift_transforms
from emalign.io.store import set_store_attributes, get_store_attributes
from emalign.io.progress import get_mongo_client, get_mongo_db, wipe_progress


logging.basicConfig(level=logging.INFO)


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


def compute_dataset_transforms(
        project_dir: str,
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

    logging.info(f'Project: {project_name}')
    logging.info(f'Root stack: {root_stack}')
    logging.info(f'Number of alignment paths: {len(paths)}')

    if any(reverse_order):
        raise NotImplementedError('reverse_order not yet implemented.')

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
        steps = ['transform_z', 'flow_z', 'mesh_relax_z', 'render_z']
        for dataset_name in dataset_configs:
            for step in steps:
                wipe_progress(db, dataset_name, step_name=step) # database progress
            attrs = get_store_attributes(dataset_configs[dataset_name]['dataset_path'])
            attrs['z_aligned'] = False # attribute flag when data has been processed
            set_store_attributes(dataset_configs[dataset_name]['dataset_path'], attrs)
            logging.info(f'Wiped progress for {dataset_name}')

    # First pass: compute raw in-stack transforms
    logging.info('Starting transforms computation...')
    logging.info(f'Number of cores used for computing transforms: {num_workers}')
    for i, path in enumerate(paths):
        prev_dataset = None
        for dataset_name in path:
            if dataset_name not in dataset_configs:
                raise RuntimeError(f'No configuration found for dataset: {dataset_name}')

            config = dataset_configs[dataset_name].copy()

            if dataset_name == path[0] and i == 0:
                assert dataset_name == root_stack, \
                    f'First dataset ({dataset_name}) of the path is not the root stack ({root_stack})'

            config['num_workers'] = num_workers
            config['wipe_progress_flag'] = any([dataset_name == s for s in wipe_progress_stacks])
            config['reference_dataset'] = prev_dataset

            # Start alignment
            params = signature(compute_transforms_stack).parameters
            relevant_args = {k: v for k, v in config.items() if k in params}
            compute_transforms_stack(**relevant_args)
            
            prev_dataset = config['dataset_path']
    
    # Second pass: chain existing transforms throughout the whole dataset
    bbox = None
    for path in paths:
        bbox = chain_transforms_path(
            path, 
            dataset_configs,
            anchor_inv=None,
            bbox=bbox
        )

    # Third pass: compute bbox and shift everything
    output_shape, (shift_y, shift_x) = compute_global_bbox(bbox)

    destination_path = os.path.abspath(destination_path)
    zarr_path = os.path.dirname(destination_path)
    dataset_names = set(np.concatenate(paths).tolist())
    for dataset_name in dataset_names:
        trsf_chained_path = os.path.join(zarr_path, 'z_intermediate', 'transform_chained', dataset_name)
        shift_transforms(
            trsf_chained_path,
            (shift_y, shift_x),
            output_shape
        )

    logging.info('Done!')

    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Compute affine transforms using pre-generated configuration files.\n'
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
                        default=0,
                        help=f'Number of threads to use for rendering. Default: 0 (all)')
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

    compute_dataset_transforms(
        project_dir=args.project_dir,
        num_workers=args.num_workers,
        start_over=args.start_over,
        wipe_progress_stacks=args.wipe_progress_stacks
    )
