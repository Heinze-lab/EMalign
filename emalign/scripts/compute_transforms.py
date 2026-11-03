import os
import argparse
import logging
import numpy as np
import sys

from inspect import signature

from emalign.align_z.config import load_and_validate_configs
from emalign.align_z.transform import (chain_transforms_path, compute_global_bbox, compute_transforms_stack,
                                       last_valid_slice, shift_transforms)
from emalign.io.progress import get_mongo_client, get_mongo_db, wipe_progress
from emalign.io.store import get_store_attributes, set_store_attributes

logger = logging.getLogger(__name__)


def compute_dataset_transforms(
        project_dir,
        num_workers=0,
        start_over=False,
        wipe_progress_stacks=None
        ):
    '''Compute pairwise affine transforms for a dataset. 

    The output creates one transform per slice that corresponds to a slice resampled to target_scale.
    Transforms are computed pairwise on pre-transformation images and later chained to obtain 
    final transforms, aligning slices throughout the whole dataset.
    '''

    config_dir = os.path.join(project_dir, 'config/z_config')
    if not os.path.exists(config_dir) or not os.listdir(config_dir):
        raise FileNotFoundError(f'Configuration directory does not exist or is empty: {config_dir}\nDid you run prep_config_z?')
    
    # Validate config directory
    logger.info(f'Loading configuration from: {config_dir}')
    align_plan, dataset_configs = load_and_validate_configs(config_dir)

    # Extract key info from align plan
    root_stack = align_plan['root_stack']
    paths = align_plan['paths']
    reverse_order = align_plan['reverse_order']
    project_name = align_plan['project_name']
    destination_path = align_plan['destination_path']

    logger.info(f'Project: {project_name}')
    logger.info(f'Root stack: {root_stack}')
    logger.info(f'Number of alignment paths: {len(paths)}\n')
    logger.info('ToDo list:')
    mentioned = []
    for path in paths:
        for dataset_name in path:
            if dataset_name in mentioned:
                continue
            logger.info(f'    - {dataset_name}')
            mentioned.append(dataset_name)
    print()

    if any(reverse_order):
        raise NotImplementedError('reverse_order not yet implemented.')

    # Handle start_over
    if start_over:
        try:
            input('WARNING: All transform progress will be wiped and all datasets will be processed.\n'
                  'Press ENTER to continue or CTRL+C to abort\n')
        except KeyboardInterrupt:
            logger.info('\nAborted by user')
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
            wipe_progress(db, dataset_name, step_name='transform_z') # database progress
            logger.info(f'Wiped transform progress for {dataset_name}')

        # Chained transforms depend on the transforms of the previous datasets in the path,
        # and the final shift on the bbox of all datasets, so they are all recomputed
        zarr_path = os.path.dirname(os.path.abspath(destination_path))
        for dataset_name in dataset_configs:
            trsf_chained_path = os.path.join(zarr_path, 'z_intermediate', 'transform_chained', dataset_name)
            attrs = get_store_attributes(trsf_chained_path)
            if attrs is not None:
                attrs |= {'complete': False, 'final': False}
                set_store_attributes(trsf_chained_path, attrs)

    # First pass: compute raw in-stack transforms
    logger.info('Starting transforms computation...')
    logger.info(f'Number of cores used for computing transforms: {num_workers}\n')
    zarr_path = os.path.dirname(os.path.abspath(destination_path))
    carry_slices = []  # Last valid slice of each stack, with align_to_reference
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
            if not config.get('align_to_reference', False):
                config['reference_path'] = prev_dataset
            else:
                # The closest slice before this stack may stand for its previous slice
                # (compute_transforms_stack checks the distance)
                before = [c for c in carry_slices if c['global_z'] < config['z_offset']]
                config['carry_slice'] = max(before, key=lambda c: c['global_z']) if before else None

            # Start computing transforms
            logger.info(dataset_name)
            params = signature(compute_transforms_stack).parameters
            relevant_args = {k: v for k, v in config.items() if k in params}
            success = compute_transforms_stack(**relevant_args)

            if success and config.get('align_to_reference', False):
                z_min = config.get('local_z_min') or 0
                last = last_valid_slice(zarr_path, dataset_name, z_min, config['local_z_max'])
                if last is not None:
                    carry_slices.append({'dataset_name': dataset_name,
                                         'dataset_path': config['dataset_path'],
                                         'z': int(last[0]),
                                         'global_z': int(last[0] + config['z_offset'] - z_min),
                                         'depth': int(last[1])})

            # If the previous dataset failed to be transformed, 
            # we would have a broken link. Instead, let's use 
            # the last dataset that worked
            if success:
                prev_dataset = config['dataset_path']
    
    logger.info('Chaining transforms...')
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
    logger.info('Placing all transforms onto the final canvas...')
    if bbox is None:
        raise RuntimeError('No stack has a valid transform, nothing can be placed in the output.')
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

    logger.info('Done!')


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
                        help=f'Number of threads to use for rendering. Default: 0 (all)')
    parser.add_argument('--start-over',
                        dest='start_over',
                        default=False,
                        action='store_true',
                        help='Wipe all transform progress and restart')
    parser.add_argument('--wipe-progress',
                        dest='wipe_progress_stacks',
                        type=str,
                        nargs='+',
                        default=[],
                        help='Wipe transform progress for one or more specific stack(s) before starting')
    return parser


if __name__ == '__main__':

    from emalign.utils.logging_utils import setup_logging
    setup_logging(logging.INFO)

    parser = argparse.ArgumentParser(
        description='Compute affine transforms using pre-generated configuration files.\n'
                    'Configuration files should be created using prep_config_z first.',
        formatter_class=argparse.RawDescriptionHelpFormatter
        )
    args = add_parser_arguments(parser).parse_args()
    compute_dataset_transforms(
        project_dir=args.project_dir,
        num_workers=args.num_workers,
        start_over=args.start_over,
        wipe_progress_stacks=args.wipe_progress_stacks
    )
