'''Prepare configuration files for Z alignment.

Usage:
    python -m emalign.prep_config_z \\
        -p /path/to/project_dir \\
        -cfg-z /path/to/z_config.json \\
        -c 4 \\
        --force-overwrite
'''

import argparse
import json
import logging
import numpy as np
import os
import sys

from glob import glob
from typing import List, Optional

from emalign.align_z.config import add_config_metadata, validate_config_directory, CONFIG_VERSION
from emalign.align_z.utils import compute_alignment_path, determine_initial_offset, determine_initial_offset_ref, get_ordered_datasets
from emalign.io.store import get_store_attributes

logging.basicConfig(level=logging.INFO)
logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('jax._src.xla_bridge').setLevel(logging.WARNING)

# Constants
PAD_OFFSET = np.array([1000, 1000])  # Offset to add to origin for drift correction


def load_configs_from_files(config_paths, exclude):
    '''Load configuration from XY main config files.

    Args:
        config_paths: List of paths to main config files
        exclude: List of patterns to exclude from datasets

    Returns:
        tuple: (datasets, z_offsets, yx_target_resolution,
                project_name, mongodb_config_filepath, output_path)
    '''
    try:
        with open(config_paths[0], 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f'Invalid JSON in config file {config_paths[0]}: {e}')
    except FileNotFoundError:
        raise FileNotFoundError(f'Config file not found: {config_paths[0]}')

    project_name = config.get('project_name')
    if not project_name:
        if 'output_path' not in config:
            raise KeyError(f'Config file {config_paths[0]} missing both "project_name" and "output_path"')
        project_name = os.path.basename(config['output_path']).rstrip('.zarr')
    mongodb_config_filepath = config.get('mongodb_config_filepath')

    if 'output_path' not in config:
        raise KeyError(f'Config file {config_paths[0]} missing required field "output_path"')
    output_path = config['output_path']

    if 'resolution' not in config and 'yx_target_resolution' not in config:
        raise KeyError(f'Config file {config_paths[0]} missing both "resolution" and "yx_target_resolution"')
    yx_target_resolution = config['resolution'][0] if 'resolution' in config else config['yx_target_resolution'][0]

    # Get list of datasets and offsets
    try:
        datasets, z_offsets = get_ordered_datasets(config_paths, exclude=exclude)
    except Exception as e:
        raise RuntimeError(f'Failed to load datasets from config files: {e}')

    return (datasets, z_offsets, yx_target_resolution,
            project_name, mongodb_config_filepath, output_path)


def create_alignment_configs(datasets, z_offsets, output_configs_dir, config_z, reference_path, 
                             reference_offset, destination_path, project_name, mongodb_config_filepath,
                             yx_target_resolution, save_downsampled):
    '''Create alignment configuration files for all datasets.

    Args:
        datasets: List of tensorstore datasets
        z_offsets: Array of z offsets for each dataset
        output_configs_dir: Directory to store config files
        config_z: Z alignment configuration dictionary
        destination_path: Path to output zarr
        project_name: Name of the project
        mongodb_config_filepath: Path to MongoDB config
        yx_target_resolution: Target resolution in YX
        save_downsampled: Downsampling factor

    Returns:
        tuple: (root_stack, paths, reverse_order, root_offset)
    '''
    logging.info('Creating Z align configuration files...')
    logging.info(f'Configuration files will be stored at: \n    {output_configs_dir}\n')

    if reference_path is None:
        # There is no reference dataset, compute the paths between datasets. 
        # Some stacks may be disconnected in some parts of the dataset
        logging.info('Computing alignment path...')
        root_stack, paths, reverse_order, ds_bounds = compute_alignment_path(
            datasets, z_offsets, target_resolution=yx_target_resolution)
        
        # Determine where to start to ensure that everything fits within the canvas
        logging.info('Computing padding...')
        root_offset = determine_initial_offset(datasets, paths)
        pad_offset = PAD_OFFSET.copy()  # pad offsets to correct for any drift
        root_offset += pad_offset
        ref_bboxes = None
    else:
        logging.info('Computing alignment path...')
        root_stack, paths, reverse_order, ds_bounds = compute_alignment_path(
            datasets, z_offsets, target_resolution=yx_target_resolution)
        
        # There is a reference dataset so we need to figure out the global offset relative to it
        logging.info('Computing padding...')
        root_offset, ref_bboxes = determine_initial_offset_ref(datasets, z_offsets, reference_path, reference_offset, yx_target_resolution)
        # Each dataset aligns to the reference independently — no chaining needed.
        # dataset_names = [os.path.basename(os.path.abspath(ds.kvstore.path)) for ds in datasets]
        # root_stack = dataset_names[0]
        # paths = [dataset_names]
        # reverse_order = [False]
        # ds_bounds = {name: (0, ds.shape[0])
        #              for name, ds in zip(dataset_names, datasets)}
        # root_offset is the (y, x) canvas origin; PAD_OFFSET is added as safety margin.
        # Since datasets align to the reference (not to each other), origin is always 0.
        pad_offset = PAD_OFFSET.copy()
        root_offset += pad_offset

    if ref_bboxes is not None:
        # Largest bounding box for reference data
        ref_global_bbox = [np.min(ref_bboxes, axis=0)[0], np.max(ref_bboxes, axis=0)[1], # y1, y2
                           np.min(ref_bboxes, axis=0)[2], np.max(ref_bboxes, axis=0)[3]] # x1, x2
        ref_global_bbox = list(map(int, ref_global_bbox))
        ref_global_bbox_start = np.array([ref_global_bbox[2], ref_global_bbox[0]]) # xy
    else:
        ref_global_bbox = None

    # Write alignment plan
    align_plan = {
        'root_stack': root_stack,
        'paths': paths,
        'reverse_order': reverse_order,
        'root_offset': root_offset.tolist(),
        'pad_offset': pad_offset.tolist(),
        'yx_target_resolution': yx_target_resolution,
        'dataset_local_bounds': ds_bounds,
        'reference_path': reference_path,
        'reference_offset': reference_offset,
        'ref_global_bbox': ref_global_bbox,
        'destination_path': destination_path,
        'project_name': project_name
    }
    align_plan = add_config_metadata(align_plan)

    os.makedirs(output_configs_dir, exist_ok=True)
    with open(os.path.join(output_configs_dir, '00_align_plan.json'), 'w') as f:
        json.dump(align_plan, f, indent=2)

    # Write configs for each dataset
    done = []
    for i, (path, order) in enumerate(zip(paths, reverse_order)):
        for dataset_name in path:
            if dataset_name in done:
                continue
            idx = [os.path.basename(os.path.abspath(d.kvstore.path)) == dataset_name for d in datasets].index(True)
            dataset = datasets[idx]
            z_offset = int(z_offsets[idx, 0]) + ds_bounds[dataset_name][0]
            config_path = os.path.join(output_configs_dir, f'z_{dataset_name}.json')

            if reference_path is not None:
                # All datasets align to the reference, there is no external first slice
                first_slice = None

                # The xy_offset is relative to the global offset to a reference
                bbox_ref = ref_bboxes[idx]
                bbox_ref_start = np.array([bbox_ref[2], bbox_ref[0]]) # xy
                xy_offset = np.abs(ref_global_bbox_start - bbox_ref_start).astype(int).tolist()
            elif dataset_name == path[0] and i == 0:
                # Very first dataset to align without reference should be root
                assert dataset_name == root_stack, f'First dataset ({dataset_name}) of the path is not the root stack ({root_stack})'
                first_slice = None
                xy_offset = list(map(int, root_offset))
                bbox_ref = None
            else:
                first_slice = z_offset - 1  # First slice is last slice from previous dataset
                xy_offset = [0, 0]
                bbox_ref = None

            config = {
                'destination_path': destination_path,
                'dataset_path': os.path.abspath(dataset.kvstore.path),
                'dataset_name': dataset_name,
                'reference_path': reference_path,
                'reference_offset': reference_offset,
                'alignment_path': path,
                'reverse_order': order,
                'project_name': project_name,
                'mongodb_config_filepath': mongodb_config_filepath,
                'bbox_ref': bbox_ref,
                'z_offset': z_offset,
                'xy_offset': xy_offset,
                'local_z_min': ds_bounds[dataset_name][0],
                'local_z_max': ds_bounds[dataset_name][1],
                'scale': config_z['scale_flow'],
                'flow_config': config_z['flow'],
                'mesh_config': config_z['mesh'],
                'warp_config': config_z['warp'],
                'first_slice': first_slice,
                'yx_target_resolution': yx_target_resolution,
                'save_downsampled': save_downsampled,
                'overwrite': False
            }
            config = add_config_metadata(config)

            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            done.append(dataset_name)

    logging.info(f'Configuration files were created at {output_configs_dir}')

    return root_stack, paths, reverse_order, root_offset


def prep_config_z(project_dir: str,
                  config_z_path: str,
                  config_paths: List[str] = None,
                  reference_path: Optional[str] = None,
                  reference_offset: Optional[int] = 0,
                  destination_path: Optional[str] = None,
                  exclude: List[str] = None,
                  save_downsampled: float = 10,
                  force_overwrite: bool = False) -> str:
    '''Generate Z alignment configuration files.

    Args:
        project_dir: Directory containing the project: config directory, and output zarr
        config_z_path: Path to Z alignment parameters config
        config_paths: List of paths to XY main config files (optional, derived from project_dir if not provided)
        destination_path: Path to output zarr (optional, derived from config if not provided)
        exclude: List of patterns to exclude from datasets
        save_downsampled: Downsampling factor for inspection store
        force_overwrite: Whether to overwrite existing configs

    Returns:
        str: Path to the created config directory
    '''
    if exclude is None:
        exclude = []

    if config_paths is None:
        # Attempt to find the config in the project directory
        config_paths = [os.path.join(project_dir, 'config/xy_config/main_config.json')]
        if not os.path.exists(config_paths[0]):
            raise FileNotFoundError(f'Main config file not found in the project directory: {config_paths[0]}')
        logging.info(f'Config file location was determined from project directory:\n{config_paths[0]}\n')

    # Check if output directory already has configs
    output_configs_dir = os.path.join(project_dir, 'config', 'z_config')
    existing_configs = glob(os.path.join(output_configs_dir, 'z_*.json'))

    if existing_configs and not force_overwrite:
        response = input(f'Config files already exist at {output_configs_dir}.\nOverwrite? [y/N] ')
        if response.lower() != 'y':
            logging.info('Exiting without overwriting existing config')
            sys.exit(0)
        logging.info('Overwriting existing config files')

    # Load Z alignment parameters
    try:
        with open(config_z_path, 'r') as f:
            config_z = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f'Z alignment config file not found: {config_z_path}')
    except json.JSONDecodeError as e:
        raise ValueError(f'Invalid JSON in Z alignment config file {config_z_path}: {e}')

    # Load datasets from XY configs
    logging.info('Loading datasets from XY configuration files...')
    (datasets, z_offsets, yx_target_resolution,
     project_name, mongodb_config_filepath, xy_output_path) = load_configs_from_files(
        config_paths, exclude)

    # Determine destination path
    if destination_path is None:
        destination_path = os.path.join(os.path.abspath(xy_output_path), project_name)
    else:
        destination_path = os.path.join(os.path.abspath(destination_path), project_name)

    if reference_path is not None:
        if os.path.exists(reference_path):
            res_ref = get_store_attributes(reference_path)['resolution']
            logging.info(f'Reference for alignment: {reference_path}')
            logging.info(f'Reference resolution: {res_ref}\n')
            logging.info(f'Reference XY resolution scaling: {res_ref[-1]/yx_target_resolution}')
        else:
            raise FileNotFoundError(f'Provided reference path does not exist: {reference_path}')
    
    # Print dataset info
    logging.info('Datasets Z offsets:')
    for dataset, z in zip(datasets, z_offsets):
        yx_res = get_store_attributes(dataset)['resolution'][1:]
        logging.info(f'    {z[0]} (res: {yx_res}): {os.path.basename(os.path.abspath(dataset.kvstore.path))}')

    if isinstance(yx_target_resolution, list):
        yx_target_resolution = np.min(yx_target_resolution, axis=0).tolist()

    logging.info(f'Target resolution (yx): {yx_target_resolution}\n')
    logging.info(f'Destination path: {destination_path}\n')

    # Create configuration files
    root_stack, paths, reverse_order, root_offset = create_alignment_configs(
        datasets, z_offsets, output_configs_dir, config_z, reference_path, reference_offset,
        destination_path, project_name, mongodb_config_filepath,
        yx_target_resolution, save_downsampled
    )

    # Validate created configs
    is_valid, errors, warnings = validate_config_directory(output_configs_dir)

    if warnings:
        for warning in warnings:
            logging.warning(warning)

    if not is_valid:
        for error in errors:
            logging.error(error)
        raise RuntimeError('Created configuration files are invalid')

    logging.info(f'Config version: {CONFIG_VERSION}')
    logging.info(f'Root stack: {root_stack}')
    logging.info(f'Number of alignment paths: {len(paths)}')
    logging.info(f'\nConfiguration complete!')
    logging.info(f'Config directory: {output_configs_dir}')
    logging.info(f'\n\nTo run alignment:\nCUDA_VISIBLE_DEVICES=0,1 python align_dataset_z.py -p {project_dir} -c 1')

    return output_configs_dir


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Prepare configuration files for Z alignment.')

    # Required arguments
    parser.add_argument('-p', '--project-dir',
                        metavar='PROJECT_DIR',
                        dest='project_dir',
                        required=True,
                        type=str,
                        help='Directory where the config will be written.')
    parser.add_argument('-cfg-z', '--config-z',
                        metavar='CONFIG_Z_PATH',
                        dest='config_z_path',
                        required=True,
                        type=str,
                        help='Path to Z alignment parameters config.')

    # Optional arguments
    parser.add_argument('-cfg', '--config',
                        metavar='CONFIG_PATHS',
                        dest='config_paths',
                        nargs='+',
                        type=str,
                        default=None,
                        help='Path(s) to XY main config file(s). Default: main config is assumed to exist at "project_dir/config/xy_config/main_config.json"')
    parser.add_argument('-d', '--destination',
                        metavar='DESTINATION',
                        dest='destination_path',
                        type=str,
                        default=None,
                        help='Path to output zarr. Default: derived from XY config')
    parser.add_argument('-r', '--reference',
                        metavar='REFERENCE_PATH',
                        dest='reference_path',
                        type=str,
                        default=None,
                        help='Path to an existing zarr to use as reference for alignment.'
                             'Each dataset will be aligned to this reference instead of to each other. Default: No reference, use previous slice of same dataset')
    parser.add_argument('--reference-offset',
                        metavar='REFERENCE_OFFSET',
                        dest='reference_offset',
                        type=int,
                        default=0,
                        help='Z offset between each dataset and its corresponding reference slice. Default: 0')
    parser.add_argument('--exclude',
                        metavar='EXCLUDE',
                        dest='exclude',
                        type=str,
                        nargs='+',
                        default=[],
                        help='Patterns to exclude from datasets. Default: all datasets are processed')
    parser.add_argument('-ds', '--downsample-scale',
                        metavar='SCALE',
                        dest='save_downsampled',
                        type=float,
                        default=10,
                        help='Downsampling factor for inspection store. Default: 10')
    parser.add_argument('--force-overwrite',
                        dest='force_overwrite',
                        action='store_true',
                        default=False,
                        help='Force overwrite of existing config files. Default: user is prompted if configs exist')

    args = parser.parse_args()

    prep_config_z(**vars(args))
