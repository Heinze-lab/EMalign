import argparse
import json
import logging
import os
import sys

from glob import glob
from typing import List, Optional

from emalign.align_z.params import CLEAN_DEFAULTS, MESH_DEFAULTS, check_step_config
from emalign.align_z.config import add_config_metadata, get_fuse_config_dir, load_fuse_plan, validate_config_directory, CONFIG_VERSION
from emalign.align_z.utils import compute_alignment_path, compute_dataset_bounds, get_ordered_datasets
from emalign.io.store import get_store_attributes

logger = logging.getLogger(__name__)
logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('jax._src.xla_bridge').setLevel(logging.WARNING)

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


def project_dir_from_config(config_path):
    '''Find the project directory an XY main config file belongs to.

    Main configs live at project_dir/config/xy_config/main_config.json.

    Args:
        config_path: Path to an XY main config file

    Returns:
        str: Path to the project directory
    '''
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(config_path))))


def check_fuse_step(config_paths, datasets):
    '''Check that every group of overlapping stacks that was planned has actually been fused.

    Fusing is optional, but a project where the fusing step was prepared and never executed still
    has several stacks covering the same Z slices. Alignment would then either render them twice or
    fail to find a root stack, so it is reported here instead. Datasets can come from several XY
    configs, each belonging to its own project directory with its own fuse configuration.

    Args:
        config_paths: List of paths to XY main config files, possibly nested in groups
        datasets: List of tensorstore datasets discovered for alignment

    Raises:
        RuntimeError: If a fused stack that was planned does not exist
    '''
    # Configs of consecutive stacks are passed as nested lists
    flat_config_paths = []
    for config_path in config_paths:
        if isinstance(config_path, list):
            flat_config_paths += config_path
        else:
            flat_config_paths.append(config_path)

    dataset_names = [os.path.basename(os.path.abspath(d.kvstore.path)) for d in datasets]

    missing = {}
    for config_path in flat_config_paths:
        project_dir = project_dir_from_config(config_path)
        plan, group_configs = load_fuse_plan(project_dir)

        if plan is None:
            logger.info(f'No fusing step was prepared for {project_dir}')
            continue

        if plan.get('_config_version') != CONFIG_VERSION:
            logger.warning(f'Fuse config files were generated with another version '
                            f'({plan.get("_config_version")} instead of {CONFIG_VERSION})')

        if not group_configs:
            logger.info(f'No stacks were found to overlap in {project_dir}')
            continue

        logger.info(f'Fused stacks ({get_fuse_config_dir(project_dir)}):')
        for config in group_configs:
            destination_name = config['destination_name']
            sources = ' + '.join(config['fused_from'])
            logger.info(f'    z {config["zmin"]}-{config["zmax"]}: {sources} -> {destination_name}')
            if destination_name not in dataset_names:
                missing.setdefault(project_dir, []).append(destination_name)
    logger.info('')

    if missing:
        message = ['Stacks overlapping on the same Z slices must be fused before Z alignment.',
                   'The following stacks were planned for fusing but do not exist yet:']
        for project_dir, names in missing.items():
            message.append(f'    {project_dir}: {names}')
            message.append(f'        Run: CUDA_VISIBLE_DEVICES=0 python emalign.scripts.fuse_stacks_xy -p {project_dir}')
        raise RuntimeError('\n'.join(message))


def create_alignment_configs(datasets, z_offsets, output_configs_dir, config_z, reference_path,
                             reference_offset, destination_path, project_name, mongodb_config_filepath,
                             yx_target_resolution, save_downsampled):
    '''Create alignment configuration files for all datasets.

    Args:
        datasets: List of tensorstore datasets
        z_offsets: Array of z offsets for each dataset
        output_configs_dir: Directory to store config files
        config_z: Z alignment configuration dictionary
        reference_path: Path to a reference zarr every dataset aligns to, or None
        reference_offset: Z offset between each dataset and its reference slice
        destination_path: Path to output zarr
        project_name: Name of the project
        mongodb_config_filepath: Path to MongoDB config
        yx_target_resolution: Target resolution in YX
        save_downsampled: Downsampling factor

    Returns:
        tuple: (root_stack, paths, reverse_order)
    '''
    # Clean-up parameters may still be given in the flow section (older Z configs)
    flow_config = {k: v for k, v in config_z['flow'].items() if k not in CLEAN_DEFAULTS}
    clean_config = {k: v for k, v in config_z['flow'].items() if k in CLEAN_DEFAULTS}
    clean_config = CLEAN_DEFAULTS | check_step_config('clean', clean_config | config_z.get('clean', {}))
    mesh_config = MESH_DEFAULTS | check_step_config('mesh', config_z['mesh'])

    logger.info('Creating Z align configuration files...')
    logger.info(f'Configuration files will be stored at: \n    {output_configs_dir}\n')

    if reference_path is None:
        # There is no reference dataset, compute the paths between datasets. 
        # Some stacks may be disconnected in some parts of the dataset
        logger.info('Computing alignment path...')
        root_stack, paths, reverse_order, ds_bounds = compute_alignment_path(
            datasets, z_offsets, target_resolution=yx_target_resolution)
    else:
        # Each dataset aligns to the reference independently, matched by Z index, so no
        # dataset is ever aligned to another one. Testing the transitions between datasets
        # would only verify a relationship the alignment never uses, so only the Z bounds
        # are computed, to avoid rendering a fused dataset and its sub-datasets twice.
        logger.info('Computing dataset bounds...')
        dataset_names, ds_bounds = compute_dataset_bounds(datasets, z_offsets)
        root_stack = dataset_names[0]
        paths = [dataset_names]  # Iteration order only, this is not an alignment chain
        reverse_order = [False]

    # Write alignment plan
    align_plan = {
        'root_stack': root_stack,
        'destination_path': destination_path,
        'reference_path': reference_path,
        'align_to_reference': reference_path is not None,
        'paths': paths,
        'reverse_order': reverse_order,
        'yx_target_resolution': yx_target_resolution,
        'project_name': project_name,
        'save_downsampled': save_downsampled
    }
    align_plan = add_config_metadata(align_plan)

    os.makedirs(output_configs_dir, exist_ok=True)
    with open(os.path.join(output_configs_dir, '00_align_plan.json'), 'w') as f:
        json.dump(align_plan, f, indent=2)

    # Write configs for each dataset
    done = []
    for path, order in zip(paths, reverse_order):
        for dataset_name in path:
            if dataset_name in done:
                continue
            idx = [os.path.basename(os.path.abspath(d.kvstore.path)) == dataset_name for d in datasets].index(True)
            dataset = datasets[idx]
            z_offset = int(z_offsets[idx, 0]) + ds_bounds[dataset_name][0]
            config_path = os.path.join(output_configs_dir, f'z_{dataset_name}.json')

            config = {
                'destination_path': destination_path,
                'dataset_path': os.path.abspath(dataset.kvstore.path),
                'dataset_name': dataset_name,
                'reference_path': reference_path,
                'align_to_reference': reference_path is not None,  # Every slice aligns to its match in the reference
                'reference_offset': reference_offset + z_offset - ds_bounds[dataset_name][0],
                'max_gap': config_z.get('max_gap', 2),  # Slices chained to a neighbour when the reference is missing
                'alignment_path': path,
                'reverse_order': order,
                'project_name': project_name,
                'mongodb_config_filepath': mongodb_config_filepath,
                'z_offset': z_offset,
                'local_z_min': ds_bounds[dataset_name][0],
                'local_z_max': ds_bounds[dataset_name][1],
                'scale': config_z['scale_flow'],
                'flow_config': flow_config,
                'clean_config': clean_config,
                'mesh_config': mesh_config,
                'warp_config': config_z['warp'],
                'yx_target_resolution': yx_target_resolution,
                'save_downsampled': save_downsampled
            }
            config = add_config_metadata(config)

            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            done.append(dataset_name)

    logger.info(f'Configuration files were created at {output_configs_dir}')

    return root_stack, paths, reverse_order


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
        logger.info(f'Config file location was determined from project directory:\n{config_paths[0]}\n')

    # Check if output directory already has configs
    output_configs_dir = os.path.join(project_dir, 'config', 'z_config')
    existing_configs = glob(os.path.join(output_configs_dir, 'z_*.json'))

    if existing_configs and not force_overwrite:
        response = input(f'Config files already exist at {output_configs_dir}.\nOverwrite? [y/N] ')
        if response.lower() != 'y':
            logger.info('Exiting without overwriting existing config')
            sys.exit(0)
        logger.info('Overwriting existing config files')

    # Load Z alignment parameters
    try:
        with open(config_z_path, 'r') as f:
            config_z = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f'Z alignment config file not found: {config_z_path}')
    except json.JSONDecodeError as e:
        raise ValueError(f'Invalid JSON in Z alignment config file {config_z_path}: {e}')

    # Load datasets from XY configs
    logger.info('Loading datasets from XY configuration files...')
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
            logger.info(f'Reference for alignment: {reference_path}')
            logger.info(f'Reference resolution: {res_ref}\n')
            logger.info(f'Reference XY resolution scaling: {res_ref[-1]/yx_target_resolution}')
        else:
            raise FileNotFoundError(f'Provided reference path does not exist: {reference_path}')
    
    # Print dataset info
    logger.info('Datasets Z offsets:')
    for dataset, z in zip(datasets, z_offsets):
        yx_res = get_store_attributes(dataset)['resolution'][1:]
        logger.info(f'    {z[0]} (res: {yx_res}): {os.path.basename(os.path.abspath(dataset.kvstore.path))}')
    logger.info('')

    # Stacks overlapping on the same Z slices must have been fused beforehand
    check_fuse_step(config_paths, datasets)

    logger.info(f'Target resolution (yx): {yx_target_resolution}\n')
    logger.info(f'Destination path: {destination_path}\n')

    # Create configuration files
    root_stack, paths, reverse_order = create_alignment_configs(
        datasets, z_offsets, output_configs_dir, config_z, reference_path, reference_offset,
        destination_path, project_name, mongodb_config_filepath,
        yx_target_resolution, save_downsampled
    )

    # Validate created configs
    is_valid, errors, warnings = validate_config_directory(output_configs_dir)

    if warnings:
        for warning in warnings:
            logger.warning(warning)

    if not is_valid:
        for error in errors:
            logger.error(error)
        raise RuntimeError('Created configuration files are invalid')

    logger.info(f'Config version: {CONFIG_VERSION}')
    logger.info(f'Root stack: {root_stack}')
    logger.info(f'Number of alignment paths: {len(paths)}')
    logger.info(f'\nConfiguration complete!')
    logger.info(f'Config directory: {output_configs_dir}')
    logger.info(f'\n\nTo compute transforms:\npython -m emalign.run_alignment z transform -p {project_dir}')

    return output_configs_dir


def add_parser_arguments(parser):
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
                        help='Path to Z alignment parameters config. This is a JSON file containing parameters used for computing flow, mesh optimization, and for rendering.')

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
    return parser

if __name__ == '__main__':
    
    from emalign.utils.logging_utils import setup_logging
    setup_logging(logging.INFO)

    parser = argparse.ArgumentParser(
        description='Prepare configuration files for Z alignment.',
        formatter_class=argparse.RawDescriptionHelpFormatter
        )
    args = add_parser_arguments(parser).parse_args()
    prep_config_z(**vars(args))
