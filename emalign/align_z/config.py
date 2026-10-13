'''
Configuration utilities for Z alignment.
'''

import datetime
import json
import logging
import os
from glob import glob


CONFIG_VERSION = '1.0'

REQUIRED_ALIGN_PLAN_FIELDS = [
    'root_stack',
    'paths',
    'reverse_order',
    'root_offset',
    'pad_offset',
    'yx_target_resolution',
    'dataset_local_bounds'
]

REQUIRED_DATASET_CONFIG_FIELDS = [
    'destination_path',
    'dataset_path',
    'dataset_name',
    'alignment_path',
    'reverse_order',
    'z_offset',
    'xy_offset',
    'local_z_min',
    'local_z_max',
    'scale',
    'flow_config',
    'mesh_config',
    'warp_config',
    'first_slice',
    'yx_target_resolution'
]


def validate_align_plan(plan_dict):
    '''Validate 00_align_plan.json structure.

    Args:
        plan_dict: Dictionary loaded from 00_align_plan.json

    Returns:
        list: List of error messages (empty if valid)
    '''
    errors = []
    for field in REQUIRED_ALIGN_PLAN_FIELDS:
        if field not in plan_dict:
            errors.append(f'Missing required field: {field}')

    # Validate paths and reverse_order have same length
    if 'paths' in plan_dict and 'reverse_order' in plan_dict:
        if len(plan_dict['paths']) != len(plan_dict['reverse_order']):
            errors.append(f'paths ({len(plan_dict["paths"])}) and reverse_order '
                         f'({len(plan_dict["reverse_order"])}) must have same length')

    return errors

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


def validate_dataset_config(config_dict, config_path):
    '''Validate z_*.json structure.

    Args:
        config_dict: Dictionary loaded from z_*.json
        config_path: Path to the config file (for error messages)

    Returns:
        list: List of error messages (empty if valid)
    '''
    errors = []
    filename = os.path.basename(config_path)

    for field in REQUIRED_DATASET_CONFIG_FIELDS:
        if field not in config_dict:
            errors.append(f'{filename}: Missing required field: {field}')

    # Validate dataset_path exists
    if 'dataset_path' in config_dict:
        if not os.path.exists(config_dict['dataset_path']):
            errors.append(f'{filename}: dataset_path does not exist: {config_dict["dataset_path"]}')

    return errors


def validate_config_directory(config_dir):
    '''Validate a complete config directory.

    Args:
        config_dir: Path to directory containing z_*.json and 00_align_plan.json

    Returns:
        tuple: (is_valid, errors, warnings)
            - is_valid: True if no critical errors
            - errors: List of error messages
            - warnings: List of warning messages
    '''
    errors = []
    warnings = []

    # Check align plan exists
    align_plan_path = os.path.join(config_dir, '00_align_plan.json')
    if not os.path.exists(align_plan_path):
        errors.append(f'Missing 00_align_plan.json in {config_dir}')
        return False, errors, warnings

    # Load and validate align plan
    try:
        with open(align_plan_path, 'r') as f:
            align_plan = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f'Invalid JSON in 00_align_plan.json: {e}')
        return False, errors, warnings

    plan_errors = validate_align_plan(align_plan)
    errors.extend(plan_errors)

    # Get all dataset names from paths
    all_datasets = set()
    for path in align_plan.get('paths', []):
        all_datasets.update(path)

    # Check each dataset has a config
    config_files = glob(os.path.join(config_dir, 'z_*.json'))
    found_datasets = set()

    for config_path in config_files:
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            errors.append(f'Invalid JSON in {os.path.basename(config_path)}: {e}')
            continue

        config_errors = validate_dataset_config(config, config_path)
        errors.extend(config_errors)

        if 'dataset_name' in config:
            found_datasets.add(config['dataset_name'])

    # Check all datasets in plan have configs
    missing = all_datasets - found_datasets
    if missing:
        errors.append(f'Missing config files for datasets: {missing}')

    # Check for orphan configs (warning only)
    orphans = found_datasets - all_datasets
    if orphans:
        warnings.append(f'Config files exist for datasets not in alignment plan: {orphans}')

    # Check config version (warning only)
    if '_config_version' not in align_plan:
        warnings.append('Config files missing version metadata (generated with older version)')

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def add_config_metadata(config_dict):
    '''Add metadata to config for versioning/staleness detection.

    Args:
        config_dict: Configuration dictionary to modify

    Returns:
        dict: Modified config with metadata added
    '''
    config_dict['_config_version'] = CONFIG_VERSION
    config_dict['_created_at'] = datetime.datetime.now().isoformat()
    return config_dict


def load_align_plan(config_dir):
    '''Load and validate the alignment plan from a config directory.

    Args:
        config_dir: Path to config directory

    Returns:
        dict: Loaded align plan

    Raises:
        FileNotFoundError: If 00_align_plan.json doesn't exist
        ValueError: If plan is invalid
    '''
    align_plan_path = os.path.join(config_dir, '00_align_plan.json')

    if not os.path.exists(align_plan_path):
        raise FileNotFoundError(f'Alignment plan not found: {align_plan_path}')

    with open(align_plan_path, 'r') as f:
        align_plan = json.load(f)

    errors = validate_align_plan(align_plan)
    if errors:
        raise ValueError(f'Invalid alignment plan: {"; ".join(errors)}')

    return align_plan


def load_dataset_configs(config_dir):
    '''Load all dataset configs from a config directory.

    Args:
        config_dir: Path to config directory

    Returns:
        dict: Mapping of dataset_name -> config dict

    Raises:
        FileNotFoundError: If no config files found
        ValueError: If any config is invalid
    '''
    config_files = glob(os.path.join(config_dir, 'z_*.json'))

    if not config_files:
        raise FileNotFoundError(f'No z_*.json config files found in {config_dir}')

    configs = {}
    all_errors = []

    for config_path in config_files:
        with open(config_path, 'r') as f:
            config = json.load(f)

        errors = validate_dataset_config(config, config_path)
        all_errors.extend(errors)

        if 'dataset_name' in config:
            configs[config['dataset_name']] = config

    if all_errors:
        raise ValueError(f'Invalid dataset configs: {"; ".join(all_errors)}')

    return configs


#---------- Fusing of XY aligned stacks ----------#
# The configuration files are written by scripts/fuse_stacks_xy.py. They are read here because
# Z alignment is what consumes the fused stacks, and must know whether the fusing step was run.

def get_fuse_config_dir(project_dir):
    '''Path to the directory holding the fuse configuration files of a project.

    Args:
        project_dir: Directory containing the project: config directory, and output zarr

    Returns:
        str: Path to project_dir/config/fuse_config
    '''
    return os.path.join(project_dir, 'config', 'fuse_config')


def load_fuse_plan(project_dir):
    '''Load the fuse plan and the configuration of every group of stacks to fuse.

    Fusing is optional: a project where no stacks overlap has no fuse configuration at all.

    Args:
        project_dir: Directory containing the project: config directory, and output zarr

    Returns:
        tuple: (plan, group_configs) where both are None if no fuse plan exists.

    Raises:
        FileNotFoundError: If the plan exists but references a missing group config
    '''
    config_dir = get_fuse_config_dir(project_dir)
    plan_path = os.path.join(config_dir, '00_fuse_plan.json')

    if not os.path.exists(plan_path):
        return None, None

    with open(plan_path, 'r') as f:
        plan = json.load(f)

    group_configs = []
    for filename in plan.get('group_configs', []):
        config_path = os.path.join(config_dir, filename)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f'Fuse plan references a missing config file: {config_path}')
        with open(config_path, 'r') as f:
            group_configs.append(json.load(f))

    return plan, group_configs
