import copy
import json
import logging


# Elastic mesh parameters for XY stitching, written to each stack config (xy_<stack>.json) under
# 'mesh_parameters' by prep_config_xy. Tiles are stitched with 'default', unless they overlap by
# less than SMALL_OVERLAP_THRESHOLD pixels, in which case 'small_overlap' is used. These parameters
# work better with very small overlaps.
# A stride of None means the stride of the main config is used.
SMALL_OVERLAP_THRESHOLD = 160

MESH_PARAMETERS_DEFAULTS = {
    'default': {
        'stride': None,
        'patch_size': 160,
        'k0': 0.01,
        'k': 0.1,
        'gamma': 0.5
    },
    # Parameters tested for very small overlap
    'small_overlap': {
        'stride': 10,
        'patch_size': 30,
        'k0': 0.07,
        'k': 0.2,
        'gamma': 0.5,
        'margin': 10
    }
}


def default_mesh_parameters(stride=None):
    '''Mesh parameters to write in a new stack config.

    Args:
        stride (int, optional): Stride used for the 'default' parameters. Defaults to None (main config stride).
    '''
    params = copy.deepcopy(MESH_PARAMETERS_DEFAULTS)
    params['default']['stride'] = stride
    return params


def parse_mesh_parameters(config_path, stride):
    '''Read the mesh parameters of a stack config, filling missing values with defaults.

    Args:
        config_path (str): Path to the stack config (xy_<stack>.json).
        stride (int): Stride of the main config, used when the 'default' stride is None.

    Returns:
        dict: {'default': {...}, 'small_overlap': {...}}
    '''
    with open(config_path, 'r') as f:
        mesh_params = json.load(f).get('mesh_parameters', {})

    unknown = set(mesh_params) - set(MESH_PARAMETERS_DEFAULTS)
    if unknown:
        raise ValueError(f'{config_path}: unknown mesh_parameters group(s) {sorted(unknown)}, '
                         f'expected {sorted(MESH_PARAMETERS_DEFAULTS)}')

    params = copy.deepcopy(MESH_PARAMETERS_DEFAULTS)
    for group, defaults in params.items():
        given = mesh_params.get(group, {})
        unknown = set(given) - set(defaults)
        if unknown:
            raise ValueError(f'{config_path}: unknown key(s) {sorted(unknown)} in mesh_parameters.{group}, '
                             f'expected {sorted(defaults)}')
        defaults.update(given)

    if params['default']['stride'] is None:
        params['default']['stride'] = stride

    if not mesh_params:
        logging.info(f'{config_path}: no mesh_parameters found, using defaults')
    return params
