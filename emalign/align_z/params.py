import dataclasses
import datetime
import difflib
import json
import logging
import os

logger = logging.getLogger(__name__)

# Arguments of combine_flow (sofima.flow_utils.clean_flow and reconcile_flows)
CLEAN_DEFAULTS = {
    'min_peak_ratio': 1.6,
    'min_peak_sharpness': 1.6,
    'max_magnitude': 80.,
    'max_deviation': 350.,
    'max_gradient': 0.,
    'min_patch_size': 400
}

# Arguments of sofima.mesh.IntegrationConfig. Fields left out use the defaults of SOFIMA.
# The stride is the one of the flow, so it is not a mesh parameter.
MESH_DEFAULTS = {
    'dt': 0.001,
    'gamma': 0.5,
    'k0': 0.01,
    'k': 0.4,
    'num_iters': 1000,
    'max_iters': 100000,
    'stop_v_max': 0.01,
    'dt_max': 1000,
    'start_cap': 0.1,
    'final_cap': 1.0,
    'prefer_orig_order': True
}

# Parameters of the flow itself, changing them means computing the flow again
FLOW_PARAMETERS = ('patch_size', 'stride', 'scale_flow')

# Config key of each step, in pipeline order: changing a step invalidates the following ones
STEP_CONFIG_KEYS = {'clean': 'clean_config', 'mesh': 'mesh_config'}


def get_step_parameters():
    '''Parameters that can be set for each step, with their type.

    Returns:
        dict: step -> {parameter name: type}
    '''
    from sofima.mesh import IntegrationConfig

    mesh_params = {f.name: f.type for f in dataclasses.fields(IntegrationConfig)
                   if f.type in (bool, int, float)}
    params = {
        'clean': {name: type(value) for name, value in CLEAN_DEFAULTS.items()},
        'mesh': mesh_params
    }

    shared = set(params['clean']) & set(params['mesh'])
    if shared:
        # A parameter name must identify its step
        raise RuntimeError(f'Parameters shared by the clean and mesh steps: {shared}')
    return params


def _get_param_type(name, value, expected_type):
    '''Convert a value to the required type of a parameter, or raise ValueError.'''
    if expected_type is bool:
        if isinstance(value, str) and value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        if isinstance(value, bool) or value in (0, 1):
            return bool(value)
    elif expected_type is int:
        if not isinstance(value, bool) and isinstance(value, (int, float)) and float(value).is_integer():
            return int(value)
    elif expected_type is float:
        if not isinstance(value, bool) and isinstance(value, (int, float)):
            return float(value)
    raise ValueError(f'Invalid value for {name}: {value!r} (expected {expected_type.__name__})')


def _unknown_parameter(name, params):
    '''Error message for a parameter that does not exist, with the closest names.'''
    if name in FLOW_PARAMETERS:
        return (f'{name} is a parameter of the flow itself, which would have to be computed again. '
                'Change it in the Z config and run z prep again instead.')
    valid = [p for step_params in params.values() for p in step_params]
    message = f'Unknown parameter: {name}.'
    close = difflib.get_close_matches(name, valid, n=3)
    if close:
        message += f' Did you mean: {", ".join(close)}?'
    for step, step_params in params.items():
        message += f'\n    {step}: {", ".join(sorted(step_params))}'
    return message


def check_step_config(step, config, params=None):
    '''Check the names and types of the parameters of a step.

    Args:
        step: 'clean' or 'mesh'
        config: Parameters of the step
        params: Output of get_step_parameters, computed if not given

    Returns:
        dict: Parameters converted to their type

    Raises:
        ValueError: If a parameter does not exist for this step or has the wrong type
    '''
    params = params or get_step_parameters()
    checked = {}
    for name, value in config.items():
        if name not in params[step]:
            raise ValueError(f'{step}: ' + _unknown_parameter(name, params))
        checked[name] = _get_param_type(name, value, params[step][name])
    return checked


def resolve_clean_config(ds_config):
    '''Parameters used to clean the flow of a dataset.'''
    legacy = {k: v for k, v in ds_config.get('flow_config', {}).items() if k in CLEAN_DEFAULTS}
    config = check_step_config('clean', legacy | ds_config.get('clean_config', {}))
    return CLEAN_DEFAULTS | config


def resolve_mesh_config(ds_config):
    '''Parameters used to relax the mesh of a dataset, without the stride.'''
    return MESH_DEFAULTS | check_step_config('mesh', ds_config.get('mesh_config', {}))


def parse_parameters(parameters):
    '''Parse name=value parameters given on the command line.

    Values are read as JSON (10, 0.5, true), names must belong to the clean or mesh step.

    Args:
        parameters: List of 'name=value' strings

    Returns:
        dict: step -> {name: value}
    '''
    params = get_step_parameters()
    parsed = {}
    for param in parameters:
        name, sep, value = param.partition('=')
        name = name.strip()
        if not sep or not name or not value.strip():
            raise ValueError(f'Invalid parameter: {param!r}, expected name=value')
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = value.strip()

        step = next((s for s, step_params in params.items() if name in step_params), None)
        if step is None:
            raise ValueError(_unknown_parameter(name, params))
        if name in parsed.get(step, {}):
            raise ValueError(f'Parameter given more than once: {name}')
        parsed.setdefault(step, {})[name] = _get_param_type(name, value, params[step][name])
    return parsed


def apply_parameters(config_dir, dataset_configs, parameters, stacks=None):
    '''Write new parameters into the configs of some datasets.

    The configs are updated in place and written to file, so that the parameters stay those of
    the project. Legacy clean-up parameters in flow_config are moved to clean_config.

    Args:
        config_dir: Directory of the z_<dataset>.json files
        dataset_configs: dataset_name -> config, as loaded from config_dir
        parameters: Output of parse_parameters
        stacks: Datasets to update, all if None or empty

    Returns:
        dict: dataset_name -> first step whose parameters changed, for the datasets that changed
    '''
    stacks = list(stacks or dataset_configs)
    unknown = [s for s in stacks if s not in dataset_configs]
    if unknown:
        raise RuntimeError(f'No configuration found for dataset(s): {unknown}')

    changed = {}
    for dataset_name in stacks:
        config = dataset_configs[dataset_name]
        current = {'clean': resolve_clean_config(config), 'mesh': resolve_mesh_config(config)}

        changed_steps = []
        for step, values in parameters.items():
            diff = {k: v for k, v in values.items() if current[step][k] != v}
            if not diff:
                continue
            changed_steps.append(step)
            for k, v in diff.items():
                logger.info(f'{dataset_name}: {k} {current[step][k]} -> {v}')

        if not changed_steps:
            logger.info(f'{dataset_name}: Parameters unchanged')
            continue

        # Move legacy clean-up parameters out of flow_config
        # This is for older versions and can be removed eventually
        flow_config = config.get('flow_config', {})
        legacy = {k: flow_config.pop(k) for k in list(flow_config) if k in CLEAN_DEFAULTS}
        config['clean_config'] = legacy | config.get('clean_config', {})

        for step, values in parameters.items():
            key = STEP_CONFIG_KEYS[step]
            config[key] = config.get(key, {}) | values
        config['_modified_at'] = datetime.datetime.now().isoformat()

        config_path = os.path.join(config_dir, f'z_{dataset_name}.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f'Config file not found: {config_path}')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        order = list(STEP_CONFIG_KEYS)
        changed[dataset_name] = min(changed_steps, key=order.index)
    return changed


def same_parameters(stored, current):
    '''Whether the parameters recorded in a store match the current ones.

    Stores written before the parameters were recorded have none, they are assumed to match.
    '''
    if stored is None:
        return True
    return json.loads(json.dumps(stored)) == json.loads(json.dumps(current))
