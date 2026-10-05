import os
import argparse
import logging
import sys
import numpy as np
import tensorstore as ts

from inspect import signature
from sofima.mesh import IntegrationConfig

from emalign.align_z.flow import compute_flow_stack, combine_flow, get_inv_map
from emalign.io.progress import get_mongo_client, get_mongo_db, wipe_progress, log_progress, check_progress
from emalign.io.store import open_store, find_ref_slice, get_store_attributes, set_store_attributes
from emalign.scripts.compute_transforms import load_and_validate_configs
from emalign.io.utils import get_zarr_root
from emalign.utils.logging_utils import setup_logging


logging.basicConfig(level=logging.INFO)
setup_logging(logging.INFO)

# Steps of this script, in order, and their name in the progress database
STEPS = {'flow': 'flow_z', 'clean': 'flow_clean_z', 'mesh': 'mesh_relax_z'}


def compute_dataset_flow(
        project_dir,
        num_workers=0,
        start_over=False,
        wipe_progress_stacks=None,
        wipe_steps=None
        ):
    '''Compute the optical flow of every stack, following the alignment paths.

    Requires the final transforms, run compute_transforms first. Each stack uses the
    same reference as for the transforms: the previous stack of its alignment path.
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
    project_name = align_plan['project_name']

    logging.info(f'Project: {project_name}')
    logging.info(f'Root stack: {root_stack}')
    logging.info(f'Number of alignment paths: {len(paths)}\n')
    logging.info('ToDo list:')
    mentioned = []
    for path in paths:
        for dataset_name in path:
            if dataset_name in mentioned:
                continue
            logging.info(f'    - {dataset_name}')
            mentioned.append(dataset_name)
    print()

    if any(reverse_order):
        raise NotImplementedError('reverse_order not yet implemented.')

    # Progress logging, shared by all datasets
    first_config = next(iter(dataset_configs.values()))
    db = get_mongo_db(get_mongo_client(first_config.get('mongodb_config_filepath')), project_name)

    # Handle start_over
    if start_over:
        try:
            input('WARNING: All flow, clean-up and mesh progress will be wiped and all datasets will be processed.\n'
                  'Press ENTER to continue or CTRL+C to abort\n')
        except KeyboardInterrupt:
            logging.info('\nAborted by user')
            sys.exit(0)
        wipe_progress_stacks = list(dataset_configs)
        wipe_steps = list(STEPS)

    # Wipe progress for the requested steps and datasets:
    #   - steps only: all datasets for these steps
    #   - datasets only: all steps for these datasets
    #   - both: these steps for these datasets
    wipe_progress_stacks = [s for s in (wipe_progress_stacks or []) if s]
    wipe_steps = list(wipe_steps or [])
    if wipe_progress_stacks and not wipe_steps:
        wipe_steps = ['flow', 'clean', 'mesh']
    if wipe_steps:
        unknown = [s for s in wipe_steps if s not in STEPS]
        if unknown:
            raise ValueError(f'Unknown step(s): {unknown}. Choose from: {list(STEPS)}')
        unknown = [d for d in wipe_progress_stacks if d not in dataset_configs]
        if unknown:
            raise RuntimeError(f'No configuration found for dataset(s): {unknown}')
        datasets_to_wipe = wipe_progress_stacks or list(dataset_configs)

        # Later steps depend on earlier ones, so they are wiped too
        order = list(STEPS)
        steps_to_wipe = order[min(order.index(s) for s in wipe_steps):]
        if set(steps_to_wipe) != set(wipe_steps):
            logging.info(f'Later steps depend on the wiped ones, also wiping: '
                         f'{[s for s in steps_to_wipe if s not in wipe_steps]}')

        for dataset_name in datasets_to_wipe:
            for step in steps_to_wipe:
                wipe_progress(db, dataset_name, step_name=STEPS[step]) # database progress
            logging.info(f'Wiped {steps_to_wipe} progress for {dataset_name}')

    #------ FIRST PASS: Compute flow ------
    logging.info('Starting flow computation...')
    logging.info(f'Number of cores used for computing flow: {num_workers}\n')
    params = signature(compute_flow_stack).parameters
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
            config['reference_dataset'] = prev_dataset

            # Compute flow at downsampled scale
            logging.info(dataset_name)
            relevant_args = {k: v for k, v in config.items() if k in params}
            compute_flow_stack(**relevant_args)

            # And flow at full scale
            relevant_args['scale'] = 1
            compute_flow_stack(**relevant_args)

            prev_dataset = config['dataset_path']
    print()
    
    #------ SECOND PASS: Clean up flow ------
    # Combine both scales of flow
    logging.info('Combining and cleaning up flow...')
    dataset_names = set(np.concatenate(paths))
    cleaned = set()  # Datasets cleaned in this run, their mesh has to be relaxed again
    for dataset_name in dataset_names:
        if dataset_name not in dataset_configs:
            raise RuntimeError(f'No configuration found for dataset: {dataset_name}')
        
        logging.info(dataset_name)

        if check_progress(db, dataset_name, STEPS['clean'], None):
            logging.info(f'    Flow already cleaned up.')
            continue

        config = dataset_configs[dataset_name].copy()
        ds_scale = config['scale']

        # Load flows
        zarr_path = get_zarr_root(config['destination_path'])
        scale_str = str(round(ds_scale, 2)).replace('.', '_')
        ds_flow_path = os.path.join(zarr_path, 'z_intermediate', f'flow{scale_str}x', dataset_name)
        ds_flow = open_store(ds_flow_path, mode='r').read().result()
        ds_flow = np.transpose(ds_flow, [1, 0, 2, 3])

        scale_str = str(round(1, 2)).replace('.', '_')
        flow_path = os.path.join(zarr_path, 'z_intermediate', f'flow{scale_str}x', dataset_name)
        flow = open_store(flow_path, mode='r').read().result()
        flow = np.transpose(flow, [1, 0, 2, 3])

        # Clean and combine flows at two scales
        final_flow = combine_flow(
                flow, 
                ds_flow,
                stride=config['flow_config']['stride'],
                patch_size=config['flow_config']['patch_size'],
                max_magnitude=config['flow_config']['max_magnitude'],
                max_deviation=config['flow_config']['max_deviation'],
                ds_scale=ds_scale,
                dataset_name=dataset_name
                )
        final_flow = np.transpose(final_flow, [1, 0, 2, 3])
        final_flow_path = os.path.join(zarr_path, 'z_intermediate', f'flow_final', dataset_name)
        dataset_flow = open_store(
            final_flow_path,
            mode='w', dtype=ts.float32,
            shape=final_flow.shape, chunks=[1, 2, 128, 128], axis_labels=['z', 'c', 'y', 'x'],
            fill_value=np.nan)
        dataset_flow.write(final_flow).result()
        log_progress(db, dataset_name, STEPS['clean'], None, None, {'skipped': False})
        cleaned.add(dataset_name)

    #------ THIRD PASS: Relax mesh ------
    # Relax mesh, carrying the state from one stack to the next along each path
    logging.info('Relaxing mesh...')
    solved_maps = {}  # dataset_name -> (solved, start) or None if not loaded yet
    relaxed = set()  # Datasets relaxed in this run, the next stacks of their path depend on them

    def _get_solved(dataset_name):
        '''Solved maps of a stack, loaded from file if it was relaxed in a previous run.'''
        if solved_maps[dataset_name] is None:
            zarr_path = get_zarr_root(dataset_configs[dataset_name]['destination_path'])
            store = open_store(os.path.join(zarr_path, 'z_intermediate', 'mesh_solved', dataset_name), mode='r')
            solved_maps[dataset_name] = (store.read().result(), tuple(get_store_attributes(store)['start']))
        return solved_maps[dataset_name]

    for i, path in enumerate(paths):
        prev_dataset = None
        for dataset_name in path:
            if dataset_name not in dataset_configs:
                raise RuntimeError(f'No configuration found for dataset: {dataset_name}')

            if dataset_name == path[0] and i == 0:
                assert dataset_name == root_stack, \
                    f'First dataset ({dataset_name}) of the path is not the root stack ({root_stack})'

            if dataset_name in solved_maps:
                # Already relaxed in a previous path
                prev_dataset = dataset_name
                continue

            logging.info(dataset_name)
            ds_config = dataset_configs[dataset_name]
            stride = ds_config['flow_config']['stride']
            zarr_path = get_zarr_root(ds_config['destination_path'])

            # Relax again if its flow or the stack it starts from changed in this run
            if (check_progress(db, dataset_name, STEPS['mesh'], None)
                    and dataset_name not in cleaned and prev_dataset not in relaxed):
                logging.info(f'{dataset_name}: Mesh already relaxed.')
                solved_maps[dataset_name] = None  # Loaded from file if needed
                prev_dataset = dataset_name
                continue

            config = ds_config['mesh_config'].copy()
            mesh_config = {
                'stride': (stride, stride),
                'dt': config.get('dt', 0.001), 
                'gamma': config.get('gamma', 0.5), 
                'k0': config.get('k0', 0.01), 
                'k': config.get('k', 0.1), 
                'num_iters': config.get('num_iters', 1000),
                'max_iters': config.get('max_iters', 100000), 
                'stop_v_max': config.get('stop_v_max', 0.005), 
                'dt_max': config.get('dt_max', 1000), 
                'start_cap': config.get('start_cap', 0.01),
                'final_cap': config.get('final_cap', 10), 
                'prefer_orig_order': config.get('prefer_orig_order', True)
                }
            mesh_config = IntegrationConfig(**mesh_config)

            # Final flow is stored as [z, c, y, x], sofima expects [c, z, y, x]
            final_flow_path = os.path.join(zarr_path, 'z_intermediate', 'flow_final', dataset_name)
            flow = open_store(final_flow_path, mode='r').read().result()
            flow = np.transpose(flow, [1, 0, 2, 3])

            # Origin of the flow grid in the output frame, in grid units
            flow_attrs = get_store_attributes(
                open_store(os.path.join(zarr_path, 'z_intermediate', 'flow1x', dataset_name), mode='r'))
            start = (-flow_attrs['shift_y'] / stride, -flow_attrs['shift_x'] / stride)

            if prev_dataset is None:
                # Root of the path, starts from the identity
                init_map, init_start = None, (0., 0.)
            else:
                # Start from the state of the anchor slice of the previous stack,
                # the same slice compute_flow_stack used as reference
                prev_solved, init_start = _get_solved(prev_dataset)
                prev_path = dataset_configs[prev_dataset]['dataset_path']
                _, z_anchor = find_ref_slice(open_store(prev_path, mode='r'), None, reverse=True)
                if z_anchor >= prev_solved.shape[1]:
                    raise RuntimeError(f'{dataset_name}: Anchor slice {z_anchor} of {prev_dataset} '
                                       f'has no mesh (last slice: {prev_solved.shape[1] - 1})')
                init_map = prev_solved[:, z_anchor:z_anchor + 1]

            inv_map, solved, _ = get_inv_map(flow, stride, mesh_config,
                                             init_map=init_map, start=start, init_start=init_start)
            solved_maps[dataset_name] = (solved, start)

            # Write solved maps as [c, z, y, x], the starting state of the next stacks of the path
            solved_store = open_store(
                os.path.join(zarr_path, 'z_intermediate', 'mesh_solved', dataset_name),
                mode='w', dtype=ts.float32,
                shape=list(solved.shape), chunks=[2, 1, 512, 512], axis_labels=['c', 'z', 'y', 'x'],
                fill_value=np.nan)
            solved_store.write(np.asarray(solved, np.float32)).result()
            set_store_attributes(solved_store, {'stride': stride, 'start': list(start)})

            # Write inverse map as [c, z, y, x], used for rendering
            inv_map_store = open_store(
                os.path.join(zarr_path, 'z_intermediate', 'inverse_map', dataset_name),
                mode='w', dtype=ts.float32,
                shape=list(inv_map.shape), chunks=[2, 1, 512, 512], axis_labels=['c', 'z', 'y', 'x'],
                fill_value=np.nan)
            inv_map_store.write(np.asarray(inv_map, np.float32)).result()
            set_store_attributes(
                inv_map_store, 
                {
                    'stride': stride, 
                    'start': list(start),
                    'canvas_shape': flow_attrs['canvas_shape'],
                    'shift_y': flow_attrs['shift_y'],
                    'shift_x': flow_attrs['shift_x']
                    })

            log_progress(db, dataset_name, STEPS['mesh'], None, None, {'skipped': False})
            relaxed.add(dataset_name)
            prev_dataset = dataset_name

    logging.info('Done!')


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
                        help='Wipe all flow, clean-up and mesh progress and restart')
    parser.add_argument('--wipe-progress',
                        dest='wipe_progress_stacks',
                        type=str,
                        nargs='+',
                        default=[],
                        help='Wipe progress for one or more specific stack(s) before starting. '
                             'Only the flow step unless --wipe-step is given')
    parser.add_argument('--wipe-step',
                        dest='wipe_steps',
                        type=str,
                        nargs='+',
                        choices=list(STEPS),
                        default=[],
                        help='Wipe progress for one or more step(s) before starting, for all stacks '
                             'or only those given with --wipe-progress. Later steps are wiped too')
    return parser

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Compute optical flow using pre-generated configuration files.\n'
                    'Configuration files should be created using prep_config_z, '
                    'and transforms computed with compute_transforms first.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    args = add_parser_arguments(parser).parse_args()

    compute_dataset_flow(
        project_dir=args.project_dir,
        num_workers=args.num_workers,
        start_over=args.start_over,
        wipe_progress_stacks=args.wipe_progress_stacks,
        wipe_steps=args.wipe_steps
    )
