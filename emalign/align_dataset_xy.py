import os

# # To prevent running out of memory because of preallocation
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# # Influences performance
# os.environ['OMP_NUM_THREADS'] = '4'
# os.environ['MKL_NUM_THREADS'] = '4'

import warnings
# Prevent printing the following warning, which does not seem to be an issue for the code to run properly:
#     [...]python3.12/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. 
#     os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
warnings.filterwarnings("ignore", category=RuntimeWarning, message="os.fork() was called")

import argparse
import json
import logging
import sys
from tqdm import tqdm

from emalign.arrays.stacks import parse_stack_info
from emalign.align_xy.render import resolve_img_q_fun
from emalign.scripts.align_stack_xy import align_stack_xy


logging.basicConfig(level=logging.INFO)
logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('jax._src.xla_bridge').setLevel(logging.WARNING)

# Constants
NUM_WORKERS = 1

def align_dataset_xy(project_dir,
                     num_workers=NUM_WORKERS,
                     overwrite=False,
                     start_over=False,
                     wipe_progress_stacks=None):
    '''Align and stitch in XY consecutive image stacks defined by a configuration file.

    Image stacks will be aligned one by one based on paths and parameters defined in a configuration file.
    Stacks will be skipped if they already exist. 
    If there are no images to align (i.e. only one tile in the stack), the image will just be written to zarr.

    Args:
        project_dir (str): Project directory containing the configurations created with prep_config_xy.
        num_workers (int): Number of threads to use for multiprocessing when relevant.
        overwrite (bool): Whether to overwrite dataset. If True, will delete existing dataset and start over. If False, will check for progress and skip processed slices. Defaults to False.
        start_over (bool): Whether to wipe the progress of all stacks and process everything again. Defaults to False.
        wipe_progress_stacks (list of str, optional): Names of the stacks to wipe progress for. Defaults to None.
    '''

    config_path = os.path.join(project_dir, 'config/xy_config/main_config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Configuration file does not exist: {config_path}\nDid you run prep_config_xy?')

    logging.info(f'Loading configuration from: {config_path}')
    with open(config_path, 'r') as f:
        main_config = json.load(f)

    project_name = main_config.get('project_name')
    if not project_name:
        project_name = os.path.basename(main_config['output_path']).rstrip('.zarr')
    mongodb_config_filepath = main_config.get('mongodb_config_filepath')

    main_dir        = main_config['input_dirs']
    output_path     = main_config['output_path']
    resolution      = main_config['resolution']
    offset          = main_config['offset']
    stride          = main_config['stride']
    apply_gaussian  = main_config['apply_gaussian']
    apply_clahe     = main_config['apply_clahe']
    stack_configs   = main_config['stack_configs']
    io_mode         = main_config['io_mode']
    # Optional: which tile is rendered on top. See resolve_img_q_fun for accepted values.
    img_q_fun       = resolve_img_q_fun(main_config.get('img_on_top', 'laplacian'))
    # Optional: minimum acceptable stitch score (0 to 1) for a slice to be written.
    min_stitch_score = main_config.get('min_stitch_score', 0.8)

    if not output_path.endswith('.zarr'):
        raise RuntimeError('Output path must be a zarr container (.zarr)')

    # Handle start_over
    if start_over:
        try:
            input('WARNING: All XY progress will be wiped and all stacks will be processed.\n'
                  'Press ENTER to continue or CTRL+C to abort\n')
        except KeyboardInterrupt:
            logging.info('\nAborted by user')
            sys.exit(0)
        wipe_progress_stacks = list(stack_configs)

    # Progress is wiped by align_stack_xy for the requested stacks
    wipe_progress_stacks = [s for s in (wipe_progress_stacks or []) if s]
    unknown = [s for s in wipe_progress_stacks if s not in stack_configs]
    if unknown:
        raise RuntimeError(f'No configuration found for stack(s): {unknown}')

    # Find tilesets with wanted resolution
    logging.info(f'Tilesets found in:\n   {main_dir}')
    logging.info(f'Destination:\n   {output_path}')
    logging.info(f' - Resolution: {resolution}')
    logging.info(f' - Apply gaussian: {apply_gaussian}')
    logging.info(f' - Apply CLAHE: {apply_clahe}\n')
    logging.info(f'Aligning {len(stack_configs)} tilesets, including {main_config.get("tilesets_combined", 0)} combined.')
    for s in stack_configs.keys():
        logging.info(f'    {s}')

    for stack_name, stack_config_path in tqdm(stack_configs.items(), 
                                                total=len(stack_configs), 
                                                position=1, 
                                                desc='Processing stacks', 
                                                leave=True):
        tile_maps_paths, tile_maps_invert, ignore_slices = parse_stack_info(stack_config_path)
        wipe_this_stack = (stack_name in wipe_progress_stacks)
        align_stack_xy(output_path=output_path,
                       stack_name=stack_name,
                       tile_maps_paths=tile_maps_paths,
                       tile_maps_invert=tile_maps_invert,
                       resolution=resolution,
                       offset=offset,
                       stride=stride,
                       apply_gaussian=apply_gaussian,
                       apply_clahe=apply_clahe,
                       project_name=project_name,
                       io_mode=io_mode,
                       ignore_slices=ignore_slices,
                       mongodb_config_filepath=mongodb_config_filepath,
                       num_cores=num_workers,
                       overwrite=overwrite,
                       wipe_progress_flag=wipe_this_stack,
                       img_q_fun=img_q_fun,
                       min_stitch_score=min_stitch_score)
    logging.info(f'Done! Output can be found at: {output_path}')
    

def add_parser_arguments(parser):
    
    # Required arguments
    parser.add_argument('-p', '--project-dir',
                        metavar='PROJECT_DIR',
                        dest='project_dir',
                        required=True,
                        type=str,
                        help='Project directory containing the configurations created with prep_config_xy.')

    # Optional arguments
    parser.add_argument('-c', '--cores',
                        metavar='CORES',
                        dest='num_workers',
                        type=int,
                        default=NUM_WORKERS,
                        help=f'Number of threads to use. Default: {NUM_WORKERS}')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing dataset.')
    parser.add_argument('--start-over',
                        dest='start_over',
                        default=False,
                        action='store_true',
                        help='Wipe all progress and restart')
    parser.add_argument('--wipe-progress',
                        dest='wipe_progress_stacks',
                        type=str,
                        nargs='+',
                        default=[],
                        help='Wipe progress for one or more specific stack(s) before starting')
    return parser


if __name__ == '__main__':


    parser=argparse.ArgumentParser('Script aligning tiles in XY based on SOFIMA (Scalable Optical Flow-based Image Montaging and Alignment). \n\
                                    This script was written to match the file structure produced by the ThermoFisher MAPs software.')

    args = add_parser_arguments(parser).parse_args()


    try:
        GPU_ids = os.environ['CUDA_VISIBLE_DEVICES']
    except Exception:
        print('To select GPUs, specify it before running python, e.g.: CUDA_VISIBLE_DEVICES=0,1 python script.py')
        sys.exit()
    print(f'Available GPU IDs: {GPU_ids}\n')

    align_dataset_xy(**vars(args))
