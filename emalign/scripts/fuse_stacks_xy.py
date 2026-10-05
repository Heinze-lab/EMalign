'''Fuse XY aligned stacks that overlap on the same Z slices.

After XY alignment, several stacks may cover the same Z slices. Z alignment expects one
image per slice, so overlapping stacks are stitched together into a single fused stack and only
that fused stack is used from then on. Stacks that cannot be matched are left alone: they likely 
belong to different alignment paths, which Z alignment resolves.

Configuration files are generated on the first run into project_dir/config/fuse_config/ and reused
afterwards, because detecting overlaps runs SIFT over full resolution slices.

Usage:
    CUDA_VISIBLE_DEVICES=0 python emalign.scripts.fuse_stacks_xy \\
        -p /path/to/project_dir \\
        -c 4
'''

import os
import warnings
# Prevent printing the following warning, which does not seem to be an issue for the code to run properly:
#     [...]python3.12/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called.
#     os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
warnings.filterwarnings('ignore', category=RuntimeWarning, message='os.fork() was called')

import argparse
import json
import logging
import numpy as np
import sys
import tensorstore as ts

from glob import glob
from tqdm import tqdm

from emalign.align_xy.prep import create_configs_fused_stacks
from emalign.align_xy.render import resolve_img_q_fun
from emalign.align_xy.stitch_offgrid import stitch_images
from emalign.align_z.config import add_config_metadata, get_fuse_config_dir, load_fuse_plan
from emalign.arrays.utils import pad_to_shape, resample
from emalign.io.process.mask import compute_greyscale_mask
from emalign.io.progress import get_mongo_client, get_mongo_db, log_progress, check_progress, wipe_progress
from emalign.io.store import get_store_attributes, open_store, set_store_attributes, write_data


logging.basicConfig(level=logging.INFO)
logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('jax._src.xla_bridge').setLevel(logging.WARNING)

# Constants
NUM_WORKERS = 1
SCALE = 0.1                         # Downsampling applied before estimating the offset with SIFT
PATCH_SIZE = 160                    # Patch size used to compute the flow map
STRIDE = 40                         # Stride used to compute the flow map
K0 = 0.01                           # Mesh relaxation parameters
K = 0.1
GAMMA = 0.5
CHUNK_SIZE = [1, 512, 512]          # For store creation
STEP_NAME = 'fuse_xy'
IMG_ON_TOP_VALUES = ('auto', '1', '2')  # Accepted by stitch_images
DEFAULT_IMG_Q_FUN = 'sharpness'     # See resolve_img_q_fun


def fused_from(config):
    '''Names of the stacks a group is built from.

    Args:
        config (dict): Configuration dictionary of a group of stacks to fuse.

    Returns:
        list of `str`: Names of the stacks, in the order they are fused.
    '''
    return [os.path.basename(os.path.abspath(path)) for path in config['dataset_paths']]


def fused_stack_name(config):
    '''Name of the stack resulting from fusing a group.

    Args:
        config (dict): Configuration dictionary of a group of stacks to fuse.

    Returns:
        str: Name of the fused stack.
    '''
    return '_'.join(fused_from(config)) + '_fused'


def prep_config_fuse(project_dir,
                     main_config_path,
                     scale=SCALE,
                     force_overwrite=False):
    '''Gather or compute the configuration files of the groups of stacks to fuse.

    Detecting which stacks overlap runs SIFT over full resolution slices, so the resulting
    configuration files are cached in project_dir/config/fuse_config/ and reused on subsequent runs.

    Args:
        project_dir (str): Directory containing the project: config directory, and output zarr.
        main_config_path (str): Absolute path to the main_config.json file of this project.
        scale (float, optional): Scale to downsample images to when looking for overlaps with SIFT.
            Defaults to SCALE.
        force_overwrite (bool, optional): Whether to recompute the configuration files even if they
            already exist. Defaults to False.

    Returns:
        list of `dict`: One configuration dictionary per group of stacks to fuse.
    '''
    config_dir = get_fuse_config_dir(project_dir)

    if not force_overwrite:
        _, group_configs = load_fuse_plan(project_dir)
        if group_configs is not None:
            logging.info(f'Loaded {len(group_configs)} group(s) of overlapping stacks from {config_dir}')
            return group_configs

    with open(main_config_path, 'r') as f:
        main_config = json.load(f)

    project_name = main_config.get('project_name')
    if not project_name:
        project_name = os.path.basename(main_config['output_path']).rstrip('.zarr')

    logging.info('Looking for stacks overlapping on the same Z slices...')
    group_configs = create_configs_fused_stacks(main_config_path, scale=scale)

    os.makedirs(config_dir, exist_ok=True)

    # Drop stale group configs so that regenerating never leaves orphans behind
    for stale_path in glob(os.path.join(config_dir, 'fuse_*.json')):
        os.remove(stale_path)

    filenames = []
    for i, config in enumerate(group_configs):
        zmin = config['zmin']
        zmax = config['zmax']

        config['fused_from'] = fused_from(config)
        config['destination_name'] = fused_stack_name(config)
        config = add_config_metadata(config)
        group_configs[i] = config

        filename = f'fuse_{zmin}_{zmax}_{i}.json'
        with open(os.path.join(config_dir, filename), 'w') as f:
            json.dump(config, f, indent=2)
        filenames.append(filename)

    plan = {
        'project_name': project_name,
        'output_path': main_config['output_path'],
        'scale': scale,
        'group_configs': filenames,
        'fused_stacks': [config['destination_name'] for config in group_configs]
    }
    plan = add_config_metadata(plan)

    with open(os.path.join(config_dir, '00_fuse_plan.json'), 'w') as f:
        json.dump(plan, f, indent=2)

    logging.info(f'Configuration files were created at {config_dir}')
    return group_configs


def fuse_stacks_group(config,
                      project_name,
                      target_res,
                      mongodb_config_filepath=None,
                      scale=SCALE,
                      patch_size=PATCH_SIZE,
                      stride=STRIDE,
                      img_on_top='auto',
                      img_q_fun=None,
                      destination_path=None,
                      overwrite=False,
                      wipe_progress_flag=False,
                      num_workers=NUM_WORKERS):
    '''Fuse a group of stacks that overlap on the XY plane.

    Slices are fused one by one and written to a new stack next to the stacks they come from.
    When two images cannot be matched, the best of them is written on its own so that the fused
    stack still holds exactly one image per slice, and the slice is reported at the end.

    Args:
        config (dict): Configuration dictionary of the group, as written by prep_config_fuse.
        project_name (str): Name of the project.
        target_res (int): Target YX resolution in nanometers. Stacks acquired at another resolution
            are resampled to it.
        mongodb_config_filepath (str, optional): Path to the MongoDB configuration file. Defaults to None.
        scale (float, optional): Scale to downsample images to when determining the offset with SIFT.
            Defaults to SCALE.
        patch_size (int, optional): Patch size used to compute the flow map with
            `sofima.flow_field.JAXMaskedXCorrWithStatsCalculator`. Defaults to PATCH_SIZE.
        stride (int, optional): Stride used to compute the flow map with
            `sofima.flow_field.JAXMaskedXCorrWithStatsCalculator`. Defaults to STRIDE.
        img_on_top (str, optional): Which image is rendered on top. One of IMG_ON_TOP_VALUES.
            Defaults to 'auto'.
        img_q_fun (callable, optional): Required when img_on_top is 'auto'. Takes an image and its
            mask, returns a value that is higher for higher quality/sharpness. Defaults to None.
        destination_path (str, optional): Path of the fused stack. Defaults to None, in which case it
            is written next to the stacks it is built from.
        overwrite (bool, optional): Whether to delete the destination and start over. Defaults to False.
        wipe_progress_flag (bool, optional): Whether to wipe progress for this stack. Defaults to False.
        num_workers (int, optional): Number of threads used to render the fused image by
            `sofima.warp.ndimage_warp`. Defaults to NUM_WORKERS.

    Returns:
        bool: True if the stack was processed, False if it was skipped.
    '''
    if img_on_top not in IMG_ON_TOP_VALUES:
        raise ValueError(f'img_on_top must be one of {IMG_ON_TOP_VALUES}, got {img_on_top!r}')
    if img_on_top == 'auto' and img_q_fun is None:
        raise ValueError('img_on_top is set to "auto". Please provide img_q_fun.')

    destination_name = config.get('destination_name') or fused_stack_name(config)

    client = get_mongo_client(mongodb_config_filepath)
    db = get_mongo_db(client, project_name)

    if wipe_progress_flag:
        logging.info(f'Wiping progress for stack: {destination_name}')
        wipe_progress(db, destination_name)

        # Since we wipe progress, we also want to make sure that all old data will be overwritten properly
        overwrite = True

    #---------- Prepare variables ----------#
    source_paths = [os.path.abspath(path) for path in config['dataset_paths']]

    if destination_path is None:
        destination_basepath = os.path.dirname(source_paths[0])
        destination_path = os.path.join(destination_basepath, destination_name)
    else:
        destination_path = os.path.abspath(destination_path)
        destination_basepath = os.path.dirname(destination_path)
    destination_mask_path = os.path.join(destination_basepath, destination_name + '_mask')
    attrs_path = os.path.join(destination_path, '.zattrs')

    z_shape = config['zmax'] - config['zmin']

    # Skip if already fully processed
    if os.path.exists(attrs_path) and not overwrite:
        logging.info(f'Skipping {destination_name} because it was already processed.')
        return False

    if overwrite:
        logging.warning(f'{destination_name}: existing dataset will be deleted and fused from scratch.')

    #---------- Open the stacks to fuse ----------#
    stacks = []
    for z_offset, dataset_path in zip(config['z_offsets'], source_paths):
        # Limit to the overlapping range only
        zmin = config['zmin'] - z_offset
        zmax = config['zmax'] - z_offset

        dataset = open_store(dataset_path, mode='r')[zmin:zmax]

        # Stacks may have been acquired at another resolution than the target one
        target_scale = get_store_attributes(dataset_path)['resolution'][-1] / target_res

        # Masks are optional, they are computed on the fly when they do not exist
        dataset_mask = open_store(dataset_path + '_mask', mode='r', dtype=ts.bool, allow_missing=True)
        if dataset_mask is not None:
            dataset_mask = dataset_mask[zmin:zmax]

        stacks.append({'name': os.path.basename(dataset_path),
                       'dataset': dataset,
                       'dataset_mask': dataset_mask,
                       'target_scale': target_scale,
                       'zmin': zmin})

    #---------- Open destinations ----------#
    # The image and its mask are always created together: a lone one cannot be resumed from
    stores_exist = os.path.exists(destination_path) and os.path.exists(destination_mask_path)
    start_from_scratch = overwrite or not stores_exist

    if start_from_scratch:
        destination = open_store(
            destination_path,
            mode='w',
            dtype=ts.uint8,
            shape=[z_shape, 1, 1],
            chunks=CHUNK_SIZE
        )

        destination_mask = open_store(
            destination_mask_path,
            mode='w',
            dtype=ts.bool,
            shape=[z_shape, 1, 1],
            chunks=CHUNK_SIZE
        )
    else:
        destination = open_store(destination_path, mode='r+', dtype=ts.uint8)
        destination_mask = open_store(destination_mask_path, mode='r+', dtype=ts.bool)

    #---------- Fuse slices ----------#
    # Check what is to be processed
    if start_from_scratch:
        slices_to_process = list(range(z_shape))
    else:
        slices_to_process = [z for z in range(z_shape)
                             if not check_progress(db, destination_name, STEP_NAME, z)]

    n_skip = z_shape - len(slices_to_process)
    if n_skip:
        logging.info(f'{destination_name}: Skipping {n_skip} already-processed slices')

    n_fused = 0
    n_single = 0
    unfused_slices = []
    empty_slices = []

    pbar = tqdm(slices_to_process, position=2, desc=f'{destination_name}: Fusing',
                dynamic_ncols=True, leave=False)
    for z in pbar:
        global_slice_index = z + config['zmin']

        canvas = None
        canvas_mask = None
        candidates = []     # Images of that slice, only used if none of them can be fused
        n_stitched = 0
        errors = []

        for stack in stacks:
            pbar.set_description(f'{destination_name}: Loading slice {global_slice_index}...')
            img = stack['dataset'][z + stack['zmin']].read().result()
            if not img.any():
                continue

            if stack['dataset_mask'] is None:
                mask = compute_greyscale_mask(img)
            else:
                mask = stack['dataset_mask'][z + stack['zmin']].read().result()

            # Resample to the target resolution
            img = resample(img, stack['target_scale'])
            mask = _match_shape(resample(mask, stack['target_scale']), img.shape)

            candidates.append((stack['name'], img, mask))

            if canvas is None:
                # First image, there is nothing to fuse it with yet
                canvas = img
                canvas_mask = mask
                continue

            pbar.set_description(f'{destination_name}: Fusing slice {global_slice_index}...')
            try:
                canvas, canvas_mask = stitch_images(canvas,
                                                    img,
                                                    mask1=canvas_mask,
                                                    mask2=mask,
                                                    scale=scale,
                                                    patch_size=patch_size,
                                                    stride=stride,
                                                    parallelism=num_workers,
                                                    img_on_top=img_on_top,
                                                    img_q_fun=img_q_fun,
                                                    k0=K0,
                                                    k=K,
                                                    gamma=GAMMA)
                n_stitched += 1
            except Exception as e:
                # Images that cannot be matched fail from inside SIFT/SOFIMA rather than reporting
                # it, typically as a TypeError on a transform that was never estimated, or an
                # IndexError on an empty overlap. A single bad slice must not abort the whole stack.
                errors.append(f'{stack["name"]}: {type(e).__name__}: {e}')
                logging.warning(f'{destination_name}: could not fuse {stack["name"]} at z = '
                                f'{global_slice_index}, images may not match ({type(e).__name__}: {e})')

        if canvas is None:
            empty_slices.append(global_slice_index)
        elif len(candidates) == 1:
            # Only one stack covers that slice, there is nothing to fuse
            n_single += 1
        elif n_stitched == 0:
            # Nothing could be fused on that slice: keep the best image on its own so that Z
            # alignment still receives exactly one image for it.
            name, canvas, canvas_mask = _best_image(candidates, img_q_fun)
            unfused_slices.append(global_slice_index)
            logging.warning(f'{destination_name}: z = {global_slice_index} written unfused, keeping {name}')
        else:
            n_fused += 1

        if canvas is not None:
            pbar.set_description(f'{destination_name}: Writing slice {global_slice_index}...')
            destination, _ = write_data(destination, canvas, z)
            destination_mask, _ = write_data(destination_mask, canvas_mask, z)

        # Log progress
        metadata = {
            'mesh_parameters':{
                            'stride':stride,
                            'patch_size':patch_size,
                            'k0':K0,
                            'k':K,
                            'gamma':GAMMA
                            },
            'empty_slice': canvas is None,
            'scale': scale,
            'img_on_top': img_on_top,
            'n_images': len(candidates),
            'fused': n_stitched > 0
                }
        if errors:
            metadata['fusion_errors'] = errors
        log_progress(db, destination_name, STEP_NAME, global_slice_index, z, metadata)

    pbar.set_description(f'{destination_name}: done')

    if slices_to_process:
        logging.info(f'{destination_name}: Fused {n_fused}/{len(slices_to_process)} slices.')
        if n_single:
            logging.info(f'{destination_name}: {n_single} slice(s) were covered by a single stack.')
        if unfused_slices:
            logging.warning(f'{destination_name}: {len(unfused_slices)} slice(s) written unfused: '
                            f'{_format_slices(unfused_slices)}')
        if empty_slices:
            logging.info(f'{destination_name}: {len(empty_slices)} empty slice(s): '
                         f'{_format_slices(empty_slices)}')

    #---------- Write attributes ----------#
    # The fused stack takes the attributes of the stacks it comes from, at the target resolution.
    # fused_from records which stacks it supersedes, so that prep_config_z ignores them.
    source_attributes = get_store_attributes(source_paths[0])
    resolution = [source_attributes['resolution'][0], target_res, target_res]
    voxel_offset = [config['zmin'], *source_attributes['voxel_offset'][1:]]

    attributes = {'voxel_offset': list(map(int, voxel_offset)),
                  'offset': list(map(int, np.array(voxel_offset) * np.array(resolution))),
                  'resolution': list(map(int, resolution)),
                  'voxel_size': list(map(int, resolution)),
                  'fused_from': config.get('fused_from') or fused_from(config)}

    set_store_attributes(destination, attributes)
    set_store_attributes(destination_mask, attributes)

    return True


def fuse_dataset_xy(project_dir,
                    config_path=None,
                    num_workers=NUM_WORKERS,
                    scale=SCALE,
                    patch_size=PATCH_SIZE,
                    stride=STRIDE,
                    img_on_top='auto',
                    overwrite=False,
                    force_overwrite=False,
                    wipe_progress_stacks=None):
    '''Fuse every group of overlapping stacks of a project, one after the other.

    Args:
        project_dir (str): Directory containing the project: config directory, and output zarr.
        config_path (str, optional): Path to the XY main config file. Defaults to None, in which case
            it is assumed to exist at "project_dir/config/xy_config/main_config.json".
        num_workers (int, optional): Number of threads to use for rendering. Defaults to NUM_WORKERS.
        scale (float, optional): Scale to downsample images to when determining the offset with SIFT.
            Defaults to SCALE.
        patch_size (int, optional): Patch size used to compute the flow map. Defaults to PATCH_SIZE.
        stride (int, optional): Stride used to compute the flow map. Defaults to STRIDE.
        img_on_top (str, optional): Which image is rendered on top. One of IMG_ON_TOP_VALUES.
            Defaults to 'auto'.
        overwrite (bool, optional): Whether to delete existing fused stacks and start over.
            Defaults to False.
        force_overwrite (bool, optional): Whether to recompute the configuration files. Defaults to False.
        wipe_progress_stacks (list of `str`, optional): Names of the fused stacks to wipe progress for.
            Defaults to None.
    '''
    if wipe_progress_stacks is None:
        wipe_progress_stacks = []

    if config_path is None:
        # Attempt to find the config in the project directory
        config_path = os.path.join(project_dir, 'config/xy_config/main_config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f'Main config file not found in the project directory: {config_path}')
        logging.info(f'Config file location was determined from project directory:\n{config_path}\n')

    with open(config_path, 'r') as f:
        main_config = json.load(f)

    project_name = main_config.get('project_name')
    if not project_name:
        project_name = os.path.basename(main_config['output_path']).rstrip('.zarr')
    mongodb_config_filepath = main_config.get('mongodb_config_filepath')

    target_res = main_config['resolution'][-1]

    # Optional: metric deciding which image ends up on top. See resolve_img_q_fun for accepted values.
    img_q_fun = resolve_img_q_fun(main_config.get('img_on_top', DEFAULT_IMG_Q_FUN))
    if img_on_top == 'auto' and img_q_fun is None:
        logging.warning(f'"img_on_top" in the main config does not define a quality metric. '
                        f'Falling back to "{DEFAULT_IMG_Q_FUN}" to decide which image goes on top.')
        img_q_fun = resolve_img_q_fun(DEFAULT_IMG_Q_FUN)

    group_configs = prep_config_fuse(project_dir,
                                     config_path,
                                     scale=scale,
                                     force_overwrite=force_overwrite)

    if not group_configs:
        logging.info('No overlapping stacks were found: there is nothing to fuse.')
        return

    logging.info(f'Fusing {len(group_configs)} group(s) of overlapping stacks:')
    for config in group_configs:
        sources = ' + '.join(config.get('fused_from') or fused_from(config))
        logging.info(f'    z {config["zmin"]}-{config["zmax"]}: {sources} -> {config["destination_name"]}')
    logging.info(f'Target resolution (yx): {target_res}')
    logging.info(f'Number of cores used for rendering: {num_workers}\n')

    pbar = tqdm(group_configs, position=1, desc='Fusing stacks', dynamic_ncols=True, leave=True)
    for config in pbar:
        destination_name = config.get('destination_name') or fused_stack_name(config)
        pbar.set_description(f'{destination_name}: Processing group of stacks')

        fuse_stacks_group(config,
                          project_name=project_name,
                          target_res=target_res,
                          mongodb_config_filepath=mongodb_config_filepath,
                          scale=scale,
                          patch_size=patch_size,
                          stride=stride,
                          img_on_top=img_on_top,
                          img_q_fun=img_q_fun,
                          overwrite=overwrite,
                          wipe_progress_flag=(destination_name in wipe_progress_stacks),
                          num_workers=num_workers)

    logging.info(f'Done! All {len(group_configs)} group(s) of stacks were fused.')
    logging.info(f'\n\nTo prepare Z alignment:\npython emalign.prep_config_z -p {project_dir} -cfg-z <config_z.json>')


def _match_shape(array, shape):
    '''Pad or crop an array so that it matches a shape exactly.

    compute_greyscale_mask downsamples and upsamples internally, so a computed mask can come back a
    few pixels off from the image it describes.

    Args:
        array (np.ndarray): Array to reshape.
        shape (tuple): Target YX shape.

    Returns:
        np.ndarray: Array of the requested shape.
    '''
    array = pad_to_shape(array, shape)
    return array[:shape[0], :shape[1]]


def _best_image(candidates, img_q_fun):
    '''Pick the image to keep when a slice could not be fused.

    Only called on the slices that failed, because scoring every image of every slice would cost
    as much as the fusing itself.

    Args:
        candidates (list of `tuple`): Images of the slice, as (name, image, mask).
        img_q_fun (callable or None): Function taking an image and its mask, returning a value that
            is higher for higher quality/sharpness. Falls back to the largest mask when None.

    Returns:
        tuple: The best (name, image, mask).
    '''
    def quality(candidate):
        _, img, mask = candidate
        if img_q_fun is not None and mask.any():
            return img_q_fun(img, mask)
        return float(mask.sum())

    return max(candidates, key=quality)


def _format_slices(slices, max_shown=10):
    '''Format a list of slice indices for logging, truncating long ones.

    Args:
        slices (list of `int`): Slice indices.
        max_shown (int, optional): How many indices to show. Defaults to 10.

    Returns:
        str: Formatted list.
    '''
    if len(slices) <= max_shown:
        return str(slices)
    return f'{slices[:max_shown]}... (+{len(slices) - max_shown} more)'


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Fuse XY aligned stacks that overlap on the same Z slices, so that Z alignment\n'
                    'starts from a single image per slice. Run after align_dataset_xy.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument('-p', '--project-dir',
                        metavar='PROJECT_DIR',
                        dest='project_dir',
                        required=True,
                        type=str,
                        help='Directory containing the project: config directory, and output zarr.')

    # Optional arguments
    parser.add_argument('-cfg', '--config',
                        metavar='CONFIG_PATH',
                        dest='config_path',
                        type=str,
                        default=None,
                        help='Path to the XY main config file. Default: main config is assumed to exist at "project_dir/config/xy_config/main_config.json"')
    parser.add_argument('-c', '--cores',
                        metavar='CORES',
                        dest='num_workers',
                        type=int,
                        default=NUM_WORKERS,
                        help=f'Number of threads to use for rendering. Default: {NUM_WORKERS}')
    parser.add_argument('--overwrite',
                        dest='overwrite',
                        default=False,
                        action='store_true',
                        help='Delete existing fused stacks and fuse them again from scratch.')
    parser.add_argument('--force-overwrite',
                        dest='force_overwrite',
                        default=False,
                        action='store_true',
                        help='Look for overlapping stacks again instead of reusing the existing configuration files.')
    parser.add_argument('--wipe-progress',
                        dest='wipe_progress_stacks',
                        type=str,
                        nargs='+',
                        default=[''],
                        help='Wipe progress for one or more specific fused stack(s) before starting.')

    args = parser.parse_args()

    # Check GPU
    try:
        GPU_ids = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        logging.error('No GPUs specified. Set CUDA_VISIBLE_DEVICES environment variable.')
        logging.error('Example: CUDA_VISIBLE_DEVICES=0 python -m emalign.scripts.fuse_stacks_xy ...')
        sys.exit(1)
    logging.info(f'Using GPU IDs: {GPU_ids}')

    fuse_dataset_xy(**vars(args))
