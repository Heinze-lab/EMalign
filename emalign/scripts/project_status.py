import argparse
import json
import os

from emalign.align_z.config import load_align_plan, load_dataset_configs, load_fuse_plan
from emalign.io.progress import get_mongo_client, get_mongo_db
from emalign.io.store import open_store, get_store_attributes
from emalign.io.utils import get_zarr_root
from emalign.scripts.fuse_stacks_xy import fused_stack_name


DONE = u'[\u2713]' # Tick mark
TODO = '[ ]'
NONE = '[-]'


def _format_slices(slices, max_shown=8):
    '''Compact a list of slice indices into ranges: 1-4, 7, 9-12, ...'''
    slices = sorted(slices)
    ranges = []
    for z in slices:
        if ranges and z == ranges[-1][1] + 1:
            ranges[-1][1] = z
        else:
            ranges.append([z, z])
    text = [f'{a}' if a == b else f'{a}-{b}' for a, b in ranges[:max_shown]]
    if len(ranges) > max_shown:
        text.append(f'... ({len(ranges) - max_shown} more ranges)')
    return ', '.join(text)


def _slices_done(db, dataset_name, step_name, doc_filter=None):
    '''Local slice indices logged for a step.'''
    query = {'step_name': step_name, **(doc_filter or {})}
    return set(db[dataset_name].distinct('local_slice', query))


def _step_done(db, dataset_name, step_name):
    '''Whether a step logged once per dataset (local_slice None) is done.'''
    return db[dataset_name].count_documents({'step_name': step_name, 'local_slice': None}) > 0


def _print_step(title, missing, total, unit='datasets', note=None):
    '''Print a step and the datasets that are not done.

    missing: dict of dataset_name -> description of what is left
    '''
    if total == 0:
        print(f'{NONE} {title}: nothing to do' + (f' ({note})' if note else ''))
        return
    if not missing:
        print(f'{DONE} {title}: finished ({total} {unit})')
        return
    print(f'{TODO} {title}: {total - len(missing)}/{total} {unit} done')
    for name, left in missing.items():
        print(f'        - {name}: {left}')


def _check_slices(db, names_to_expected, step_name, doc_filter=None):
    '''Datasets whose expected slices are not all logged, with a description of what is left.'''
    missing = {}
    for name, expected in names_to_expected.items():
        left = set(expected) - _slices_done(db, name, step_name, doc_filter)
        if left:
            n = len(expected)
            missing[name] = f'{n - len(left)}/{n} slices, missing: {_format_slices(left)}'
    return missing


def project_status(project_dir, return_next_step=False):

    next_step = None

    project_dir = os.path.abspath(project_dir)
    xy_config_path = os.path.join(project_dir, 'config', 'xy_config', 'main_config.json')
    z_config_dir = os.path.join(project_dir, 'config', 'z_config')

    if not os.path.exists(xy_config_path):
        if return_next_step:
            return 'xy_prep'
        raise FileNotFoundError(f'XY configuration not found: {xy_config_path}\nDid you run prep_config_xy?')
    with open(xy_config_path, 'r') as f:
        main_config = json.load(f)

    project_name = main_config['project_name']
    db = get_mongo_db(get_mongo_client(main_config.get('mongodb_config_filepath')), project_name)

    print(f'Project: {project_name}')
    print(f'Project directory: {project_dir}')
    print()

    # ---------- XY alignment ----------
    # Local slice index is the slice number minus the first slice of the stack
    expected = {}
    for stack_name, stack_config_path in main_config['stack_configs'].items():
        with open(stack_config_path, 'r') as f:
            slices = sorted(int(z) for z in json.load(f)['tile_maps'])
        expected[stack_name] = [z - slices[0] for z in slices] if slices else []
    missing = _check_slices(db, expected, 'align_xy')
    _print_step('XY alignment', missing, len(expected), 'stacks')
    if len(missing) > 0 and next_step is None:
        next_step = 'xy_alignment'
        if return_next_step:
            return next_step

    # ---------- Fuse XY stacks (optional) ----------
    _, group_configs = load_fuse_plan(project_dir)
    if group_configs is None:
        _print_step('Fuse XY stacks', {}, 0, note='no fuse plan: not run yet, or no stacks overlap')
        if next_step is None:
            next_step = 'fuse_step'

    else:
        missing = _check_slices(db, expected, 'fuse_xy')
        expected = {fused_stack_name(c): range(c['zmax'] - c['zmin']) for c in group_configs}
        _print_step('Fuse XY stacks', missing, len(expected), 'groups')
        if len(missing) > 0 and next_step is None:
            next_step = 'fuse_step'
    
    if next_step is not None and return_next_step:
        return next_step

    # ---------- Z alignment ----------
    z_steps = ['Z transforms', 'Z transform chaining', 'Z flow', 'Z flow clean-up',
               'Z mesh relaxation', 'Z rendering']
    if not os.path.exists(os.path.join(z_config_dir, '00_align_plan.json')):
        if return_next_step:
            return 'z_prep'
        for title in z_steps:
            print(f'{TODO} {title}: not configured, run prep_config_z')
        print('python prep_config_z.py -p {project_dir} ...')
        return

    align_plan = load_align_plan(z_config_dir)
    dataset_configs = load_dataset_configs(z_config_dir)
    if db.name != f'alignment_{align_plan["project_name"]}':
        # Z configs may log to their own database
        first_config = next(iter(dataset_configs.values()))
        db = get_mongo_db(get_mongo_client(first_config.get('mongodb_config_filepath')),
                          align_plan['project_name'])

    # Only stacks that are part of an alignment path are aligned along Z
    names = sorted({name for path in align_plan['paths'] for name in path})
    local_ranges = {name: range(dataset_configs[name]['local_z_min'], dataset_configs[name]['local_z_max'])
                    for name in names}
    zarr_path = get_zarr_root(align_plan['destination_path'])

    # Transforms, per slice
    missing = _check_slices(db, local_ranges, 'transform_z')
    _print_step('Z transforms', missing, len(names))
    if len(missing) > 0 and next_step is None:
        next_step = 'z_transforms'

    # Chaining, final transforms are flagged in the store attributes
    missing = {}
    for name in names:
        path = os.path.join(zarr_path, 'z_intermediate', 'transform_chained', name)
        store = open_store(path, mode='r', allow_missing=True)
        attrs = get_store_attributes(store) if store is not None else None
        if not attrs or not attrs.get('final', False):
            missing[name] = 'final transforms not written'
    _print_step('Z transform chaining', missing, len(names))
    if len(missing) > 0 and next_step is None:
        next_step = 'z_transforms'

    # Flow, per slice and per scale: downsampled and full scale
    missing = {}
    for name in names:
        left = []
        for scale in sorted({dataset_configs[name]['scale'], 1}):
            m = _check_slices(db, {name: local_ranges[name]}, 'flow_z', {'scale': scale})
            if m:
                left.append(f'scale {scale}: {m[name]}')
        if left:
            missing[name] = ' | '.join(left)
    _print_step('Z flow', missing, len(names))
    if len(missing) > 0 and next_step is None:
        next_step = 'z_flow'

    # Clean-up and mesh, once per dataset
    for title, step_name in [('Z flow clean-up', 'flow_clean_z'), ('Z mesh relaxation', 'mesh_relax_z')]:
        missing = {name: 'not done' for name in names if not _step_done(db, name, step_name)}
        _print_step(title, missing, len(names))
        if len(missing) > 0 and next_step is None:
            next_step = 'z_flow'

    # Rendering, per slice
    missing = _check_slices(db, local_ranges, 'render_z')
    _print_step('Z rendering', missing, len(names))
    if len(missing) > 0 and next_step is None:
        next_step = 'z_render'
    
    if return_next_step:
        return next_step
        
    print()
    if next_step == 'xy_alignment':
        print('Next step to run: XY alignment')
        print(f'python run_xy_alignment.py -p {project_dir} ...')
    elif next_step == 'fuse_step':
        print('Next step to run: fusing step')
        print(f'python fuse_stacks_xy.py -p {project_dir} ...')
    elif next_step == 'z_transforms':
        print('Next step to run: Z transforms')
        print(f'python scripts/compute_transforms.py -p {project_dir} ...')
    elif next_step == 'z_flow':
        print('Next step to run: Z flow computation')
        print(f'python scripts/compute_flow.py -p {project_dir} ...')
    elif next_step == 'z_rendering':
        print('Next step to run: image rendering')
        print(f'python scripts/render_images.py -p {project_dir} ...')

    return next_step

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Print the to-do list of a project, from XY alignment to final rendering.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-p', '--project-dir',
                        metavar='PROJECT_DIR',
                        dest='project_dir',
                        required=True,
                        type=str,
                        help='Project directory containing the config directory.')

    args = parser.parse_args()
    project_status(args.project_dir)
