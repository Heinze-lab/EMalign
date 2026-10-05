import argparse
import importlib
import logging
import os

# Step modules load JAX, so they are only imported once CUDA_VISIBLE_DEVICES is set (see __main__).
# Steps are given by name and resolved when needed.
STEP_TO_FUNC = {
    'xy_prep': None,
    'z_prep': None,
    'xy_alignment': 'emalign.align_dataset_xy.align_dataset_xy',
    'fuse_step': 'emalign.scripts.fuse_stacks_xy.fuse_dataset_xy',
    'z_transforms': 'emalign.scripts.compute_transforms.compute_dataset_transforms',
    'z_flow': 'emalign.scripts.compute_flow.compute_dataset_flow',
    'z_render': 'emalign.scripts.render_dataset.render_dataset'
}


def get_project_next_step(project_dir):
    from emalign.scripts.project_status import project_status

    next_step = project_status(project_dir, return_next_step=True)
    if next_step == 'xy_prep':
        print('XY configuration not found. ' \
        'Run the following command for help:\n    run_alignment.py xy prep --help')
        return None
    elif next_step == 'z_prep':
        print('Z configuration not found. ' \
        'Run the following command for help:\n    run_alignment.py z prep --help')
        return None
    elif next_step is None:
        print('All steps are finished.')
        return None

    # All steps take the project directory
    module_name, func_name = STEP_TO_FUNC[next_step].rsplit('.', 1)
    return getattr(importlib.import_module(module_name), func_name)


if __name__ == '__main__':

    # Select the GPU(s) first to set CUDA_VISIBLE_DEVICES before imports
    gpu_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    gpu_parser.add_argument('-g', '--gpu', dest='gpu_id', type=str, default=None)
    gpu_id = gpu_parser.parse_known_args()[0].gpu_id
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    from emalign import prep_config_xy, align_dataset_xy, prep_config_z
    from emalign.scripts import (
        fuse_stacks_xy,
        compute_transforms,
        compute_flow,
        render_dataset
        )

    parser = argparse.ArgumentParser(prog='run_alignment')
    parser.add_argument('--verbose', action='store_true')  # global flag
    parser.add_argument('-p', '--project-dir',
                        metavar='PROJECT_DIR',
                        dest='auto_project_dir',
                        default=None,
                        type=str,
                        help='Without a step (xy or z): run whatever should run next for this project.')
    parser.add_argument('-c', '--cores',
                        metavar='CORES',
                        dest='auto_num_workers',
                        type=int,
                        default=None,
                        help='Without a step: number of threads for the next step. Default: the step\'s own default')
    parser.add_argument('-g', '--gpu',
                        metavar='GPU_ID',
                        dest='gpu_id',
                        type=str,
                        default=None,
                        help='GPU ID(s) to use, e.g. 0 or 0,1. Default: CUDA_VISIBLE_DEVICES')
    steps = parser.add_subparsers(dest='alignment_step', required=False)

    # ---------- XY ----------
    xy = steps.add_parser('xy', help='Stitch tiles within each slice along the XY axes')
    xy_steps = xy.add_subparsers(dest='xy_step', required=True)

    # Prep
    prep_xy = xy_steps.add_parser('prep', help='Generate configuration files for XY alignment')
    prep_config_xy.add_parser_arguments(prep_xy)
    prep_xy.set_defaults(func=prep_config_xy.prep_align_stacks)

    # Align
    align_xy = xy_steps.add_parser('align', help='Align and stitch the XY stacks')
    align_dataset_xy.add_parser_arguments(align_xy)
    align_xy.set_defaults(func=align_dataset_xy.align_dataset_xy)

    # Fuse
    fuse_xy = xy_steps.add_parser('fuse', help='Fuse stitched images overlapping on the same Z slices')
    fuse_stacks_xy.add_parser_arguments(fuse_xy)
    fuse_xy.set_defaults(func=fuse_stacks_xy.fuse_dataset_xy)

    # ---------- Z ----------
    z = steps.add_parser('z', help='Align slices along the Z axis')
    z_steps = z.add_subparsers(dest='z_step', required=True)

    # Prep
    prep_z = z_steps.add_parser('prep', help='Generate configuration files for Z alignment')
    prep_config_z.add_parser_arguments(prep_z)
    prep_z.set_defaults(func=prep_config_z.prep_config_z)

    # Transforms
    z_trsf = z_steps.add_parser('transform', help='Compute affine transformations for rough alignment along the Z axis')
    compute_transforms.add_parser_arguments(z_trsf)
    z_trsf.set_defaults(func=compute_transforms.compute_dataset_transforms)

    # Flow
    z_flow = z_steps.add_parser('flow', help='Compute flow and relax the resulting mesh')
    compute_flow.add_parser_arguments(z_flow)
    z_flow.set_defaults(func=compute_flow.compute_dataset_flow)

    # Render
    z_render = z_steps.add_parser('render', help='Render the aligned dataset')
    render_dataset.add_parser_arguments(z_render)
    z_render.set_defaults(func=render_dataset.render_dataset)

    # ---------- Run ----------
    args = vars(parser.parse_args())
    logging.basicConfig(level=logging.DEBUG if args.pop('verbose') else logging.INFO)

    auto_project_dir = args.pop('auto_project_dir')
    auto_num_workers = args.pop('auto_num_workers')
    args.pop('gpu_id')  # Already applied before the imports

    if args['alignment_step'] is None:
        # No step given, figure out what comes next
        if auto_project_dir is None:
            parser.error('Give a step (xy or z), or a project directory with -p to run the next step.')
        func = get_project_next_step(auto_project_dir)
        if func is not None:
            logging.info(f'Next step: {func.__module__}.{func.__name__}\n')
            kwargs = {'project_dir': auto_project_dir}
            if auto_num_workers is not None:
                kwargs['num_workers'] = auto_num_workers
            func(**kwargs)
    else:
        # Keep only the arguments of the step itself
        func = args.pop('func')
        for key in ('alignment_step', 'xy_step', 'z_step'):
            args.pop(key, None)
        func(**args)
