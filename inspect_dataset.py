import argparse
import json
import os
import sys
import tensorstore as ts

from glob import glob

from emalign.align_z.utils import get_ordered_datasets
from emalign.visualize.nglancer import add_layers, start_nglancer_viewer


def read_data(
            dataset_path,
            data_range=None,
            keep_missing=False):
    
    spec = {'driver': 'zarr',
                        'kvstore': {
                                'driver': 'file',
                                'path': dataset_path,
                                    }}
    dataset = ts.open(spec,
                      read=True,
                      dtype=ts.uint8
                      ).result()
    
    if data_range is None:
        data = dataset[:].read().result()
    else:
        z0 = data_range[0]
        if len(data_range) == 2:
            # Bound range to the possible values
            z1 = min(data_range[1], dataset.domain.exclusive_max[0])
        elif len(data_range) == 1:
            # Only one value, means we go from that value to the end
            z1 = dataset.domain.exclusive_max[0]
        data = dataset[z0:z1].read().result()

    if not keep_missing:
        data = data[data.any(axis=(1,2))]
    
    return data


def inspect_dataset(
            dataset_path,
            data_range=[0],
            keep_missing=False,
            project_configs=[],
            mode=None,
            bind_port=55555,
            print_shape=False):
    '''Display images from a zarr store.

    Loads images from a zarr store as defined by the data range, and display it in a neuroglancer viewer.
    Viewer's bind_address is currently hard-coded to be localhost.

    Args:
        dataset_path (str): Absolute path to the zarr store to read data from.
        data_range (list of `int`, optional): Range of z indices to read data from: [inclusive_min, exclusive_max]
            If only one is int given, it will be considered the start and the end will be the last possible index. Defaults to [0].
        keep_missing (bool, optional): Whether to skip fully black images. Defaults to False.
        project_configs (list of `str`, optional): List of absolute paths to configuration files containing information about datasets to display, when mode=z_transitions. Defaults to [].
        mode (str, optional): Mode to use to display data. If no mode is given, will simply read data from the path provided.
            One of: `None`, z_transitions, all_ds. Defaults to None.
            z_transitions: Determines from project_configs all the z indices where a transition occurred (i.e. two stacks were aligned) and show images around transitions.
            all_ds: Reads data within data_range from all the datasets found in the provided store.
        bind_port (int, optional): Port to bind the neuroglancer viewer to. Defaults to 55555.
    '''
    
    modes = ['z_transitions', 'all_ds']
    if mode is not None and mode not in modes:
        raise ValueError(f'Invalid mode. Must be one of: {modes}')

    if print_shape:
        spec = {'driver': 'zarr',
                'kvstore': {
                         'driver': 'file',
                         'path': dataset_path,
                            }}
        dataset = ts.open(spec,
                          read=True,
                          dtype=ts.uint8
                          ).result()
        print(f'Dataset shape (ZYX):\n    {dataset.shape}')
        sys.exit()

    
    # Start viewer
    viewer = start_nglancer_viewer(bind_address='localhost',
                                   bind_port=bind_port)
    print('Neuroglancer viewer: ' + viewer.get_viewer_url())
    print('Please wait for images to load (CTRL+C to cancel).')
    
    # Prepare data
    dataset_name = os.path.basename(os.path.abspath(dataset_path))
    if mode is None:
        d = read_data(dataset_path, data_range=tuple(data_range), keep_missing=keep_missing)
        add_layers([d], 
                   viewer, 
                   voxel_offsets=[[0,0,data_range[0]]],
                   names=[dataset_name])
    elif mode == 'z_transitions':
        dataset_paths = []
        config_paths = glob(os.path.join(project_configs, '*.json'))

        for config_path in config_paths:
            with open(config_path, 'r') as f:
                config = json.load(f)
            dataset_paths.append(os.path.join(config['dataset_path']))

        _, z_offsets = get_ordered_datasets(dataset_paths)

        window = 20
        for z, _, _ in z_offsets:
            data_range = [int(z - max(1, window/2)), int(z + max(1, window/2))]

            try:
                d = read_data(dataset_path, data_range=tuple(data_range), keep_missing=keep_missing)
            except:
                continue
            visible = z == z_offsets[0]
            add_layers([d], 
                       viewer, 
                       names=[f'{dataset_name}_{z}'], 
                       voxel_offsets=[[0,0,data_range[0]]],
                       visible=visible,
                       clear_viewer=False)
    elif mode == 'all_ds':
        dataset_paths = [d for d in sorted(glob(os.path.join(dataset_path, '*'))) if '_mask' not in d]

        for dataset_path in dataset_paths:
            dataset_name = os.path.basename(dataset_path)
            d = read_data(dataset_path, data_range=tuple(data_range), keep_missing=keep_missing)
            visible = dataset_path == dataset_paths[0]
            add_layers([d], 
                       viewer, 
                       names=[dataset_name], 
                       voxel_offsets=[[0,0,data_range[0]]],
                       visible=visible,
                       clear_viewer=False)
    input('All data loaded. Press ENTER or ESCAPE to exit.')


if __name__ == '__main__':

    parser=argparse.ArgumentParser('Inspect image data stored in a zarr container.')
    
    parser.add_argument('-d', '--dataset-path',
                        metavar='DATASET_PATH',
                        dest='dataset_path',
                        required=True,
                        type=str,
                        default=None,
                        help='Path to the zarr container containing the final alignment.')
    parser.add_argument('--data-range',
                        metavar='DATA_RANGE',
                        dest='data_range',
                        nargs='+',
                        type=int,
                        default=[0],
                        help='Range of slice indices to show. One value will be consider as the lower bound.' \
                             'If too high, will be bounded to the max possible value.')
    parser.add_argument('--keep-missing',
                        dest='keep_missing',
                        default=False,
                        action='store_true',
                        help='Keep missing slices as black images. Default: False')
    parser.add_argument('-cfg', '--config',
                        metavar='PROJECT_CONFIGS',
                        dest='project_configs',
                        required=False,
                        # nargs='+',
                        type=str,
                        help='Path to the project configs containing information about the dataset\'s transitions.')
    parser.add_argument('--mode',
                        dest='mode',
                        type=str,
                        default=None,
                        help='Visualization mode. One of: z_transitions, all_ds')
    parser.add_argument('--port',
                        metavar='PORT',
                        dest='bind_port',
                        required=False,
                        type=int,
                        default=55555,
                        help='Port to use for neuroglancer.')
    parser.add_argument('--print-shape',
                        dest='print_shape',
                        required=False,
                        action='store_true',
                        default=False,
                        help='Print the dataset\'s shape in pixels without displaying it')
    
    args=parser.parse_args()

    inspect_dataset(**vars(args))
