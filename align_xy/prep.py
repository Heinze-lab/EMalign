import json
import logging
import networkx as nx
import numpy as np
import os
import pandas as pd

from concurrent import futures
from itertools import combinations
from glob import glob

from emprocess.utils.io import load_tif
from emprocess.utils.io import get_dataset_attributes
from emalign.io.store import find_ref_slice
from ..arrays.sift import estimate_transform_sift
from ..arrays.stacks import Stack
from ..arrays.utils import resample
from ..visualize.nglancer import add_layers, start_nglancer_viewer
from ..align_z.utils import get_ordered_datasets



def find_offset_from_main_config(main_config_path):
    '''Find z offset of a new stack based on the config of a preceding one.

    Args:
        main_config_path (str): Absolute path to a main_config JSON file as written by prep_config_z.json

    Returns:
        int: Z offset of a new stack that would follow the one represented by main_config (i.e. previous stack end + 1)
    '''

    if not os.path.exists(main_config_path):
        raise FileNotFoundError(f'Config file not found: {main_config_path}')

    with open(main_config_path, 'r') as f:
        main_config = json.load(f)

    if 'stack_configs' not in main_config:
        raise ValueError(f'Invalid config: missing "stack_configs" key in {main_config_path}')

    stack_configs = main_config['stack_configs']
    if not stack_configs:
        raise ValueError(f'No stacks found in config: {main_config_path}')

    z_offsets = []
    for stack_name, config_path in stack_configs.items():
        if not os.path.exists(config_path):
            logging.warning(f'Stack config not found: {config_path}, skipping')
            continue

        with open(config_path, 'r') as f:
            stack_config = json.load(f)

        if 'z_end' not in stack_config:
            logging.warning(f'Missing z_end in {config_path}, skipping')
            continue

        z_offsets.append(stack_config['z_end'])

    if not z_offsets:
        raise ValueError(f'No valid z_end values found in any stack configs')

    return max(z_offsets) + 1


def get_stacks(stack_paths, 
               invert_instructions):
    '''Get segments of potentially overlapping stacks from paths. 

    Use a list of tileset paths to fetch stacks and split them into overlapping segments.
    Invert instructions are used to determine whether images in a tilest need to be inverted.

    Args:
        stack_paths (list of `str`): List of paths to folders each containing images from a single stack.
        invert_instructions (dict of `bool`): Dictionary from stack to boolean determining whether to invert images in that stack.

    Returns:
        dict: Dictionary from stack name to Stack object. Value may be a list of Stack objects if they share Z indices.
    '''

    # Load stacks
    stacks = []
    for stack_path in stack_paths:
        stack = Stack(stack_path)
        stack._get_tilemaps_paths()
        if stack.stack_name not in invert_instructions:
            logging.error(f'Stack "{stack.stack_name}" not found in invert_instructions')
            raise ValueError(f'Missing invert instructions for stack: {stack.stack_name}')
        for k in stack.tile_maps_invert.keys():
            stack.tile_maps_invert[k]=invert_instructions[stack.stack_name]
        stacks.append(stack) 

    # Split stacks if there are overlaps
    unique_slices = sorted(np.unique(np.concatenate([stack.slices for stack in stacks])).tolist())
    df = pd.DataFrame({'z': unique_slices, 
                       'stack_name': [[] for _ in range(len(unique_slices))], 
                       'tile_paths':[[] for _ in range(len(unique_slices))]
                      })

    for stack in stacks:
        for z in stack.slices:
            # Join existing name and this stack at that slice
            df.loc[df.z == z, ['stack_name']] += [[stack.stack_name]]

            # Concatenate tile paths
            df.loc[df.z == z, ['tile_paths']] += [[stack.slice_to_tilemap[z]]]

    df['group'] = df['stack_name'].ne(df['stack_name'].shift()).cumsum()

    new_stacks = {}
    for group, group_df in df.groupby('group'):    
        stack_names = group_df.stack_name.iloc[0]

        if len(stack_names) == 1:
            # Stack name becomes name + group (gives an idea of order too)
            new_stack_name = str(group).zfill(2) + '_' + stack_names[0]

            tile_map = {}
            for z in group_df.z:
                tile_map[z] = group_df.loc[group_df.z == z, 'tile_paths'].item()[0]
            
            stack = Stack()
            stack.stack_name = new_stack_name
            stack._set_tilemaps_paths(tile_map)
            if stack_names[0] not in invert_instructions:
                logging.error(f'Stack "{stack_names[0]}" not found in invert_instructions')
                raise ValueError(f'Missing invert instructions for stack: {stack_names[0]}')
            stack.tile_maps_invert = {k: invert_instructions[stack_names[0]] for k in tile_map[z].keys()}

            new_stacks[new_stack_name] = stack
        else:
            combined_stack_name = '_'.join([str(group).zfill(2)] + stack_names)
            pair = []
            for i in range(len(stack_names)):
                new_stack_name = str(group).zfill(2) + '_' + stack_names[i]
                
                tile_map = {}
                for z in group_df.z:
                    tile_map[z] = group_df.loc[group_df.z == z, 'tile_paths'].item()[i]

                stack = Stack()
                stack.stack_name = new_stack_name
                stack._set_tilemaps_paths(tile_map)
                if stack_names[i] not in invert_instructions:
                    logging.error(f'Stack "{stack_names[i]}" not found in invert_instructions')
                    raise ValueError(f'Missing invert instructions for stack: {stack_names[i]}')
                stack.tile_maps_invert = {k: invert_instructions[stack_names[i]] for k in tile_map[z].keys()}

                pair.append(stack)
            new_stacks[combined_stack_name] = pair
        
    return new_stacks


def check_stacks_to_invert(stack_list, 
                           num_workers=1, 
                           **kwargs):

    '''Check what stacks must be inverted

    Display the first image of each stack in neuroglancer viewer, and prompt user to determine whether a stack needs to be inverted. 

    Args:
        stack_list (`list` of `emalign.align.xy.stacks.Stack`): List of stacks to visualize.
        num_workers (int, optional): Number of threads used to open images. Defaults to 1.
        **kwargs: Arguments passed to `start_nglancer_viewer`.

    Returns:
        dict: Dictionary from stack_names to decision to invert: either True or False.
    '''

    viewer = start_nglancer_viewer(**kwargs)
    print('Neuroglancer viewer: ' + viewer.get_viewer_url())
    print('Please wait for images to load (CTRL+C to cancel).')

    to_invert = {}
    with futures.ThreadPoolExecutor(num_workers) as tpe:
        fs = {}
        for stack_path in sorted(stack_list):
            stack_name = os.path.basename(os.path.normpath(stack_path))
            tif_files = glob(os.path.join(stack_path, '*.tif'))
            if not tif_files:
                logging.warning(f'No TIF files found in {stack_path}, skipping')
                to_invert[stack_name] = False
                continue

            fs[stack_name] = tpe.submit(load_tif, tif_files[0], 1, {})

        for i, (stack_name, f) in enumerate(fs.items()):
            arr = f.result()[0]
            add_layers([arr],
                        viewer,
                        names=[stack_name],
                        clear_viewer=True)

            answer = input(f'{str(i+1).zfill(2)}/{len(fs)} - Invert {stack_name}? (y/n) ').strip(' ')
            while answer.lower() not in ['y', 'n', '']:
                answer = input(f'{str(i+1).zfill(2)}/{len(fs)} - Please provide a valid answer for {stack_name}: (y/n) ').strip(' ')

            if answer.lower() == 'y':
                to_invert.update({stack_name: True})
            elif answer.lower() == 'n' or answer == '':
                to_invert.update({stack_name: False})
    return to_invert


# FUSE STACKS
def create_configs_fused_stacks(main_config_path,
                                scale = 0.1
                                ):
    
    # Target resolution
    with open(main_config_path, 'r') as f:
        target_res = json.load(f)['resolution'][-1]

    # Find datasets
    datasets, z_offsets = get_ordered_datasets([main_config_path], exclude=['flow', 'mask', '10x'])
    z_ranges = [np.arange(z[0], z[0] + ds.shape[0]) for z, ds in zip(z_offsets, datasets)]

    # Find all ranges over which there is overlap
    unique_slices = sorted(np.unique(np.concatenate(z_ranges)).tolist())
    df = pd.DataFrame({'z': unique_slices, 
                    'ds_indices': [[] for _ in range(len(unique_slices))]
                        })
    extend_list = lambda lst: lst + [datasets.index(ds)]
    for ds, z_range in zip(datasets, z_ranges):
        df.loc[df.z.isin(z_range), 'ds_indices'] = df.loc[df.z.isin(z_range), 'ds_indices'].apply(extend_list)
    df['group'] = df['ds_indices'].ne(df['ds_indices'].shift()).cumsum()

    # Test overlap and create fused config in consequence
    fused_configs = []
    for _, group in df.groupby('group'):
        z = group.z.min()
        indices = np.unique(group.ds_indices.to_numpy())[0]

        images = []
        for i in indices:
            ds = datasets[i]

            # Downsample if necessary
            yx_res = get_dataset_attributes(ds)['resolution'][-1]
            target_scale = yx_res/target_res
            img, _ = find_ref_slice(ds, z - z_offsets[i, 0]) # Could be better by accounting for gaps
            images.append(resample(img, target_scale))

        # Test images and store valid matches
        G = nx.Graph()
        for i, j in combinations(range(len(images)), 2):
            valid_estimate = estimate_transform_sift(images[i], images[j], scale, refine_estimate=True)[3]
            if valid_estimate:
                G.add_edge(indices[i], indices[j])

        # Valid matches are chained in case there are more than 2 matches for a range
        for cc in nx.connected_components(G):
            config = {
                'dataset_paths': [datasets[i].kvstore.path for i in cc], 
                'z_offsets': [int(z_offsets[i,0]) for i in cc],
                'zmin': int(z), 
                'zmax': int(group.z.max()) + 1 # Exclusive max
                } 
            fused_configs.append(config)
    return fused_configs