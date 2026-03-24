''' Utilities for alignment of stacks along Z axis.'''

import json
from emalign.arrays.utils import resample
from emalign.io.store import find_ref_slice, open_store
import networkx as nx
import numpy as np
import os
import tensorstore as ts
import pandas as pd

from cv2 import warpAffine
from glob import glob

from emalign.io.store import get_store_attributes

from ..arrays.sift import estimate_transform_sift


def get_ordered_datasets(config_paths, exclude=[]):
    '''Open and order datastacks based on Z offset.

    Args:
        dataset_paths (list): List of paths to the datasets to open and order.
        exclude (list, optional): List of strings to find in paths. If the string is found, the path will be ignored. 

    Returns:
        tuple: tuple of:
            List of tensorstore.TensorStore
            List of corresponding voxel offsets.
    '''

    config_groups = []
    for config in config_paths:
        if isinstance(config, list):
            config_groups.append(config)
        else:
            config_groups.append([config])

    # Check config one by one
    dataset_stores = []
    offsets = []
    previous_offset = 0
    for config_group in config_groups:
        group_offsets = []
        z_shapes = []
        for config_path in config_group:
            with open(config_path, 'r') as f:
                main_config = json.load(f)
            
            # Get info from config
            output_path     = main_config['output_path']
            dataset_paths = glob(os.path.join(output_path, 'xy_intermediate', '*/'))

            for ds in dataset_paths:
                check = [pattern in ds for pattern in exclude]
                if any(check) or ds.endswith('_mask'):
                    # Always exclude masks from query
                    continue
                dataset = open_store(ds, mode='r')
                z_shapes.append(dataset.shape[0])

                offset = get_store_attributes(dataset)['voxel_offset']
                offset[0] += previous_offset # Shift this dataset by the previous dataset's offset
                group_offsets.append(offset)
                offsets.append(offset)
                dataset_stores.append(dataset)

        # If configs are supposed to be consecutive stacks, the offsets should match that
        previous_offset = np.array(group_offsets)[:,0].max() + z_shapes[np.array(group_offsets)[:,0].argmax()]

    offsets = np.array(offsets)

    # Make sure that datasets come in the right order (offsets)
    dataset_stores = [dataset_stores[i] for i in np.argsort(offsets[:, 0])]
    offsets = offsets[np.argsort(offsets[:, 0])]
    return dataset_stores, offsets


def extract_paths_from_root(G, root_node):
    '''Produce alignment path(s) starting at the root dataset.

    Args:
        G (nx.Graph): Undirected graph where nodes are dataset indices and edges represent valid overlap between neighboring datasets.
        root_node (int): Index of the root dataset, from which alignment will start.

    Returns:
        list: List of lists of int defining the order of alignment with dataset indices.
    '''
    # Special nodes: degree != 2 (root, leaves, and branch points)
    special = [root_node] + list({n for n, d in G.degree() if d != 2 and n != root_node})
    paths = []

    for node in special:
        for neigh in G.neighbors(node):
            if node < neigh:  # prevent duplicating opposite directions
                path = [node, neigh]
                prev, current = node, neigh
                while current not in special or current == node:
                    next_nodes = [n for n in G.neighbors(current) if n != prev]
                    if not next_nodes:
                        break
                    prev, current = current, next_nodes[0]
                    path.append(current)
                paths.append(path)

    # Order paths so they are traversed properly
    ordered_paths = []
    remove = []
    for p in paths:
        if root_node == p[0]:
            ordered_paths.append(p)
            remove.append(p)
        elif root_node == p[-1]:
            ordered_paths.append(p[::-1])
            remove.append(p)
    [paths.remove(p) for p in remove]

    try_reverse = False # Prioritize forward pass
    while paths:
        remove = []
        for path in paths:
            if any([path[0] in p for p in ordered_paths]):
                ordered_paths.append(path)
                remove.append(path)
            elif any([path[-1] in p for p in ordered_paths]) and try_reverse:
                ordered_paths.append(path[::-1])
                remove.append(path)
        try_reverse = not bool(remove)
        [paths.remove(r) for r in remove]
    return ordered_paths


def compute_alignment_path(datasets,
                           z_offsets,
                           target_resolution,
                           scale=0.2):
    '''Compute alignment paths between overlapping datasets using SIFT feature matching.

    Analyzes Z-overlap between datasets and builds a graph where edges represent
    valid alignment transitions (verified by SIFT matching at slice boundaries).
    Returns paths from a root dataset (one with no overlap) to all other datasets.

    Args:
        datasets (list): List of tensorstore.TensorStore objects to align.
        z_offsets (np.ndarray): Array of shape (N, 3) with [z, y, x] voxel offsets for each dataset.
        target_resolution (int or list): Target resolution in nm for SIFT matching. If int, used for both Y and X.
        scale (float, optional): Scale factor for SIFT feature detection. Defaults to 0.2.

    Raises:
        RuntimeError: If no root dataset is found (all datasets have Z overlap).
        RuntimeError: If some datasets are disconnected from the main alignment graph.

    Returns:
        tuple: (root_node, paths, reverse_z, ds_bounds) where:
            - root_node (str): Name of the root dataset from which alignment starts.
            - paths (list): List of lists of dataset names defining alignment order.
            - reverse_z (list): List of bools indicating if path traverses Z in reverse.
            - ds_bounds (dict): Dict mapping dataset names to (z_min, z_max) bounds.
    '''
    
    if isinstance(target_resolution, int):
        target_resolution = [target_resolution, target_resolution]
    
    def _get_slice(store, z, reverse, target_resolution=target_resolution):
        resolution = get_store_attributes(store)['resolution']
        assert resolution[-2] == resolution[-1], 'Resolution must be the same in X and Y'
        assert target_resolution[-2] == target_resolution[-1], 'Target resolution must be the same in X and Y'
        target_scale = resolution[-1]/target_resolution[-1]

        img = find_ref_slice(store, z, reverse=reverse)[0]
        return resample(img, target_scale)
    
    # Find all ranges over which there is overlap
    z_ranges = [np.arange(z[0], z[0] + ds.shape[0]) for z, ds in zip(z_offsets, datasets)]
    unique_slices = sorted(np.unique(np.concatenate(z_ranges)).tolist())
    df = pd.DataFrame({'z': unique_slices, 
                    'ds_indices': [[] for _ in range(len(unique_slices))]
                        })
    extend_list = lambda lst: lst + [datasets.index(ds)]
    for ds, z_range in zip(datasets, z_ranges):
        df.loc[df.z.isin(z_range), 'ds_indices'] = df.loc[df.z.isin(z_range), 'ds_indices'].apply(extend_list)
    df['group'] = df['ds_indices'].ne(df['ds_indices'].shift()).cumsum()

    # Remove datasets that we fused only at the relevant Z indices
    z_levels = {}
    for g, group in df.groupby('group'):
        # Find datasets to remove from that slice
        indices = np.unique(group.ds_indices.to_numpy())[0]
        names = [datasets[i].kvstore.path.split('/')[-2] for i in indices]
        
        ignore_indices = []
        for name in names:
            if 'fused' in name:
                ignore_indices += [indices[i] for i,n in enumerate(names) if n in name and n != name]
        df.loc[df.group == g, 'ds_indices'] = df.loc[df.group == g, 'ds_indices'].apply(lambda l: list(set(l).difference(ignore_indices)))

        # Keep track of where there are changes of datasets
        z_levels[g] = (group.z.min(), group.z.max())

    # Find first dataset alone at its own z level
    root_datasets = df.ds_indices[df.ds_indices.apply(len) == 1] 

    if len(root_datasets) == 0:
        raise RuntimeError('No potential root dataset was found: no dataset with no overlap along Z.')

    root_node_idx = root_datasets[0][0]
    root_node = os.path.basename(os.path.abspath(datasets[root_node_idx].kvstore.path))

    # Compute valid alignment paths
    G = nx.Graph()
    G.add_nodes_from(np.unique(np.concatenate(df.ds_indices)).tolist())
    grouped = df.groupby('group')
    for g, curr_group in grouped:
        if g == df.group.max():
            break
        next_group = grouped.get_group(g+1)
        for u in curr_group.ds_indices.iloc[0]:
            for v in next_group.ds_indices.iloc[0]:
                # Check for match at the boundary of the relevant range
                ref = _get_slice(datasets[u], curr_group.z.max() - z_offsets[u, 0], reverse=True)
                mov = _get_slice(datasets[v], next_group.z.min() - z_offsets[v, 0], reverse=False)
                M, out_shape, ref_offset, valid_estimate, _ = estimate_transform_sift(ref.copy(), mov.copy(), scale=scale, refine_estimate=True)

                if valid_estimate:
                    # Keep track of everything, mostly for debugging
                    G.add_edge(u,v, M=M, out_shape=out_shape, ref_offset=ref_offset, valid_estimate=valid_estimate)

    if not nx.is_connected(G):
        # Some datasets are disconnected from the main alignment path
        x = [[os.path.basename(os.path.abspath(datasets[i].kvstore.path)) for i in cc] for cc in nx.connected_components(G)]
        raise RuntimeError(f'Some datasets are isolated: \n{x}')

    paths = extract_paths_from_root(G, root_node_idx)
    reverse_z = [bool(z_offsets[p[0], 0] > z_offsets[p[-1], 0]) for p in paths]
    paths = [[os.path.basename(os.path.abspath(datasets[i].kvstore.path)) for i in p] for p in paths]

    # Datasets will need to be bounded to not re-use fused images
    ds_bounds = {}
    for i in np.unique(np.concatenate(df.ds_indices.to_numpy())):
        zmin = df.loc[df.ds_indices.apply(lambda l: i in l), 'z'].min() - z_offsets[i, 0]
        zmax = df.loc[df.ds_indices.apply(lambda l: i in l), 'z'].max() - z_offsets[i, 0] + 1  # Exclusive max
        ds_bounds[os.path.basename(os.path.abspath(datasets[i].kvstore.path))] = (int(zmin), int(zmax))
    return root_node, paths, reverse_z, ds_bounds


def determine_initial_offset(datasets, paths):
    '''Estimate cumulative XY offset needed to accommodate drift across alignment paths.

    Traverses each alignment path, computing SIFT-based transforms between consecutive
    datasets, and tracks the accumulated offset. Returns the maximum negative offset
    encountered, which represents the padding needed at the origin.

    Args:
        datasets (list or dict): Either a list of tensorstore.TensorStore objects, or a
            dict mapping dataset names to TensorStore objects.
        paths (list): List of alignment paths (each path is a list of dataset names).

    Returns:
        np.ndarray: Array of shape (2,) with [y, x] offset to apply as padding at origin.
    '''
    if not isinstance(datasets, dict):
        datasets = {os.path.basename(os.path.abspath(d.kvstore.path)): d for d in datasets}
    
    global_offset = np.array([0,0])
    for path in paths:
        path_offset = np.array([0,0])

        prev = find_ref_slice(datasets[path[0]], reverse=True)[0]
        for stack_name in path[1:]:
            ds_curr = datasets[stack_name]
            curr = find_ref_slice(ds_curr, reverse=False)[0]

            M, output_shape, prev_offset, _, _ = estimate_transform_sift(prev, curr, scale=0.1, refine_estimate=True)

            prev = warpAffine(find_ref_slice(ds_curr, reverse=True)[0], M, output_shape[::-1])
            path_offset += prev_offset
        
        global_offset = np.min([global_offset, path_offset], axis=0)
    
    return np.abs(global_offset)