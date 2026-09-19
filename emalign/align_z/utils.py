''' Utilities for alignment of stacks along Z axis.'''

import json
from emalign.arrays.utils import resample
from emalign.io.store import find_ref_slice, open_store
import logging
import networkx as nx
import numpy as np
import os
import tensorstore as ts
import pandas as pd

from cv2 import warpAffine
from glob import glob
from tqdm import tqdm

from ..io.store import get_store_attributes
from ..io.process.mask import compute_greyscale_mask
from ..arrays.sift import estimate_transform_sift


from ..arrays.sift import SIFT_SCALES
N_CANDIDATE_SLICES = 3    # Number of slice pairs to try per transition before rejecting it


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
                if any(check) or os.path.abspath(ds).endswith('_mask'):
                    # Always exclude masks from query
                    continue
                dataset = open_store(ds, mode='r')
                z_shapes.append(dataset.shape[0])

                
                attrs = get_store_attributes(dataset)
                if attrs is None:
                    raise FileNotFoundError(f'Attributes do not exist for dataset: {ds}')

                offset = attrs['voxel_offset']
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


def _datasets_without_masks(datasets):
    '''Drop the mask stores from a dataset list.

    Args:
        datasets (list): List of tensorstore.TensorStore objects.

    Returns:
        tuple: (list of TensorStore without the mask stores, list of their names).
    '''
    datasets_nomask = [d for d in datasets
                       if not os.path.abspath(d.kvstore.path).endswith('_mask')]
    names = [os.path.basename(os.path.abspath(d.kvstore.path)) for d in datasets_nomask]
    return datasets_nomask, names


def _occupancy_table(datasets_nomask, names, z_offsets):
    '''Build the table of which datasets cover which global Z slice.

    Contiguous runs of slices covered by the same set of datasets are labelled with a
    'group' id, so that group boundaries mark where the set of datasets changes. Datasets
    that were fused into another one are dropped from the groups where their fused version
    is present, so that their images are not used twice. A fused dataset declares the datasets
    it supersedes in its 'fused_from' attribute, written by scripts/fuse_stacks_xy.py.

    The drop is per group rather than global, because a dataset may extend beyond the Z range
    over which it was fused, and must still be used outside of it.

    Args:
        datasets_nomask (list): List of tensorstore.TensorStore objects, masks excluded.
        names (list): Dataset names, indexed like datasets_nomask.
        z_offsets (np.ndarray): Array of shape (N, 3) with [z, y, x] voxel offsets.

    Returns:
        pandas.DataFrame: Table with columns 'z', 'ds_indices' and 'group'.
    '''
    z_ranges = [np.arange(z[0], z[0] + ds.shape[0]) for z, ds in zip(z_offsets, datasets_nomask)]
    unique_slices = sorted(np.unique(np.concatenate(z_ranges)).tolist())
    df = pd.DataFrame({
        'z': unique_slices,
        'ds_indices': [[] for _ in range(len(unique_slices))]
                        })
    extend_list = lambda lst: lst + [datasets_nomask.index(ds)]
    for ds, z_range in zip(datasets_nomask, z_ranges):
        df.loc[df.z.isin(z_range), 'ds_indices'] = df.loc[df.z.isin(z_range), 'ds_indices'].apply(extend_list)
    df['group'] = df['ds_indices'].ne(df['ds_indices'].shift()).cumsum()

    # Which datasets each fused dataset supersedes
    fused_sources = {}
    for i, dataset in enumerate(datasets_nomask):
        sources = get_store_attributes(dataset).get('fused_from')
        if sources is None and 'fused' in names[i]:
            # Stacks fused before fused_from was recorded: fall back to the naming convention,
            # where the name of a fused stack contains the names of the stacks it comes from
            sources = [n for n in names if n != names[i] and n in names[i]]
        if sources:
            fused_sources[i] = set(sources)

    # Remove datasets that we fused only at the relevant Z indices
    for g, group in df.groupby('group'):
        # Find datasets to remove from that slice
        indices = np.unique(group.ds_indices.to_numpy())[0]

        superseded = set()
        for i in indices:
            superseded |= fused_sources.get(i, set())
        ignore_indices = [i for i in indices if names[i] in superseded]

        df.loc[df.group == g, 'ds_indices'] = df.loc[df.group == g, 'ds_indices'].apply(lambda l: list(set(l).difference(ignore_indices)))

    return df


def _dataset_bounds(df, names, z_offsets):
    '''Derive the local Z bounds of every dataset from an occupancy table.

    Args:
        df (pandas.DataFrame): Occupancy table from _occupancy_table.
        names (list): Dataset names, indexed like the occupancy table indices.
        z_offsets (np.ndarray): Array of shape (N, 3) with [z, y, x] voxel offsets.

    Returns:
        dict: Dict mapping dataset names to (z_min, z_max) local bounds, z_max exclusive.
    '''
    ds_bounds = {}
    for i in np.unique(np.concatenate(df.ds_indices.to_numpy())):
        zmin = df.loc[df.ds_indices.apply(lambda l: i in l), 'z'].min() - z_offsets[i, 0]
        zmax = df.loc[df.ds_indices.apply(lambda l: i in l), 'z'].max() - z_offsets[i, 0] + 1  # Exclusive max
        ds_bounds[names[i]] = (int(zmin), int(zmax))
    return ds_bounds


def compute_dataset_bounds(datasets, z_offsets):
    '''Compute the local Z bounds of every dataset, without testing any transition.

    This is the part of compute_alignment_path that does not depend on datasets being
    aligned to each other. Use it when every dataset is aligned to an external reference:
    there is no chaining between datasets, so the SIFT transition graph is not needed, but
    the bounds still are, to avoid rendering a fused dataset and its sub-datasets twice.

    Args:
        datasets (list): List of tensorstore.TensorStore objects to align.
        z_offsets (np.ndarray): Array of shape (N, 3) with [z, y, x] voxel offsets for each dataset.

    Returns:
        tuple: (dataset_names, ds_bounds) where:
            - dataset_names (list): Names of the datasets that have bounds, in Z order.
              Datasets entirely superseded by a fused version are not included.
            - ds_bounds (dict): Dict mapping dataset names to (z_min, z_max) bounds.
    '''
    datasets_nomask, names = _datasets_without_masks(datasets)

    if len(datasets_nomask) == 1:
        return names, {names[0]: (0, datasets_nomask[0].shape[0])}

    df = _occupancy_table(datasets_nomask, names, z_offsets)
    ds_bounds = _dataset_bounds(df, names, z_offsets)

    # Preserve the Z ordering of the input, minus anything the fused filtering removed
    return [n for n in names if n in ds_bounds], ds_bounds


def format_transition(record, names):
    '''Render one transition test (and every SIFT attempt it made) as log lines.

    Args:
        record (dict): Transition record produced by compute_alignment_path.
        names (list): List of dataset names indexed by dataset index.

    Returns:
        str: Multi-line, human readable summary of the transition test.
    '''
    verdict = 'ACCEPT' if record['accepted'] else 'REJECT'
    lines = [f'  {verdict}  {names[record["u"]]} -> {names[record["v"]]} '
             f'(boundary at global z={record["z_boundary"]})']
    for attempt in record['attempts']:
        if 'error' in attempt:
            lines.append(f'      step {attempt["step"]}: {attempt["error"]}')
            continue

        metrics = attempt['metrics'] or {}
        if 'robustness_index' in metrics:
            detail = (f'index={metrics["robustness_index"]:.3f} '
                      f'matches={metrics["n_inliers"]}/{metrics["n_matches"]} '
                      f'residual={metrics["mean_residual"]:.1f} '
                      f'consistency={metrics["consistency_ratio"]:.2f}')
        else:
            detail = f'no transform ({metrics.get("reason", "SIFT found no usable matches")})'
        lines.append(f'      z {attempt["z_ref"]}/{attempt["z_mov"]} @ scale {attempt["scale"]}: '
                     f'{detail} -> {"valid" if attempt["valid"] else "invalid"}')
    return '\n'.join(lines)


def compute_alignment_path(datasets,
                           z_offsets,
                           target_resolution):
    '''Compute alignment paths between overlapping datasets using SIFT feature matching.

    Analyzes Z-overlap between datasets and builds a graph where edges represent
    valid alignment transitions (verified by SIFT matching at slice boundaries).
    Returns paths from a root dataset (one with no overlap) to all other datasets.

    Each transition is tested with the dataset masks applied, over up to N_CANDIDATE_SLICES
    slice pairs stepping inward from the boundary, and at every scale in SIFT_SCALES. The
    metrics of every attempt are logged, and are reported in full if the graph ends up
    disconnected.

    Args:
        datasets (list): List of tensorstore.TensorStore objects to align.
        z_offsets (np.ndarray): Array of shape (N, 3) with [z, y, x] voxel offsets for each dataset.
        target_resolution (int or list): Target resolution in nm for SIFT matching. If int, used for both Y and X.

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

    datasets_nomask, names = _datasets_without_masks(datasets)

    if len(datasets_nomask) == 1:
        root_node = names[0]
        ds_bounds = {root_node: (0, datasets_nomask[0].shape[0])}
        return root_node, [[root_node]], [False], ds_bounds

    # Masks are excluded from the dataset list, so the companion stores are reopened here.
    # SIFT keypoints found on the black canvas around the tissue are not informative, and
    # the alignment itself (align_z.transform) also matches slices with masks applied.
    dataset_masks = [open_store(os.path.abspath(d.kvstore.path) + '_mask',
                                mode='r', dtype=ts.bool, allow_missing=True)
                     for d in datasets_nomask]

    def _get_slice(idx, z, reverse, target_resolution=target_resolution):
        '''Read a boundary slice and its mask, resampled to the target resolution.'''
        store = datasets_nomask[idx]
        resolution = get_store_attributes(store)['resolution']
        assert resolution[-2] == resolution[-1], 'Resolution must be the same in X and Y'
        assert target_resolution[-2] == target_resolution[-1], 'Target resolution must be the same in X and Y'
        target_scale = resolution[-1]/target_resolution[-1]

        img, z_found = find_ref_slice(store, z, reverse=reverse)
        img = resample(img, target_scale)

        if dataset_masks[idx] is not None:
            mask = resample(dataset_masks[idx][z_found].read().result(), target_scale)
        else:
            mask = compute_greyscale_mask(img, downsample_factor=10)

        if mask.shape != img.shape or not mask.any():
            # An empty or ill-shaped mask would leave SIFT with no keypoints at all
            mask = None
        return img, mask, z_found

    def _test_transition(u, v, local_z_u, local_z_v):
        '''Test whether the boundary of dataset u can be matched to the boundary of dataset v.

        Steps inward from the boundary over several candidate slice pairs and tries every
        SIFT scale, so that a single damaged or low contrast slice does not invalidate an
        otherwise valid transition.

        Returns:
            tuple: (edge attributes dict or None if no pair matched, list of attempt records)
        '''
        attempts = []
        tried = set()
        for step in range(N_CANDIDATE_SLICES):
            try:
                ref, ref_mask, z_ref = _get_slice(u, local_z_u - step, reverse=True)
                mov, mov_mask, z_mov = _get_slice(v, local_z_v + step, reverse=False)
            except IndexError as e:
                attempts.append({'step': step, 'error': str(e)})
                break

            if (z_ref, z_mov) in tried:
                # find_ref_slice skipped back to a pair we already tested
                continue
            tried.add((z_ref, z_mov))

            for sift_scale in SIFT_SCALES:
                M, out_shape, ref_offset, valid_estimate, metrics = estimate_transform_sift(
                    ref.copy(), mov.copy(), scale=sift_scale,
                    ref_mask=ref_mask, mov_mask=mov_mask, refine_estimate=False)

                attempts.append({'step': step, 'z_ref': int(z_ref), 'z_mov': int(z_mov),
                                 'scale': sift_scale, 'valid': bool(valid_estimate),
                                 'metrics': metrics})
                if valid_estimate:
                    return {'M': M, 'out_shape': out_shape, 'ref_offset': ref_offset,
                            'valid_estimate': True}, attempts
        return None, attempts


    # Find all ranges over which there is overlap
    df = _occupancy_table(datasets_nomask, names, z_offsets)

    # Find first dataset alone at its own z level
    root_datasets = df.ds_indices[df.ds_indices.apply(len) == 1] 

    if len(root_datasets) == 0:
        raise RuntimeError('No potential root dataset was found: no dataset with no overlap along Z.')

    root_node_idx = root_datasets.iloc[0][0]
    root_node = names[root_node_idx]

    # Compute valid alignment paths
    G = nx.Graph()
    G.add_nodes_from(np.unique(np.concatenate(df.ds_indices)).tolist())
    transition_log = []
    grouped = df.groupby('group')
    for g, curr_group in grouped:
        if g == df.group.max():
            break
        next_group = grouped.get_group(g+1)
        for u in curr_group.ds_indices.iloc[0]:
            for v in next_group.ds_indices.iloc[0]:
                if u == v:
                    continue  # same dataset spans both groups — no inter-dataset transition needed
                if G.has_edge(u, v):
                    continue  # already validated at an earlier group boundary
                # Check for match at the boundary of the relevant range
                edge, attempts = _test_transition(u, v,
                                                  curr_group.z.max() - z_offsets[u, 0],
                                                  next_group.z.min() - z_offsets[v, 0])
                transition_log.append({'u': u, 'v': v,
                                       'z_boundary': int(curr_group.z.max()),
                                       'accepted': edge is not None,
                                       'attempts': attempts})
                logging.info(format_transition(transition_log[-1], names))

                if edge is not None:
                    # Keep track of everything, mostly for debugging
                    G.add_edge(u, v, **edge)

    if not nx.is_connected(G):
        # Some datasets are disconnected from the main alignment path
        x = [[names[i] for i in cc] for cc in nx.connected_components(G)]
        logging.error('Alignment graph is disconnected. Every transition that was tested:')
        for record in transition_log:
            logging.error(format_transition(record, names))

        rejected = [f'{names[r["u"]]} -> {names[r["v"]]} (global z={r["z_boundary"]})'
                    for r in transition_log if not r['accepted']]
        rejected = '\n    '.join(rejected) if rejected else 'none (no transition was even tested)'
        raise RuntimeError(f'Some datasets are isolated: \n{x}\n'
                           f'Rejected transitions:\n    {rejected}\n'
                           f'See the log above for the SIFT metrics of every attempt.')

    paths = extract_paths_from_root(G, root_node_idx)
    if not paths:
        # Root is the only effective dataset after graph construction (e.g. all others were
        # filtered as fused sub-datasets).  Treat it as a single-stack project.
        logging.warning(f'No alignment paths found from root "{root_node}"; treating it as the sole dataset.')
        ds_bounds = {root_node: (0, datasets_nomask[root_node_idx].shape[0])}
        return root_node, [[root_node]], [False], ds_bounds

    reverse_z = [bool(z_offsets[p[0], 0] > z_offsets[p[-1], 0]) for p in paths]
    paths = [[names[i] for i in p] for p in paths]

    # Datasets will need to be bounded to not re-use fused images
    ds_bounds = _dataset_bounds(df, names, z_offsets)
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

    mask_stores = {}

    def _masked_slice(store, reverse):
        '''Read the first or last non-empty slice of a store, with its mask.'''
        img, z = find_ref_slice(store, reverse=reverse)
        path = os.path.abspath(store.kvstore.path)
        if path not in mask_stores:
            # A path is walked once per alignment path it belongs to, so cache the lookup
            mask_stores[path] = open_store(path + '_mask', mode='r', dtype=ts.bool,
                                           allow_missing=True)
        mask_store = mask_stores[path]
        if mask_store is not None:
            mask = mask_store[z].read().result()
        else:
            mask = compute_greyscale_mask(img, downsample_factor=10)

        if mask.shape != img.shape or not mask.any():
            mask = None
        return img, mask

    global_offset = np.array([0,0])
    pbar = tqdm(position=0,
                desc='Computing global offset without reference',
                dynamic_ncols=True,
                leave=True,
                total=sum([len(p) for p in paths]))
    for path in paths:
        path_offset = np.array([0,0])

        prev, prev_mask = _masked_slice(datasets[path[0]], reverse=True)
        pbar.update(1)
        for stack_name in path[1:]:
            ds_curr = datasets[stack_name]
            curr, curr_mask = _masked_slice(ds_curr, reverse=False)

            M, output_shape, prev_offset, valid_estimate, _ = estimate_transform_sift(
                prev, curr, scale=0.1, ref_mask=prev_mask, mov_mask=curr_mask, refine_estimate=True)

            if not valid_estimate or M is None:
                # The transition was accepted when building the graph but could not be
                # matched here. Skip its contribution rather than dereferencing None, and
                # carry on from the next dataset in its own frame of reference.
                logging.warning(f'Could not estimate the offset between "{stack_name}" and its '
                                f'predecessor in path {path}. Its drift is not accounted for in '
                                f'the canvas padding.')
                prev, prev_mask = _masked_slice(ds_curr, reverse=True)
            else:
                prev = warpAffine(_masked_slice(ds_curr, reverse=True)[0], M, output_shape[::-1])
                prev_mask = None  # The mask no longer matches the warped image
                path_offset += prev_offset
            pbar.update(1)

        global_offset = np.min([global_offset, path_offset], axis=0)

    pbar.close()
    return np.abs(global_offset)


def determine_initial_offset_ref(
        datasets, 
        z_offsets, 
        reference_path, 
        reference_offset, 
        yx_target_resolution,
        no_resample=True
        ):

    from .align_z import PAD_OVERLAP
    from ..arrays.overlap import get_overlap_ref
    # Verify that each dataset overlaps the reference and derive the canvas size
    # from the maximum warped extent across all datasets.
    reference = open_store(reference_path, mode='r', dtype=ts.uint8)
    ref_resolution = get_store_attributes(reference)['resolution']

    dataset_names = []
    global_offset = np.array([0,0])
    ref_bboxes = []
    
    for i, dataset in tqdm(enumerate(datasets),
                           position=0,
                           desc='Computing global offset with reference',
                           dynamic_ncols=True,
                           leave=True,
                           total=len(datasets)):
        z_offset_val = int(z_offsets[i, 0])
        dataset_name = os.path.basename(os.path.abspath(dataset.kvstore.path))
        dataset_names.append(dataset_name)

        # Get reference image and resample it to the dataset resolution
        ref_scale = ref_resolution[-1] / yx_target_resolution
        ref_img, ref_z = find_ref_slice(reference, z_offset_val + reference_offset)
        if not no_resample:
            ref_img = resample(ref_img, ref_scale)

        reference_mask_path = os.path.abspath(reference.kvstore.path) + '_mask'
        if os.path.exists(reference_mask_path):
            reference_mask = open_store(reference_mask_path, 'r')
            ref_mask = reference_mask[ref_z].read().result()
            if not no_resample:
                ref_mask = resample(ref_mask, ref_scale)
        else:
            ref_mask = None

        # Get test image and mask
        resolution = get_store_attributes(dataset)['resolution']
        target_scale = resolution[-1] / yx_target_resolution
        z_ds = dataset.domain.inclusive_min[0]
        test_img = dataset[z_ds].read().result()
        if not no_resample:
            test_img = resample(test_img, target_scale)
        dataset_mask = open_store(os.path.abspath(dataset.kvstore.path) + '_mask', 'r')
        if dataset_mask is not None:
            test_mask = dataset_mask[z_ds].read().result()
            if not no_resample:
                test_mask = resample(test_mask, target_scale)
        else:
            test_mask = None
        
        # Test overlap by computing the bounding box.
        # pad_overlap is a dilation in pixels, so when matching on the native images it has
        # to be expressed in native pixels to survive the conversion back to target space.
        pad_overlap = int(round(PAD_OVERLAP / ref_scale)) if no_resample else PAD_OVERLAP
        _, _, bbox_ref, sift_res = get_overlap_ref(ref_img,
                                                    test_img,
                                                    ref_mask=ref_mask,
                                                    mov_mask=test_mask,
                                                    bbox_ref=None,
                                                    pad_overlap=pad_overlap,
                                                    return_sift=True)
        _, _, xy_offset, valid_estimate, sift_stats = sift_res
        if not valid_estimate:
            raise RuntimeError(
                f'Overlap with reference dataset could not be found for dataset: {dataset.kvstore.path}\n'
                f'  reference slice   : {ref_z} (z_offset {z_offset_val} + reference_offset {reference_offset})\n'
                f'  dataset slice     : {z_ds}\n'
                f'  no_resample       : {no_resample}\n'
                f'  reference shape   : {ref_img.shape} at {ref_resolution[-1]}nm '
                f'(resample ratio to target would be {ref_scale:.4g})\n'
                f'  dataset shape     : {test_img.shape} at {resolution[-1]}nm '
                f'(resample ratio to target would be {target_scale:.4g})\n'
                f'  masks used        : reference={ref_mask is not None}, dataset={test_mask is not None}\n'
                f'  SIFT scales tried : {SIFT_SCALES} (applied on top of the resampling above)\n'
                f'  SIFT metrics      : {sift_stats}')
        if no_resample:
            # Matching ran on the native images, so bbox_ref and xy_offset came back in
            # native reference pixels. Everything downstream (the bbox crop in
            # align_z._compute_flow, the canvas origin) works in target-resolution
            # pixels, so convert them here. No-op when ref_scale is 1.
            bbox_ref = (np.array(bbox_ref) * ref_scale).astype(int).tolist()
            xy_offset = (np.array(xy_offset) * ref_scale).astype(int)

        global_offset = np.min([global_offset, np.abs(xy_offset[::-1])], axis=0)
        ref_bboxes.append(bbox_ref)

    return np.abs(global_offset), ref_bboxes