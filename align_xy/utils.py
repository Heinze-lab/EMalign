''' Utilities for alignment of tilesets in XY plane.'''

import cv2
import numpy as np
import warnings

from ..arrays.utils import compute_laplacian_var_diff
from ..arrays.overlap import get_overlap


def mask_to_mesh(mask, yx_shape):
        
    # Create coordinate arrays for sampling
    y_coords = np.linspace(0, mask.shape[0] - 1, yx_shape[0]).astype(int)
    x_coords = np.linspace(0, mask.shape[1] - 1, yx_shape[1]).astype(int)
    
    # Sample the mask at these coordinates
    x = mask[np.ix_(y_coords, x_coords)]
    x = np.repeat(x[None, None, ...], 2, axis=0).astype(np.float32) - 1 
    x[x<0] = np.nan
    return x_target


def check_stitch(warped_tiles, margin):

    tile_space = (np.array(list(warped_tiles.keys()))[:,1].max()+1, 
                  np.array(list(warped_tiles.keys()))[:,0].max()+1)
    
    overlap_scores = []
    for x in range(0, tile_space[1] - 1):
        for y in range(0, tile_space[0]):
            x1,y1,left = warped_tiles[(x,y)] 
            x2,y2,right = warped_tiles[(x+1,y)]

            offset = np.array([x1-x2, y1-y2])
            
            overlap1, overlap2 = get_overlap(left, right, offset, 0, homogenize_shapes=True)

            if overlap1.size == 0 or overlap2.size == 0:
                overlap_score = 0
            else:
                try:
                    overlap_score = compute_laplacian_var_diff(overlap1[:, :-margin], 
                                                               overlap2[:, margin:])
                except cv2.error as e:
                    if e.err == '!_src.empty()':
                        overlap_score = 0
                        warnings.warn('Empty overlap. There may not be overlap between tiles. Overlap score set to 0.')
                    else:
                        raise e
            overlap_scores.append(overlap_score)

    for y in range(0, tile_space[0] - 1):
        for x in range(0, tile_space[1]):
            x1,y1,bot = warped_tiles[(x,y)] 
            x2,y2,top = warped_tiles[(x,y+1)] 
            
            offset = np.array([x1-x2, y1-y2])
            
            overlap1, overlap2 = get_overlap(bot, top, offset, 0, homogenize_shapes=True)

            if overlap1.size == 0 or overlap2.size == 0:
                overlap_score = 0
            else:
                try:
                    overlap_score = compute_laplacian_var_diff(overlap1[:-margin, :], 
                                                               overlap2[margin:, :])
                except cv2.error as e:
                    if e.err == '!_src.empty()':
                        overlap_score = 0
                        warnings.warn('Empty overlap. There may not be overlap between tiles. Overlap score set to 0.')
                    else:
                        raise e
            overlap_scores.append(overlap_score)
    return overlap_scores
