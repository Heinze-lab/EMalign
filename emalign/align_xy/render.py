import numpy as np

from sofima import warp

from emalign.io.process.mask import mask_to_bbox

from .utils import check_stitch
from ..io.store import write_data


def resolve_img_q_fun(spec):
    '''Resolve a tile-quality function from a config value.

    Used to turn a JSON-friendly config entry into the img_q_fun callable consumed by
    get_render_order / align_stack_xy. The callable takes an image and its mask and returns
    a scalar that is higher for higher quality/sharpness.

    Args:
        spec: One of:
            - None, '', 'position' or 'default': no quality function (position-based order, the default).
            - 'laplacian', 'auto' or 'quality': Laplacian-variance sharpness.
            - 'sobel': mean Sobel gradient magnitude.
            - 'sharpness': composite of Laplacian variance, Sobel mean and gradient magnitude
              (matches the example in stitch_offgrid.stitch_images).
            - a callable: returned unchanged.

    Returns:
        A callable img_q_fun, or None for the default position-based order.
    '''
    if spec is None or callable(spec):
        return spec

    from ..arrays.utils import compute_laplacian_var, compute_sobel_mean, compute_grad_mag

    key = str(spec).lower()
    funcs = {
        '': None,
        'position': None,
        'default': None,
        'laplacian': lambda img, m: compute_laplacian_var(img, m),
        'auto': lambda img, m: compute_laplacian_var(img, m),
        'quality': lambda img, m: compute_laplacian_var(img, m),
        'sobel': lambda img, m: compute_sobel_mean(img, m),
        'sharpness': lambda img, m: compute_laplacian_var(img, m) * 0.5
                                    + compute_sobel_mean(img, m)
                                    + compute_grad_mag(img, m) * 100,
    }
    if key not in funcs:
        raise ValueError(f"Unknown img_on_top quality metric: {spec!r}. "
                         f"Expected one of {sorted(funcs)} or a callable.")
    return funcs[key]


def get_render_order(tile_map, tile_masks=None, img_q_fun=None):
    '''Determine the order in which tiles are rendered. Tiles rendered last end up on top.

    By default (img_q_fun is None), tiles are ordered by their grid position so that the
    first tiles acquired are rendered last (on top), because they are sharper.

    If img_q_fun is provided, tiles are ordered by ascending quality so that the
    highest-quality tile is rendered last (on top). This mirrors the automatic
    'image on top' selection used in stitch_offgrid.stitch_images.

    Args:
        tile_map (dict of `np.ndarray`): Dictionary from [x,y] tile position to [y,x] image.
        tile_masks (dict of `np.ndarray`, optional): Dictionary from [x,y] tile position to
            [y,x] boolean masks corresponding to tile_map. Defaults to None.
        img_q_fun (callable, optional): Function taking an image and its mask, returning a
            scalar that is higher for higher quality/sharpness. If None, the default
            position-based order is used.
            e.g.: img_q_fun = lambda img, m: compute_laplacian_var(img, m)

    Returns:
        list: Tile position keys ordered from rendered-first (bottom) to rendered-last (top).
    '''
    if img_q_fun is None:
        return sorted(tile_map)

    masks = tile_masks or {}

    def quality(k):
        # tile_masks may be uint8 (np.ones_like the image); the quality functions index with
        # the mask, so it must be boolean to act as a selection rather than fancy indexing.
        mask = masks.get(k)
        if mask is not None:
            mask = mask.astype(bool)
        return img_q_fun(tile_map[k], mask)

    return sorted(tile_map, key=quality)


def render_slice_xy(destination,
                    z,
                    tile_map,
                    meshes,
                    stride,
                    tile_masks=None,
                    parallelism=1,
                    margin=50,
                    dest_mask=None,
                    return_render=False,
                    resize_canvas=True,
                    min_stitch_score=0,
                    **kwargs):
    '''Render an aligned image from a tile map.

    Use a tile_map and corresponding meshes to produce an aligned image and mask. 
    Overlaps are assessed with check_stitch to produce a stitch_score that will be logged to find flawed slices.
    The score is based on a laplacian filter, between 0 (no similarity) and 1 (exact match).

    Args:
        destination (`tensorstore.TensorStore`): Zarr store where to write aligned slice.
        z (int): Z index at which to write the slice (axis at first position).
        tile_map (dict of `np.ndarray`): Dictionary from [x,y] tile position to [y,x] image.
        meshes (dict of `np.ndarray`): Dictionary from [x,y] tile position to [2, z, x, y] array of mesh positions. Order of keys determines the order of render.
        stride (int): Step used to determine mesh node positions.
        tile_masks (dict of `np.ndarray`, optional): Dictionary from [x,y] tile position to [y,x] boolean masks corresponding to tile_map. Defaults to None.
        parallelism (int, optional): Number of threads used by warp.render_tiles to warp tiles in parallel (max one thread per tile). Defaults to 1.
        margin (int, optional): Number of pixels cropped from each tile's boundaries to remove artifacts from deformation. Defaults to 50.
        dest_mask (_type_, optional): Zarr store where to write aligned slice's mask. Defaults to None.
        return_render (bool, optional): Whether to return the aligned image rather than writing it. Defaults to False.
        resize_canvas (bool, optional): Whether the image to the size of a bounding box defined by the mask. Defaults to True.
        **kwargs (optional): Additional arguments passed to warp.render_tiles. 
            e.g.: margin_overrides provides specific margins per direction per tile.

    Returns:
        int: 
            If return_render == False (Default): stitch score describing how well overlaps match, between 0 and 1 as defined by check_stitch. 
            If return_render == True: tuple of: aligned image, stitch score.
    '''

    if len(tile_map) > 1:
        # warp.render_tiles only uses workers to distribute tiles
        parallelism = min(len(tile_map.keys()), parallelism)

        # Render stitched image
        stitched, mask, warped_tiles = warp.render_tiles(tile_map, meshes, 
                                                    tile_masks=tile_masks, 
                                                    parallelism=parallelism, 
                                                    stride=(stride, stride), 
                                                    return_warped_tiles=True,
                                                    margin=margin,
                                                    **kwargs)
        # Evaluate overlap
        stitch_score = check_stitch(warped_tiles, margin)
    else:
        stitched = list(tile_map.values())[0]
        mask = np.ones_like(list(tile_map.values())[0]).astype(bool)
        stitch_score = 1
    
    if resize_canvas:
        y1,y2,x1,x2 = mask_to_bbox(mask)
        stitched = stitched[y1:y2,x1:x2]
        mask = mask[y1:y2,x1:x2]

    if return_render:
        return stitched, stitch_score
    elif np.min(stitch_score) > min_stitch_score:
        # Stitch good enough, write data
        destination, _ = write_data(destination, stitched, z)

        if dest_mask is not None:
            dest_mask, _ = write_data(dest_mask, mask, z)
            return destination, dest_mask, stitch_score
        return destination, stitch_score
    else:
        # Bad stitch, don't write data
        if dest_mask is not None:
            return destination, dest_mask, stitch_score
        return destination, stitch_score