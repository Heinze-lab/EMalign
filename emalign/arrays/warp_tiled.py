'''Tiled cv2.remap for images larger than OpenCV's 32767-pixel limit.

cv2.remap asserts that both source and destination are < SHRT_MAX (32767) in
each dimension, because the interpolation tables are computed in 16-bit
fixed point. The workaround is to remap in blocks: for each output tile,
find the bounding box of source pixels it needs, slice only that region,
rebase the coordinate maps onto it, and call remap on the (small) pair.
Only the tile's source footprint is ever materialised.
'''

from __future__ import annotations

import cv2
import numpy as np


SHRT_MAX = 32767


def remap_tiled(
    src,
    map_x: np.ndarray,
    map_y: np.ndarray,
    tile: int = 8192,
    margin: int = 2,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value=0,
    out: np.ndarray | None = None,
) -> np.ndarray:
    '''Remap `src` onto the grid given by (map_x, map_y), tile by tile.

    Parameters
    ----------
    src
        2D (H, W) or 3D (H, W, C) array-like. Sliced lazily.
    map_x, map_y
        float32 arrays of the *output* shape, holding absolute source
        coordinates. Non-finite entries (NaN, common in SOFIMA-style
        flow fields) are treated as 'no data' and filled with border_value.
    tile
        Output tile edge length in pixels. Must be < SHRT_MAX; 4096-8192
        is a reasonable range.
    margin
        Extra pixels added around each tile's source bounding box so the
        interpolation kernel has support at the edges. 2 is enough for
        linear, use 3-4 for cubic/Lanczos.
    out
        Optional preallocated output array (e.g. a memmap or zarr array)
        of shape map_x.shape (+ (C,)). Allocated if None.

    Returns
    -------
    The warped image, same dtype as `src`.
    '''
    if map_x.shape != map_y.shape:
        raise ValueError(f'map shape mismatch: {map_x.shape} vs {map_y.shape}')
    if map_x.ndim != 2:
        raise ValueError('maps must be 2D (output grid)')
    if not 0 < tile < SHRT_MAX:
        raise ValueError(f'tile must be in (0, {SHRT_MAX})')

    map_x = np.asarray(map_x, dtype=np.float32)
    map_y = np.asarray(map_y, dtype=np.float32)

    src_h, src_w = src.shape[:2]
    n_chan = src.shape[2] if len(src.shape) == 3 else None
    out_h, out_w = map_x.shape

    if out is None:
        shape = (out_h, out_w) if n_chan is None else (out_h, out_w, n_chan)
        out = np.empty(shape, dtype=src.dtype)

    for y0 in range(0, out_h, tile):
        y1 = min(y0 + tile, out_h)
        for x0 in range(0, out_w, tile):
            x1 = min(x0 + tile, out_w)

            mx = map_x[y0:y1, x0:x1]
            my = map_y[y0:y1, x0:x1]

            valid = np.isfinite(mx) & np.isfinite(my)
            if not valid.any():
                out[y0:y1, x0:x1] = border_value
                continue

            # Source footprint needed by this output tile.
            sx0 = int(np.floor(mx[valid].min())) - margin
            sx1 = int(np.ceil(mx[valid].max())) + margin + 1
            sy0 = int(np.floor(my[valid].min())) - margin
            sy1 = int(np.ceil(my[valid].max())) + margin + 1

            # Clip to the source, but keep the *unclipped* origin so that
            # out-of-bounds coordinates stay out of bounds after rebasing
            # and are handled by border_mode.
            cx0, cx1 = max(sx0, 0), min(sx1, src_w)
            cy0, cy1 = max(sy0, 0), min(sy1, src_h)
            if cx1 <= cx0 or cy1 <= cy0:
                out[y0:y1, x0:x1] = border_value
                continue

            if (cx1 - cx0) >= SHRT_MAX or (cy1 - cy0) >= SHRT_MAX:
                raise ValueError(
                    f'tile at ({y0}, {x0}) needs a source region of '
                    f'{cy1 - cy0}x{cx1 - cx0} px, which still exceeds the '
                    f'OpenCV limit. Reduce `tile`, or the displacement field '
                    f'is not local enough for this approach.'
                )

            block = np.ascontiguousarray(src[cy0:cy1, cx0:cx1])

            # Rebase coords onto the block. Doing the subtraction in float64
            # then casting keeps precision for large absolute coordinates:
            # float32 only has ~24 bits of mantissa, so coordinates beyond
            # ~2e7 lose sub-pixel resolution.
            bx = (mx.astype(np.float64) - cx0).astype(np.float32)
            by = (my.astype(np.float64) - cy0).astype(np.float32)

            # Send invalid pixels far outside the block so border_mode wins.
            bad = ~valid
            if bad.any():
                bx = np.where(bad, np.float32(-1e4), bx)
                by = np.where(bad, np.float32(-1e4), by)

            out[y0:y1, x0:x1] = cv2.remap(
                block,
                bx,
                by,
                interpolation=interpolation,
                borderMode=border_mode,
                borderValue=border_value,
            )

    return out


def warp_affine_tiled(src, M: np.ndarray, dsize: tuple[int, int], **kwargs):
    '''cv2.warpAffine equivalent for oversized images.

    `M` is the usual 2x3 forward matrix (src -> dst); it is inverted here to
    build the backward maps that remap needs. `dsize` is (width, height).
    '''
    w, h = dsize
    M_inv = cv2.invertAffineTransform(np.asarray(M, dtype=np.float64))
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing='ij',
    )
    map_x = M_inv[0, 0] * xx + M_inv[0, 1] * yy + M_inv[0, 2]
    map_y = M_inv[1, 0] * xx + M_inv[1, 1] * yy + M_inv[1, 2]
    return remap_tiled(src, map_x.astype(np.float32), map_y.astype(np.float32), **kwargs)

def warp_affine_auto(data, M, output_shape):
    '''Choose between the two available functions for warping based on input and output size'''
    
    is_mask = data.dtype == bool
    if is_mask:
        data = data.astype(np.uint8)

    M = np.asarray(M, dtype=np.float64)[:2]
    dsize = (int(output_shape[1]), int(output_shape[0]))
    if max(*data.shape, *dsize) > SHRT_MAX:
        data = warp_affine_tiled(data, M, dsize)
    else:
        data = cv2.warpAffine(data, M, dsize)

    return data.astype(bool) if is_mask else data
