import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import tensorstore as ts
from tifffile import imwrite
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)


"""
Convert a final aligned zarr container to a lossless TIFF image series.

Each z slice is written as one uncompressed .tiff file. The source dtype and
pixel values are preserved exactly; no scaling, normalization, compression, or
other lossy processing is applied.
"""


def _normalize_slice_range(slice_range, z_size):
    """Validate and normalize an optional [start, end] slice range."""
    if slice_range is None:
        return [0, z_size]
    if len(slice_range) == 1:
        return [slice_range[0], z_size]
    if len(slice_range) == 2:
        return slice_range
    raise ValueError('Slice range must have 1 (start) or 2 elements ([start, end]).')


def _validate_slice_range(slice_range, z_size):
    """Validate that a slice range is inside the zarr volume."""
    start, end = slice_range
    if start < 0 or end < 0:
        raise ValueError('Slice range values must be non-negative.')
    if end <= start:
        raise ValueError('Invalid slice range: upper bound must exceed lower bound.')
    if start >= z_size or end > z_size:
        raise ValueError(f'Slice range {slice_range} is outside z dimension 0:{z_size}.')


def _format_tiff_path(output_path, prefix, z_index, digits):
    """Build the output path for a single TIFF slice."""
    return os.path.join(output_path, f'{prefix}{z_index:0{digits}d}.tiff')


def _write_tiff_slice(dataset, z_index, output_path, prefix, digits, overwrite):
    """Read one zarr slice and write it as an uncompressed TIFF."""
    output_file = _format_tiff_path(output_path, prefix, z_index, digits)
    if os.path.exists(output_file) and not overwrite:
        raise FileExistsError(
            f'Output file already exists: {output_file}. Use --overwrite to replace it.'
        )

    data = dataset[z_index].read().result()
    data = np.asarray(data)
    imwrite(output_file, data, compression=None)
    return output_file


def zarr_to_tiff_series(dataset_path,
                        output_path,
                        slice_range=None,
                        prefix='slice_',
                        digits=6,
                        num_threads=1,
                        overwrite=False):
    """
    Convert a zarr dataset into an uncompressed TIFF image series.

    Args:
        dataset_path: Path to the aligned zarr dataset.
        output_path: Directory where .tiff files will be written.
        slice_range: Optional [start, end] z slice interval. End is exclusive.
        prefix: Filename prefix for each TIFF.
        digits: Zero-padding width for z indices in filenames.
        num_threads: Number of concurrent slice-writing threads.
        overwrite: Whether to overwrite existing output TIFF files.
    """
    os.makedirs(output_path, exist_ok=True)

    dataset = ts.open({
        'driver': 'zarr',
        'kvstore': {
            'driver': 'file',
            'path': dataset_path,
        }
    }, read=True).result()

    if len(dataset.shape) < 3:
        raise ValueError(
            f'Expected at least a 3D zarr dataset with z as the first axis, got shape {dataset.shape}.'
        )

    slice_range = _normalize_slice_range(slice_range, dataset.shape[0])
    _validate_slice_range(slice_range, dataset.shape[0])
    start, end = slice_range

    logging.info(f'Input zarr: {dataset_path}')
    logging.info(f'Input shape: {dataset.shape}')
    logging.info(f'Input dtype: {dataset.dtype}')
    logging.info(f'Writing slices {start}:{end} as uncompressed TIFF image series')
    logging.info(f'Output directory: {output_path}')

    z_indices = list(range(start, end))
    if num_threads < 1:
        raise ValueError('Number of threads must be at least 1.')

    if num_threads == 1:
        for z_index in tqdm(z_indices, desc='Writing TIFF slices', unit='slices', dynamic_ncols=True):
            _write_tiff_slice(dataset, z_index, output_path, prefix, digits, overwrite)
    else:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(
                    _write_tiff_slice,
                    dataset,
                    z_index,
                    output_path,
                    prefix,
                    digits,
                    overwrite,
                )
                for z_index in z_indices
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc='Writing TIFF slices', unit='slices', dynamic_ncols=True):
                future.result()

    logging.info('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert a final aligned zarr dataset to an uncompressed TIFF image series.'
    )
    parser.add_argument('-i', '--input',
                        metavar='DATASET_PATH',
                        dest='dataset_path',
                        required=True,
                        type=str,
                        help='Path to the aligned zarr dataset.')
    parser.add_argument('-o', '--output',
                        metavar='OUTPUT_PATH',
                        dest='output_path',
                        required=True,
                        type=str,
                        help='Output directory for the TIFF image series.')
    parser.add_argument('-z', '--slice-range',
                        metavar='SLICE_RANGE',
                        dest='slice_range',
                        nargs='+',
                        type=int,
                        default=None,
                        help='Slice range [start end], where end is exclusive (default: all slices).')
    parser.add_argument('--prefix',
                        metavar='PREFIX',
                        dest='prefix',
                        type=str,
                        default='slice_',
                        help='Filename prefix for output TIFFs (default: slice_).')
    parser.add_argument('--digits',
                        metavar='DIGITS',
                        dest='digits',
                        type=int,
                        default=6,
                        help='Zero-padding width for output slice indices (default: 6).')
    parser.add_argument('-c', '--cores',
                        metavar='CORES',
                        dest='num_threads',
                        type=int,
                        default=1,
                        help='Number of concurrent writer threads (default: 1).')
    parser.add_argument('--overwrite',
                        dest='overwrite',
                        action='store_true',
                        help='Overwrite existing TIFF files in the output directory.')

    args = parser.parse_args()
    zarr_to_tiff_series(**vars(args))
