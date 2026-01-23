'''
Utilities for finding and reading files produced by the ThermoFisher MAPs software.
'''


import logging
import os
import re

from concurrent import futures
from glob import glob
from tqdm import tqdm

FILE_EXT = '.tif'

def get_tileset_resolution(tileset_path):
    '''
    Find resolution of a tileset by reading metadata from the stack directory.

    For VolumeScope, this reads a .info file and extracts pixel size from a line
    containing 'resolution' or 'pixel size' (e.g., 'Pixel Size: 10 10 nm').

    Args:
        tileset_path: Path to a directory containing tile images and metadata.

    Returns:
        Tuple of (tileset_path, (y_res, x_res)) if resolution found, else None.
        Resolution is in the units specified by the metadata file (typically nm).
        Example: ('/path/to/stack/', (10, 10))

    Notes for implementing other backends:
        - Must return (tileset_path, (y_resolution, x_resolution)) or None
        - Resolution values should be integers in consistent units (e.g., nm)
        - Return None if resolution cannot be determined (will be skipped)
        - The path is returned to allow filtering tilesets by resolution
    '''
    info=None
    with os.scandir(tileset_path) as entries:
        for entry in entries:
            if entry.name.endswith('.info'):
                info = entry.path
                break

    if info is None:
        return None

    with open(info, 'r') as f:
        content = f.readlines()

    resolution = None
    for line in content:
        if 'resolution' in line.lower() or 'pixel size' in line.lower():
            matches = re.findall(r'\d+', line)
            if len(matches) >= 2:
                resolution = tuple(map(int, matches[:2]))
                break

    # Fallback to line 5 for backward compatibility with existing .info files
    if resolution is None:
        if len(content) > 5:
            matches = re.findall(r'\d+', content[5])
            if len(matches) >= 2:
                resolution = tuple(map(int, matches[:2]))
            else:
                logging.warning(f'Could not determine resolution from .info file in {tileset_path}')
                return None
        else:
            logging.warning(f'Could not determine resolution from .info file in {tileset_path} (insufficient lines)')
            return None

    return (tileset_path, resolution)


def get_tilesets(main_dir, resolution, dir_pattern, num_workers):
    '''
    Find all tileset directories matching a given resolution and naming pattern.

    For VolumeScope, searches subdirectories of main_dir for .info files and
    filters by resolution and directory name pattern.

    Args:
        main_dir: Parent directory containing tileset subdirectories.
        resolution: Target resolution as (y_res, x_res) tuple (e.g., (10, 10)).
        dir_pattern: List of substrings that must appear in directory names
            to be included (e.g., ['Sample1', 'ROI']).
        num_workers: Number of parallel threads for scanning directories.

    Returns:
        Sorted list of absolute paths to matching tileset directories.
        Example: ['/data/Sample1_ROI1/', '/data/Sample1_ROI2/']

    Notes for implementing other backends:
        - Must return a list of directory paths containing tile images
        - Each directory should contain all tiles for one tileset/stack
        - Paths should be absolute and sorted for reproducibility
        - Filter by resolution to handle multi-resolution acquisitions
    '''
    # Get all directories containing tilesets that are present in main_dir
    tileset_dirs = glob(os.path.join(main_dir, '*', ''))

    stack_list = []
    # Find the ones with the right resolution
    with futures.ThreadPoolExecutor(num_workers) as tpe:
        fs = []
        for d in tileset_dirs:
            fs.append(tpe.submit(get_tileset_resolution, d))

        for f in tqdm(futures.as_completed(fs), total=len(fs), desc=f'Looking for resolution: {resolution}', leave=False):
            result = f.result()
            if result is None:
                continue
            # Find the directory with the right pattern if relevant
            for d in dir_pattern:
                if d in result[0].split('/')[-2] and result[1] == tuple(resolution):
                    stack_list.append(result[0])
    return sorted(stack_list)


def parse_yx_pos_from_name(n):
    '''
    Extract tile grid position (y, x) from a tile filename.

    For VolumeScope/MAPs, filenames follow the pattern: Tile_XX-YY_sZZZZ.tif
    where XX is the x-position and YY is the y-position (1-indexed).

    Args:
        n: Tile filename or full path (e.g., 'Tile_03-02_s0001.tif')

    Returns:
        Tuple (y, x) as 0-indexed integers representing the tile's position
        in the grid. Example: 'Tile_03-02_s0001.tif' -> (1, 2)

    Notes for implementing other backends:
        - Must return a tuple of (y, x) as 0-indexed integers
        - The tuple is used as a dictionary key to identify tiles
        - All tiles in a slice should have unique (y, x) positions
        - Convention: (0, 0) is top-left of the tile grid
    '''
    xy_pos = n.split('Tile_')[-1][:7]
    return tuple(int(i)-1 for i in xy_pos.split('-'))[::-1]


def parse_slice_from_name(n):
    '''
    Extract the z-slice index from a tile filename.

    For VolumeScope/MAPs, filenames follow the pattern: Tile_XX-YY_sZZZZ.tif
    where ZZZZ is the slice number (e.g., s0001 for slice 1).

    Args:
        n: Tile filename or full path (e.g., 'Tile_03-02_s0001.tif')

    Returns:
        Integer z-slice index. Example: 'Tile_03-02_s0001.tif' -> 1

    Notes for implementing other backends:
        - Must return an integer representing the z-slice
        - Slices are sorted numerically, so consistent numbering is required
        - The returned value is used as a dictionary key to group tiles by slice
    '''
    return int(n.split('s')[-1][:4])