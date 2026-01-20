# Implementing IO Backends

This guide explains how to add support for a new microscope or file format.

## Overview

IO backends parse tile filenames and metadata from different microscope systems. Each backend is a Python module in `emalign/io/` that provides standardized functions for extracting tile positions, slice indices, and resolution information.
Currently, parsing functions rely on information contained in the file name, but it could be parsed from anything as long as the output is consistent.

## Required Interface

Your backend module must define the following:

### Constants

```python
FILE_EXT = '.tif'  # File extension for tile images (e.g., '.tif', '.png')
```

### Functions

#### `parse_yx_pos_from_name(n) -> tuple[int, int]`

Extract the (y, x) grid position from a tile filename.

- **Input**: Filename or full path (e.g., `'/path/to/tile_001_002.tif'`)
- **Output**: Tuple `(y, x)` as 0-indexed integers
- **Requirements**:
  - Must be 0-indexed (first tile is `(0, 0)`)
  - Convention: `(0, 0)` is top-left of the tile grid
  - Each tile in a slice must have a unique position

#### `parse_slice_from_name(n) -> int`

Extract the z-slice index from a tile filename.

- **Input**: Filename or full path
- **Output**: Integer z-slice index
- **Requirements**:
  - Slices are sorted numerically
  - Used as dictionary key to group tiles

#### `get_tileset_resolution(tileset_path) -> tuple | None`

Read resolution metadata from a tileset directory.

- **Input**: Path to directory containing tiles
- **Output**: `(tileset_path, (y_res, x_res))` or `None` if not found
- **Requirements**:
  - Resolution in consistent units (typically nm)
  - Return `None` if resolution cannot be determined

#### `get_tilesets(main_dir, resolution, dir_pattern, num_workers) -> list[str]`

Find all tileset directories matching criteria.

- **Input**:
  - `main_dir`: Parent directory to search
  - `resolution`: Target `(y_res, x_res)` tuple
  - `dir_pattern`: List of required substrings in directory names
  - `num_workers`: Thread count for parallel scanning
- **Output**: Sorted list of absolute paths to matching directories

## Registering the Backend

Add your backend to `emalign/io/backend.py`:

```python
_BACKENDS = {
    'volumescope': 'emalign.io.volumescope',
    'my_microscope': 'emalign.io.my_microscope',  # Add this line
}
```

## Usage

The backend is selected via the `mode` argument in config preparation:

```bash
python -m emalign.prep_config_xy --mode my_microscope ...
```

In code:

```python
from emalign.io.backend import get_io_backend

backend = get_io_backend('my_microscope')
pos = backend.parse_yx_pos_from_name('slice_0001_row_02_col_03.png')  # (2, 3)
z = backend.parse_slice_from_name('slice_0001_row_02_col_03.png')     # 1
```

## Reference

See `emalign/io/volumescope.py` for a complete implementation example with detailed docstrings.
