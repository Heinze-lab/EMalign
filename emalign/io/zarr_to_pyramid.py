import argparse
import asyncio
import json
import logging
import numpy as np
import os
import cv2
import tensorstore as ts
import shutil
import sys

from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from tqdm.asyncio import tqdm

from emalign.io.store import get_store_attributes


logging.basicConfig(level=logging.INFO)


'''
Convert a zarr container to image pyramids
'''

CONCURRENT_SLICES = 10 # Defines how many slices are retrieved at the same time. Influences memory usage.

class JsonNumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def rotate_image_pil(img, angle, center=None):
    '''Use PIL to rotate large images that opencv cannot handle without cropping.'''
    image = Image.fromarray(img)
    if center is not None and not isinstance(center, tuple):
        center = tuple(center)
        
    return np.array(image.rotate(angle, center=center, resample=Image.BILINEAR, expand=True))

def create_CAVE_info_file(tile_shape,
                          shape, 
                          resolution,
                          voxel_offset,
                          downsample_factor,
                          max_layer):
    
    '''Creates info file to view images within CAVE'''
    
    chunk_size = [tile_shape, tile_shape, 1]

    # Convert to numpy arrays for computation
    voxel_offset = np.array(voxel_offset, dtype=np.float64)
    resolution = np.array(resolution, dtype=np.float64)
    downsample_factor_arr = np.array([downsample_factor, downsample_factor, 1], dtype=np.float64)

    scales = [{
          'chunk_sizes': [chunk_size],
          'encoding': 'jpeg',
          'key': 'volume/0',
          'resolution': resolution.astype(int).tolist(),
          'size': shape,
          'voxel_offset': voxel_offset.astype(int).tolist()
        }]

    for layer in range(1, max_layer + 2):
        resolution = resolution * downsample_factor_arr
        voxel_offset = voxel_offset / downsample_factor_arr
        
        scales.append({
              'chunk_sizes': [chunk_size],
              'encoding': 'jpeg',
              'key': 'volume/' + str(layer),
              'resolution': resolution.astype(int).tolist(),
              'size': shape,
              'voxel_offset': voxel_offset.astype(int).tolist()
                    })

    info = {
        'data_type': 'uint8',
        'num_channels': 1,
        'scales': scales,
        'type': 'image'
    }
        
    return info


def write_single_tile(args):
    '''Write a single tile - runs in thread pool.'''
    filepath, tile, encode_params, jpeg_quality = args
    
    success, encoded = cv2.imencode('.jpg', tile, encode_params)
    if success:
        encoded.tofile(filepath)
    else:
        Image.fromarray(tile).save(filepath, 'JPEG', quality=jpeg_quality)


def build_pyramid(image: np.ndarray, max_layer: int) -> list:
    '''
    Build image pyramid using OpenCV.
    
    Args:
        image: Input 2D image array
        max_layer: Number of downsampling levels
        downsample_factor: Downsampling factor (default 2 for pyrDown)
    
    Returns:
        List of images from original to most downsampled
    '''
    pyramid = [image]
    current = image
    
    for _ in range(max_layer):
        current = cv2.pyrDown(current)
        pyramid.append(current)
    
    return pyramid


def write_pyramid_tiles(pyramid: list, 
                        z: int, 
                        output_path: str, 
                        tile_shape: int, 
                        jpeg_quality: int = 90, 
                        executor: ThreadPoolExecutor = None) -> None:
    '''
    Write pyramid tiles to disk using OpenCV for JPEG encoding.
    Optionally parallelizes tile encoding using a thread pool.
    '''
    tile_dir = os.path.join(output_path, str(z))
    os.makedirs(tile_dir, exist_ok=True)

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    
    # Collect all tiles to write
    tiles_to_write = []

    for ds_factor, data in enumerate(pyramid):
        level_dir = os.path.join(tile_dir, str(ds_factor))
        os.makedirs(level_dir, exist_ok=True)

        # Ensure uint8
        if data.dtype != np.uint8:
            data = (255 * np.clip(data, 0, 1)).astype(np.uint8)

        height, width = data.shape[:2]
        tiles_y = (height + tile_shape - 1) // tile_shape
        tiles_x = (width + tile_shape - 1) // tile_shape

        for y in range(tiles_y):
            y_start = y * tile_shape
            y_end = min(y_start + tile_shape, height)
            
            for x in range(tiles_x):
                x_start = x * tile_shape
                x_end = min(x_start + tile_shape, width)
                
                tile = data[y_start:y_end, x_start:x_end]

                if not np.any(tile):
                    continue
                
                # Pad if needed
                pad_y = tile_shape - tile.shape[0]
                pad_x = tile_shape - tile.shape[1]
                
                if pad_y > 0 or pad_x > 0:
                    tile = np.pad(tile, ((0, pad_y), (0, pad_x)), 
                                  mode='constant', constant_values=0)

                filepath = os.path.join(level_dir, f'{y}_{x}.jpg')
                tiles_to_write.append((filepath, tile, encode_params, jpeg_quality))

    # Write tiles - parallel or sequential
    if executor is not None and len(tiles_to_write) > 1:
        # Submit all tiles to thread pool and wait for completion
        futures = [executor.submit(write_single_tile, args) for args in tiles_to_write]
        # Wait for all to complete and raise any exceptions
        for future in futures:
            future.result()
    else:
        # Sequential fallback
        for args in tiles_to_write:
            write_single_tile(args)


def find_bbox(store_path: str,
              downsampled_factor: int = 10) -> tuple:
    '''
    Find bounding box containing image data.
    Returns minpt (yx) and maxpt (yx)
    '''
    ds_store = ts.open({
        'driver': 'zarr',
        'kvstore': {
            'driver': 'file',
            'path': store_path,
        }
    }, read=True).result()
    
    z_dim, y_dim, x_dim = ds_store.shape
    
    min_y, min_x = y_dim, x_dim
    max_y, max_x = 0, 0
    
    for z in range(z_dim):
        slice_data = ds_store[z].read().result()
        
        row_has_data = np.any(slice_data != 0, axis=1)
        col_has_data = np.any(slice_data != 0, axis=0)
        
        if not np.any(row_has_data):
            # If no row, no col
            continue
        
        # argmax returns index of first True (treats True as 1)
        # For last True: flip array, find first, convert index
        min_y = min(min_y, np.argmax(row_has_data))
        max_y = max(max_y, y_dim - 1 - np.argmax(row_has_data[::-1]))
        min_x = min(min_x, np.argmax(col_has_data))
        max_x = max(max_x, x_dim - 1 - np.argmax(col_has_data[::-1]))
    
    minpt = np.array([min_y, min_x]) * downsampled_factor - 100
    maxpt = np.array([max_y, max_x]) * downsampled_factor + 100
    
    return minpt, maxpt


def dataset_to_pyramid(dataset_path: str, 
                       output_path: str, 
                       max_layer: int = 3, 
                       tile_shape: int = 1024, 
                       downsample_factor: int = 2, 
                       num_threads: int = 1,
                       rotate: int = 0,
                       autoresize_canvas: bool = True,
                       duplicate_missing_slices: bool = True,
                       slice_range: list = None,
                       rotation_center: tuple = None,
                       downsampled_factor: int = 10, 
                       downsampled_dataset_prefix: str = '10x_',
                       jpeg_quality: int = 90) -> None:
    '''
    Convert a zarr dataset to image pyramid tiles.
    
    Args:
        dataset_path: Path to input zarr dataset
        output_path: Base output directory
        max_layer: Maximum pyramid depth
        tile_shape: Tile size in pixels
        downsample_factor: Downsampling factor between levels
        num_threads: Number of concurrent processing threads
        rotate: Rotation angle in degrees
        autoresize_canvas: Whether to crop to bounding box
        duplicate_missing_slices: Whether to duplicate previous slice for missing data
        slice_range: [start, end] range of slices to process
        rotation_center: Center point for rotation
        downsampled_factor: Factor of downsampled reference dataset
        downsampled_dataset_prefix: Prefix of downsampled reference dataset
        jpeg_quality: JPEG compression quality (0-100)
    '''

    if downsample_factor != 2:
        raise NotImplementedError('Downsample factor different from 2 is not implemented.')
      
    # Set tensorstore context to manage load
    cache_size_mb = 300
    context = ts.Context({
        'cache_pool': {'total_bytes_limit': cache_size_mb * 1024 * 1024},
        'data_copy_concurrency': {'limit': 4},
        'file_io_concurrency': {'limit': 4}
    })
    
    # Create/check output path
    output_path = os.path.join(output_path, 'pyramid')
    os.makedirs(output_path, exist_ok=True)
    if os.path.exists(os.path.join(output_path, 'info')):
        response = input('An info file already exists. You risk overwriting data.\nY to continue or ENTER to exit: ').lower()
        if response != 'y':
            sys.exit('Exiting.')

    # Normalize rotation angle
    rotation_angle = rotate
    if abs(rotation_angle) > 360:
        rotation_angle = rotation_angle % 360

    # Prepare dataset
    dataset_sync = ts.open({
        'driver': 'zarr',
        'kvstore': {
            'driver': 'file',
            'path': dataset_path,
        }
    }, read=True).result()  
    dataset_spec = dataset_sync.spec(retain_context=False).to_json()

    # Handle slice range
    if slice_range is None:
        slice_range = [0, dataset_sync.domain.exclusive_max[0]]
    elif len(slice_range) == 1:
        slice_range += [dataset_sync.domain.exclusive_max[0]]
        logging.info(f'Setting upper bound to max: {slice_range[1]}')
    elif len(slice_range) > 2:
        raise ValueError('Slice range must have 1 (start) or 2 elements ([start, end])')
    if slice_range[1] <= slice_range[0]:
        raise ValueError('Invalid slice range: upper bound must exceed lower bound')
    
    # Auto-resize canvas (crop to bounding box)
    if autoresize_canvas:
        if rotation_angle in [0, 90, 180, 270, 360]:
            logging.info('Resizing canvas to bounding box...')
            base_store, dataset_name = os.path.abspath(dataset_sync.kvstore.path).rsplit('/', maxsplit=1)
            ds_store_path = os.path.join(base_store, downsampled_dataset_prefix + dataset_name)
            minpt, maxpt = find_bbox(ds_store_path,
                                     downsampled_factor=downsampled_factor)
            # Clamp to valid dataset bounds
            ds_min = np.array(dataset_sync.domain.inclusive_min[1:])
            ds_max = np.array(dataset_sync.domain.exclusive_max[1:])
            minpt = np.maximum(minpt, ds_min)
            maxpt = np.minimum(maxpt, ds_max)
        else:
            raise ValueError('Cannot auto-resize canvas with non-90-degree rotation.')
    else:
        minpt = dataset_sync.domain.inclusive_min[1:]
        maxpt = dataset_sync.domain.exclusive_max[1:]
    dataset_sync = dataset_sync[slice_range[0]:slice_range[1], minpt[0]:maxpt[0], minpt[1]:maxpt[1]]
    bbox = slice_range + list(map(int, [minpt[0],maxpt[0],minpt[1],maxpt[1]]))

    # TPE for writing tiles in parallel
    tile_executor = ThreadPoolExecutor(max_workers=num_threads)
    
    # Process asynchronously
    async def process_dataset():              
        dataset = await ts.open(dataset_spec, context=context, read=True)
        logging.info(f'Original shape: {dataset.shape}')
        logging.info(f'Bbox: {bbox}')
        dataset = dataset[slice_range[0]:slice_range[1], minpt[0]:maxpt[0], minpt[1]:maxpt[1]]

        # Apply rotation transforms
        if rotation_angle == 90:
            logging.info('Rotating dataset by 90 degrees')
            dataset = dataset.transpose((0, 2, 1))[:, :, ::-1]
        elif rotation_angle == 180:
            logging.info('Rotating dataset by 180 degrees')
            dataset = dataset[:, ::-1, ::-1]
        elif rotation_angle == 270:
            logging.info('Rotating dataset by 270 degrees')
            dataset = dataset.transpose((0, 2, 1))[:, ::-1, :]
        elif rotation_angle == 360:
            logging.info('Rotating dataset by 360 degrees (no-op)')
        elif rotation_angle != 0:
            logging.info(f'Data will be rotated slice-by-slice by {rotation_angle} degrees')

        logging.info(f'Output shape: {dataset.shape}')
        logging.info(f'Slice interval: {slice_range[0]} - {slice_range[1]}')
        
        #logging.info('Warming up...')
        # This allegedly would help not stalling at the start by resolving chunk location
        #min_loc = dataset.domain.inclusive_min
        #_ = await dataset[slice_range[0], min_loc[1]:min_loc[1]+1, min_loc[2]:min_loc[2]+1].read()  

        # Process slices
        num_slices = slice_range[1] - slice_range[0]
        semaphore = asyncio.Semaphore(CONCURRENT_SLICES)
        slice_has_data = {}  

        ##### FIRST PASS: deal with dataset ignoring missing slices #####
        pbar = tqdm(total=num_slices, desc='Processing slices', unit='slices', dynamic_ncols=True)
        async def process_slice(z: int):
            async with semaphore:
                try:
                    pbar.set_description('Retrieving slice...')
                    data = await dataset[z].read()
                    has_data = np.any(data)
                    slice_has_data[z] = has_data
                    
                    if has_data:
                        # Apply per-slice rotation if needed
                        if rotation_angle % 90 != 0 and rotation_angle != 0:
                            pbar.set_description('Rotating slice...')
                            data = rotate_image_pil(data, rotation_angle, center=rotation_center)
                        
                        # Build pyramid using OpenCV. Downsamples by a factor of 2
                        pbar.set_description('Building pyramid...')
                        pyramid = build_pyramid(data, max_layer)
                        
                        # Write tiles, starting from slice zero because that's what catmaid wants
                        pbar.set_description('Writing pyramid...')
                        write_pyramid_tiles(pyramid, z - slice_range[0], output_path, tile_shape, 
                                            jpeg_quality, executor=tile_executor)
                    
                    pbar.update(1)
                    return {'z': z, 'status': 'success' if has_data else 'empty'}
                    
                except Exception as e:
                    logging.error(f'Error processing slice {z}: {e}')
                    slice_has_data[z] = False
                    pbar.update(1)
                    return {'z': z, 'status': 'error', 'error': str(e)}
                
        # Create and run tasks
        tasks = [
            process_slice(z) 
            for z in range(slice_range[0], slice_range[1])
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        pbar.close()
        
        # Report failures
        errors = [r for r in results if isinstance(r, dict) and r.get('status') == 'error']
        if errors:
            logging.warning(f'{len(errors)} slices failed to process')
            for err in errors[:5]:  # Show first 5 errors
                logging.warning(f"  Slice {err['z']}: {err.get('error', 'Unknown error')}")

        ##### SECOND PASS: deal with missing slices if they should be replaced with duplicates #####
        copy_from = {}
        if duplicate_missing_slices:
            # Build copy mapping: only copy to empty slices that follow at least one slice with data
            last_valid_z = None
            
            for z in range(slice_range[0], slice_range[1]):
                if slice_has_data.get(z, False):
                    last_valid_z = z
                elif last_valid_z is not None:
                    # This empty slice is preceded by at least one slice with data
                    copy_from[z] = last_valid_z
            
            if copy_from:
                logging.info(f'Filling {len(copy_from)} empty slices from previous valid slices...')
                
                for dst_z, src_z in tqdm(copy_from.items(), desc='Copying empty slices', unit='slices', dynamic_ncols=True):
                    src_dir = os.path.join(output_path, str(src_z))
                    dst_dir = os.path.join(output_path, str(dst_z))
                    
                    if os.path.exists(src_dir):
                        # Copy directory tree (all pyramid levels and tiles)
                        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

        return slice_range, copy_from

    # Run async processing
    try:
        slice_range, copy_from = asyncio.run(process_dataset())
    finally:
        # Clean up threads
        tile_executor.shutdown(wait=True)  

    # Generate info files
    z, y, x = dataset_sync.shape

    # Get dataset attributes
    resolution = [1, 1, 1]
    voxel_offset = [slice_range[0], 0, 0] # We started at index zero so Z offset is the start of range
    try:
        attrs = get_store_attributes(dataset_sync)
        resolution = attrs.get('resolution', resolution)[::-1]
        voxel_offset = attrs.get('voxel_offset', voxel_offset)[::-1]
    except FileNotFoundError:
        logging.warning('No attributes found for input dataset. Using default values.')
    except Exception as e:
        raise e
    
    # Write CAVE info file
    info = create_CAVE_info_file(tile_shape,
                                 [x, y, z], 
                                 resolution,
                                 voxel_offset,
                                 downsample_factor,
                                 max_layer)
    
    with open(os.path.join(output_path, 'info'), 'w') as f:
        json.dump(info, f, indent=2, cls=JsonNumpyEncoder)

    # Set rotation center
    if rotation_center is None:
        rotation_center = [x / 2, y / 2]

    # Write pyramid metadata
    info_pyramid = {
        'dataset_path': dataset_path,
        'max_layer': max_layer,
        'tile_shape': tile_shape,
        'downsample_factor': downsample_factor,
        'rotation': rotate,
        'rotation_center': rotation_center,
        'autoresize_canvas': autoresize_canvas,
        'duplicate_missing_slices': duplicate_missing_slices,
        'duplicated_slices': copy_from,
        'original_offset_xyz': voxel_offset,
        'bbox': bbox
    }
    
    with open(os.path.join(output_path, 'info_pyramid.json'), 'w') as f:
        json.dump(info_pyramid, f, indent=2, cls=JsonNumpyEncoder)

    logging.info('Done!')
    logging.info(f'Output written at: {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert aligned zarr dataset to image pyramid tiles.'
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
                        help='Output directory for pyramid tiles.')
    parser.add_argument('-l', '--max-layer',
                        metavar='MAX_LAYER',
                        dest='max_layer',
                        type=int,
                        default=5,
                        help='Maximum pyramid depth (default: 5)')
    parser.add_argument('--tile-shape',
                        metavar='TILE_SHAPE',
                        dest='tile_shape',
                        type=int,
                        default=1024,
                        help='Tile size in pixels (default: 1024)')
    parser.add_argument('-ds', '--downsample-factor',
                        metavar='SCALE',
                        dest='downsample_factor',
                        type=int,
                        default=2,
                        help='Downsampling factor between levels (default: 2, other values not supported)')
    parser.add_argument('-c', '--cores',
                        metavar='CORES',
                        dest='num_threads',
                        type=int,
                        default=1,
                        help='Number of concurrent threads (default: 1)')
    parser.add_argument('-z', '--slice-range',
                        metavar='SLICE_RANGE',
                        dest='slice_range',
                        nargs='+',
                        type=int,
                        default=None,
                        help='Slice range [start end] (default: all slices)')
    parser.add_argument('--error-missing',
                        dest='duplicate_missing_slices',
                        default=True,
                        action='store_false',
                        help='Error on missing slices instead of duplicating')
    parser.add_argument('--no-resize',
                        dest='autoresize_canvas',
                        default=True,
                        action='store_false',
                        help='Disable automatic canvas cropping')
    parser.add_argument('--rotate',
                        metavar='DEGREES',
                        dest='rotate',
                        type=int,
                        default=0,
                        help='Rotation angle in degrees (default: 0)')
    parser.add_argument('--rotation-center',
                        metavar=('X', 'Y'),
                        dest='rotation_center',
                        nargs=2,
                        type=float,
                        default=None,
                        help='Rotation center for non-90° rotations')
    parser.add_argument('--jpeg-quality',
                        metavar='QUALITY',
                        dest='jpeg_quality',
                        type=int,
                        default=90,
                        help='JPEG quality 0-100 (default: 90)')

    args = parser.parse_args()
    dataset_to_pyramid(**vars(args))
