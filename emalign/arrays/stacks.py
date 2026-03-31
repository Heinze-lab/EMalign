import json
import os
import re

from glob import glob
from collections import defaultdict

from ..io.tif import load_tilemap, load_tif
from .tile_map import TileMap


class Stack:
    def __init__(self, stack_path=None, stack_name=None, tile_maps_paths=None, tile_maps_invert=None, io_backend=None):

        if io_backend is None:
            raise ValueError('Please provide a valid IO backend')
        
        self.io_backend = io_backend
        self.file_ext = io_backend.FILE_EXT

        if stack_path is not None:
            self.stack_path = os.path.abspath(stack_path)
        else:
            self.stack_path = None
        
        if stack_path is not None and stack_name is None:
            self.stack_name = stack_path.split('/')[-2]
        else:
            self.stack_name = stack_name

        if tile_maps_paths is not None:
            self._set_tilemaps_paths(tile_maps_paths)
        
        if tile_maps_invert is not None:
            self.tile_maps_invert = tile_maps_invert
            
    def __str__(self):
        return self.stack_name

    def _get_tilemaps_paths(self):
        # Produces lists of paths of all tifs contained in self.stack_path
        tile_paths = glob(os.path.join(self.stack_path, f'*{self.file_ext}'))

        # Get paths and group by slice
        self.slice_to_paths = defaultdict(list)
        for tile_path in tile_paths:
            self.slice_to_paths[self.io_backend.parse_slice_from_name(tile_path)].append(tile_path)

        self.slices = sorted(list(self.slice_to_paths.keys()))
        
        # Sort
        self.slice_to_paths = {k: self.slice_to_paths[k] for k in self.slices}

        # Prep tilemaps
        self.slice_to_tilemap = defaultdict(dict)
        tile_indices = set()
        for s in self.slices:
            d = {}
            for t in self.slice_to_paths[s]:
                tile_indices.add(self.io_backend.parse_yx_pos_from_name(t))
                d.update({self.io_backend.parse_yx_pos_from_name(t): t for t in self.slice_to_paths[s]})
            self.slice_to_tilemap.update({s: d})   

        self.tile_maps_invert = dict(zip(tile_indices, [None]*len(tile_indices)))
        
    def _set_tilemaps_paths(self, tile_map_paths):

        self.slice_to_tilemap = tile_map_paths
        self.slices = sorted(list(self.slice_to_tilemap.keys()))
        self.slice_to_tilemap = {k: self.slice_to_tilemap[k] for k in self.slices}
        
        self.slice_to_paths = defaultdict(list)
        for z, d in self.slice_to_tilemap.items():
            self.slice_to_paths[z].append(list(d.values()))

        tile_indices = list(d.keys())
        self.tile_maps_invert = dict(zip(tile_indices, [None]*len(tile_indices)))

    def get_tile_map(self, z, apply_gaussian, apply_clahe, skip_missing=True):

        process_scheme = {}
        if apply_gaussian:
            process_scheme['gaussian'] = {'kernel_size': [3,3], 'sigma': 1}
        if apply_clahe:
            process_scheme['clahe'] = {'clip_limit': 2, 'tile_grid_size': [10,10]}

        scale = 1
        # Load tile_map
        tile_map_paths = self.slice_to_tilemap[z]
        _, tile_map, _ = load_tilemap({z: tile_map_paths}, 
                                       self.tile_maps_invert,
                                       process_scheme,
                                       scale,
                                       skip_missing=True)
        
        # Check for missing tiles and replace with previous or next adjacent tile
        missing_tiles = []
        for k, img in tile_map.items():
            if img is not None:
                continue
            elif not skip_missing:
                raise RuntimeError(f'Missing slice {k} for z={z}')
            else:
                proc = process_scheme.copy()
                if self.tile_maps_invert[k]:
                    proc['invert'] = True

                missing_tiles.append(k)
                # Try previous Z
                prev_z = self.slices[self.slices.index(z)-1]
                tif_path = self.slice_to_tilemap[prev_z][k]
                img, _, _ = load_tif(tif_path, 
                                     scale,
                                     proc)
            if img is None:
                # Try next Z
                next_z = self.slices[self.slices.index(z)+1]
                tif_path = self.slice_to_tilemap[next_z][k]
                img, _, _ = load_tif(tif_path, 
                                     scale,
                                     proc)
            if img is None:
                raise RuntimeError(f'Tiles {k} missing or corrupted for three slices in a row ({prev_z}, {z}, {next_z})')
            tile_map_paths[k] = tif_path
            tile_map[k] = img

        tm = TileMap(z=z, tile_map_paths=tile_map_paths, tile_map=tile_map, stack_name=self.stack_name)
        tm.missing_tiles = missing_tiles
        tm.processing = {'tile_maps_invert': self.tile_maps_invert,
                         'gaussian': apply_gaussian,
                         'clahe': apply_clahe,
                         'scale': scale}

        return tm


def parse_stack_info(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    tile_maps_paths = {}

    for z, tm in config['tile_maps'].items():
        tm = {tuple(int(i) 
                for i in re.findall(r'\b\d+\b', k)): v for k,v in tm.items()}
        tile_maps_paths.update({int(z): tm})

    tile_maps_invert = {tuple(int(i) for i in re.findall(r'\b\d+\b', k)): v 
                            for k,v in config['tile_maps_invert'].items()}
    
    ignore_slices = [config.get('ignore_slices_local', []), 
                     config.get('ignore_slices_global', [])]
    return tile_maps_paths, tile_maps_invert, ignore_slices