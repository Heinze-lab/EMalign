# EMalign

A Python package for aligning Serial Block-face Electron Microscopy (SBEM) image tiles into 3D volumetric stacks using [SOFIMA](https://github.com/google-research/sofima) (Scalable Optical Flow-based Image Montaging and Alignment).

## Installation

```bash
pip install -e .
```

### Dependencies

Core dependencies:
- numpy
- tensorstore
- JAX/XLA (for GPU acceleration)
- OpenCV (cv2)
- sofima
- pymongo (progress tracking)
- neuroglancer (visualization)

## Quick Start

### 1. XY Alignment (Tile Stitching)

**Prepare configuration**:
```bash
python -m emalign.prep_config_xy \
  -m /path/to/tiles \
  -p /path/to/project_dir \
  -o output_name \
  -res 8.0 \
  -c 4
```

**Execute alignment**:
```bash
CUDA_VISIBLE_DEVICES=0 python -m emalign.align_dataset_xy \
  -cfg /path/to/main_config.json \
  -c 4 \
  --overwrite
```

### 2. Z Alignment (Cross-Slice Alignment)

**Prepare configuration**:
```bash
python -m emalign.prep_config_z \
  -cfg /path/to/main_config.json \
  -cfg-z /path/to/z_config.json \
  -o /path/to/config/z_config/ \
  -c 4 \
  --exclude /flow _mask
```

**Execute alignment**:
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m emalign.align_dataset_z \
  -cfg /path/to/config/z_config/ \
  -cfg-z /path/to/z_config.json \
  -c 4 \
  -ds 10
```

### 3. Inspect Results

```bash
python -m emalign.inspect_dataset /path/to/output.zarr
```

## Configuration

### XY Main Configuration (`main_config.json`)

```json
{
  "project_name": "my_project",
  "main_dir": "/path/to/tiles",
  "output_path": "/path/to/output.zarr",
  "resolution": [8.0, 8.0],
  "offset": [0, 0, 0],
  "stride": 20,
  "apply_gaussian": false,
  "apply_clahe": false,
  "stack_configs": {
    "stack_name": "/path/to/stack_config.json"
  },
  "mongodb_config_filepath": null
}
```

### Z Parameters Configuration (`config_z.json`)

```json
{
  "scale_flow": 0.5,
  "stride": 20,
  "patch_size": [160, 160],
  "max_deviation": 5,
  "max_magnitude": 0,
  "step_slices": 1,
  "yx_target_resolution": [10, 10],
  "k0": 0.01,
  "k": 0.4,
  "gamma": 0.5,
  "flow": {},
  "mesh": {},
  "warp": {}
}
```

## API Usage

## License

See LICENSE file for details.

## Author

Valentin Gillet (valentin.gillet@biol.lu.se)

## Acknowledgments

EMalign is built on [SOFIMA](https://github.com/google-research/sofima) by Google Research for optical flow-based image alignment.
