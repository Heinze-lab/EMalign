# EMalign

EMalign is a Python package for aligning Serial Electron Microscopy (SEM) image tiles into 3D volumetric stacks using [SOFIMA](https://github.com/google-research/sofima) (Scalable Optical Flow-based Image Montaging and Alignment). 

## Installation

Clone the repository locally
```bash
git clone https://github.com/Heinze-lab/EMalign.git
```

Create a new environment and activate it (here using conda)
```bash
conda create -n myenv python=3.12
conda activate myenv
```

Install JAX with CUDA support to make use of GPU (recommended)
```bash
pip install jax[cuda12]   # Replace cuda12 with your version
```

Install dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

_Installation was tested on Ubuntu 20.04 with Python 3.12._

## Other requirements

### MongoDB
A MongoDB database is currently required for metadata and progress tracking. The free community edition is sufficient for this. It can be found [here](https://www.mongodb.com/products/self-managed/community-edition) along with installation tutorials.

### Neuroglancer
Neuroglancer is currently required for data visualization. Its installation may require extra steps described [here](https://github.com/google/neuroglancer?tab=readme-ov-file#building).

## Usage

Alignment can be split into two major steps:
* XY alignment: image tiles representing one large region of interest are stitched along overlapping regions to obtain one contiguous image per slice.
* Z alignment: warping of slices along the Z axis to produce smooth scrolling through the output stack.

Run alignment with the `run_alignment.py` script. Every step takes the project directory (`-p`), where configurations, progress and outputs are kept. Use `--help` on any step for its full list of options.

See docs about the resulting outputs [here](/docs/OUTPUT.md).

### Step by step

```bash
# 1. XY alignment
python run_alignment.py xy prep -p /path/to/project_dir -i /path/to/raw_data -r 10 10
python run_alignment.py -g 0 xy align -p /path/to/project_dir -c 4

# (Optional) Fuse stacks overlapping on the same Z slices
python run_alignment.py -g 0 xy fuse -p /path/to/project_dir -c 4

# 2. Z alignment
python run_alignment.py z prep -p /path/to/project_dir -cfg-z /path/to/z_config.json
python run_alignment.py -g 0 z transform -p /path/to/project_dir -c 4
python run_alignment.py -g 0 z flow -p /path/to/project_dir -c 4
python run_alignment.py -g 0 z render -p /path/to/project_dir -c 4
```

`-g` selects the GPU(s) to use (e.g. `-g 0,1`). An example of Z parameters can be found in [config_z.json](emalign/config/config_z.json). See other flags below.

### Automatic mode

Without a step, `run_alignment.py` runs the next pending step of the project. Preparation steps are however not ran automatically because they require extra information, in which case, the script stops and tells you when `xy prep` or `z prep` is needed.

```bash
python run_alignment.py -g 0 -p /path/to/project_dir -c 4
```

Check where a project stands with:

```bash
python scripts/project_status.py -p /path/to/project_dir
```

### Inspect results

Data can be inspected at any stage through your browser with Neuroglancer:

```bash
python inspect_dataset.py -d /path/to/output.zarr/dataset
```

Pointing -d to the container root shows the final output if it exists, otherwise all XY aligned stacks. It can also point to a single dataset or to xy_intermediate. 
Open the printed link into a browser to open a Neuroglancer viewer. The data will appear once loaded into memory. 

* `--bbox` min_z max_z [min_y max_y min_x max_x]: only load part of the data. Data is loaded into memory, so prefer this or the downsampled output (10x_<project_name>) for large datasets.
* `--mode all_ds_xy_first_z`: only show the first slice of each XY stack, a quick check of stitching.
* `--print-shape`: only print the dataset shape and exit.
* `--port`: Neuroglancer port (default: 55555). The viewer is bound to localhost, so forward the port when working on a remote machine.

### Flags

Main flags used by `run_alignment.py`. Run `python run_alignment.py <step> <substep> --help` for the full list.

| Flag | Steps | Description |
|------|-------|-------------|
| `-p`, `--project-dir` | all | Project directory holding configurations, progress, and outputs. |
| `-c`, `--cores` | all | Number of threads to use for parallel steps. |
| `-g`, `--gpu` | all | GPU ID(s) to use, e.g. `0` or `0,1`. Default: `CUDA_VISIBLE_DEVICES`. |
| `--verbose` | all | Print debug logs. |
| `-i`, `--input_dir` | `xy prep` | Directory(ies) containing the raw tiles to use as input. Several directories are concatenated along Z in the order given. |
| `-r`, `--resolution` | `xy prep` | XY resolution of the tileset to align. Used to find tiles in the input directory. |
| `-o`, `--output_name` | `xy prep` | Name of the output zarr container. Default: project directory name. |
| `--mode` | `xy prep` | How to parse tile file names (see [io modes](/io/BACKENDS.md)). Default: `volumescope`. |
| `-cfg-z`, `--config-z` | `z prep` | JSON file with the Z alignment parameters (flow, mesh, rendering). |
| `--exclude` | `z prep` | Patterns of datasets to leave out of Z alignment. |
| `-ds`, `--downsample-scale` | `z prep` | Downsampling factor of the inspection store. Default: 10. |
| `--force-overwrite` | `xy prep`, `xy fuse`, `z prep` | Overwrite existing configuration files without prompting. |
| `--overwrite` | `xy align`, `xy fuse` | Overwrite existing output datasets. |
| `--start-over` | `xy align`, `z transform`, `z flow`, `z render` | Wipe all progress and restart. |
| `--wipe-progress` | `xy align`, `xy fuse`, `z transform`, `z flow`, `z render` | Wipe progress of specific stack(s) before starting. |

## Configuration

Documentation about configuration files can be found [here](/docs/CONFIG.md).

## License

See LICENSE file for details.

## Author

Valentin Gillet (valentin.gillet@biol.lu.se)

## Acknowledgments

EMalign is built on [SOFIMA](https://github.com/google-research/sofima) by Google Research for optical flow-based image alignment.
