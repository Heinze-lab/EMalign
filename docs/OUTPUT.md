The pipeline produces multiple files throughout alignment. Configuration files guide each alignment steps, while temporary datasets provide intermediate checkpoints and are all written in a zarr container at the root of the project directory.

```
project_dir/
```

## Configuration

Configuration files are JSON files created by each preparation step in the `config` folder:

```
project_dir/
└── config/
    ├── xy_config/ 
    │   ├── main_config.json
    │   └── xy_<dataset_name>.json
    ├── fuse_config/
    │   ├── 00_fuse_plan.json
    │   └── fuse_<zmin>_<zmax>_<i>.json
    └── z_config/
        ├── 00_align_plan.json
        └── z_<dataset_name>.json
```

`xy_config` contains all configuration files necessary for XY alignment. 
* `main_config.json` is a centralized inventory of the paths for the individual datasets configuration files. It also contains parameters required by SOFIMA for flow computations necessary for stitching, among other things.
* `xy_<dataset_name>.json` are one configuration file per stack of overlapping tiles (or tilesets). They each contain a dictionary of grid position to image path per slice for this tileset.

`fuse_config` only exists if some stacks overlap on the same Z slices. It is written by `xy fuse` on its first run (not by a preparation step) and reused afterwards.
* `00_fuse_plan.json` lists the groups of stacks to fuse.
* `fuse_<zmin>_<zmax>_<i>.json` are one configuration file per group, covering slices `zmin` to `zmax`. They each contain the stacks to fuse and the name of the resulting fused stack.

`z_config` contains all configuration files necessary for Z alignment.
* `00_align_plan.json` contains ordered lists of datasets that are neighbors along the Z axis and should therefore be aligned to each other. 
* `z_<dataset_name>.json` are one configuration file per stack. They each contain parameters required by SOFIMA for flow computations, mesh relaxation, and rendering. 

For more details about the content of configuration files, see the dedicated [docs](/docs/CONFIG.md).

## Intermediate outputs

Each step of the pipeline produces intermediate datasets containing the result of XY alignment, and transformations computed for Z alignment. The output of each step is required for the following one. They also act as checkpoints to enable stopping a job and restarting from there. 

In principle, configuration files and intermediate transformations provide the recipe to produce the final output from the raw data. They can be stored cheaply along the archived raw data to recompute data necessary for analysis at a reduced computational and temporal cost. 

Intermediate steps are written at the root of the destination zarr container. Its name can be specified during the xy preparation steps or will be named as the project directory by default.

```
project_dir/
├── config/
└── project_dir.zarr/
    ├── xy_intermediate/
    │   ├── <dataset_name>/
    │   ├── <dataset_name>_mask/
    │   ├── <a>_<b>_fused/
    │   └── <a>_<b>_fused_mask/
    └── z_intermediate/
        ├── transform/
        │   └── <dataset_name>/
        ├── transform_chained/
        │   └── <dataset_name>/
        ├── flow1x/
        │   └── <dataset_name>/
        ├── flow0_1x/
        │   └── <dataset_name>/
        ├── flow_final/
        │   └── <dataset_name>/
        ├── mesh_solved/
        │   └── <dataset_name>/
        └── inverse_map/
            └── <dataset_name>/
```
`xy_intermediate` contains one stack per dataset, which saves the stitched slices.
* `<dataset_name>` ([z, y, x], uint8): stitched slices
* `<dataset_name>_mask` ([z, y, x], bool): mask of where data is present in the stitched slices, used to ignore the black padding during Z alignment
* `<a>_<b>_fused` and `<a>_<b>_fused_mask`: only written by `xy fuse`, when stacks `<a>` and `<b>` overlap on the same Z slices. The names of the stacks it replaces are stored under `fused_from` in its `.zattrs`, which is how `z prep` knows to use the fused stack instead of the original ones over that Z range.

`z_intermediate` contains data produced by each Z alignment steps to transform xy_intermediate slices into one contiguous Z stack.
* `transform` ([z, 2, 3]): affine transformations roughly aligning each slice to the previous one
* `transform_chained` ([z, 2, 3]): the same transformations chained along the Z axis, so that each slice is placed in a common reference frame
* `flow1x` ([z, 4, y, x]): flow fields computed at full resolution, i.e. the translation of each patch needed to align a slice to the previous one. `y` and `x` are the image size divided by the stride.
* `flow0_1x` ([z, 4, y, x]): same as above, computed at a lower resolution. The name follows `scale_flow` in the Z config (here 0.1), so it may differ based on the chosen value.
* `flow_final` ([z, 2, y, x]): both flows cleaned up and combined into one
* `mesh_solved` ([2, z, y, x]): relaxed mesh, used as the starting point for the next stack along an alignment path
* `inverse_map` ([2, z, y, x]): inverted relaxed mesh, used to warp the slices into the final output

For flows, the 4 channels are the X and Y translations followed by two quality measures of the match. Channels of `mesh_solved` and `inverse_map` are the X and Y components.

## Final outputs

The final step of the pipeline is to warp and render each slice to obtain one aligned stack. It produces three datasets with different purposes for downstream analysis and visualization.

```
project_dir/
├── config/
└── project_dir.zarr/
    ├── xy_intermediate/
    ├── z_intermediate/
    ├── <project_name>/
    ├── 10x_<project_name>/
    └── <project_name>_mask/
```

* `<project_name>` ([z, y, x], uint8): final aligned data at full resolution
* `10x_<project_name>` ([z, y/10, x/10], uint8): the same data downsampled in Y and X, here by a factor of 10 (set with `-ds` during `z prep`). This lightweight stack is cheap to store and easy to load into memory to inspect the quality of alignment.
* `<project_name>_mask` ([z, y, x], bool): mask of where data is present, as opposed to the black padding

## Dataset attributes

Every dataset of the output zarr container contains a `.zattrs` file alongside the data. This file contains information about the corresponding dataset.
Among other things, resolution is stored in the attribute files for the xy aligned data and final output. The downsampled output `10x_<project_name>` stores its own resolution, i.e. the YX resolution multiplied by the downsampling factor.

ZYX resolution is notably written twice under the `voxel_size` and `resolution` keys for compatibility with libraries required for downstream steps (such as segmentation). Currently, Z resolution is hard-coded to be 50 nm.

## Progress database (MongoDB)

At each step, every slice of the project logs a document in a MongoDB database to keep track of progress. The MongoDB database host can be defined by the user (`localhost` by default), and the database name is `alignment_<project_name>`. 
Each dataset defined during the XY alignment step then creates its own collection `<dataset_name>` containing one document per step and per slice. Every document contains at least the key `step_name` to point to the pipeline step that produced it. 

When a job is interrupted and started over, a check of the database ensures that no slice is processed twice, and computations thus start where the job was left off. The `--start-over` flag starts by wiping every progress document for the corresponding step so that every slice is processed from scratch. To reprocess only some stacks, use `--wipe-progress <dataset_name> ...` instead, which wipes the progress of these stacks only. 