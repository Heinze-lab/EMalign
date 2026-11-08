# Configuration files
Configuration files stored as JSON dictionaries are required for each alignment step. Most of them are created automatically from user input by the preparation steps (`xy prep` and `z prep`).

## Configuration directory

Configuration files are written in the `config` folder at the root of the project directory, given with `-p` to every step:

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

`xy_config` is created by `xy prep` and contains all configuration files necessary for XY alignment.
* `main_config.json`: general information about the project and parameters for XY alignment
* `xy_<dataset_name>.json`: one file per stack, with instructions and input paths specific to this stack

`fuse_config` is created by `xy fuse` on its first run, only if some stacks overlap on the same Z slices.
* `00_fuse_plan.json`: groups of stacks to fuse
* `fuse_<zmin>_<zmax>_<i>.json`: one file per group, with the stacks to fuse and the name of the fused stack

`z_config` is created by `z prep` and contains all configuration files necessary for Z alignment.
* `00_align_plan.json`: general information about the project and order in which stacks are aligned
* `z_<dataset_name>.json`: one file per stack, with instructions, parameters, and input path specific to this stack

See [OUTPUT.md](OUTPUT.md) for the datasets produced by each step.

## Configuration for Z alignment (user created)

This dictionary is a required input to `z prep` (`-cfg-z`). It contains parameters for flow computations, elastic optimization, and rendering.

```json
{
  "scale_flow": 0.1,
  "flow": {
    "patch_size": 200,
    "stride": 50
  },
  "clean": {
    "min_peak_ratio": 1.6,
    "min_peak_sharpness": 1.6,
    "max_magnitude": 80,
    "max_deviation": 350,
    "max_gradient": 0,
    "min_patch_size": 400
  },
  "mesh": {
    "dt": 0.001,
    "gamma": 0.5,
    "k0": 0.01,
    "k": 0.4,
    "num_iters": 1000,
    "max_iters": 100000,
    "stop_v_max": 0.01,
    "dt_max": 1000,
    "start_cap": 0.1,
    "final_cap": 1.0,
    "prefer_orig_order": true,
    "remove_drift": false
  },
  "warp": {
    "work_size": 512,
    "overlap": 1
  }
}
```


*   scale_flow (`float`): value between 0 and 1 specifying the scale at which to compute the downsampled flow fields for estimating local displacement. Lower values are faster but less precise, which can be desirable to account for large structures in the underlying data. Full resolution flows are always computed alongside downsampled ones.

*   flow (`dict`): dictionary of parameters for optical flow computation between consecutive slices:
    *   patch_size (`int`): size in pixels of the cross-correlation window. Larger patches are more robust to noise but reduce spatial resolution of the flow field.
    *   stride (`int`): spacing in pixels between adjacent flow vector estimates. Smaller values produce a denser flow field at the cost of computation time.

*   clean (`dict`): dictionary of parameters for cleaning up the flow fields and combining the two scales. Each scale is cleaned up, then the gaps of the full scale flow are filled with the downsampled flow, and the combined flow is filtered again. Optional, missing values use the defaults shown above:
    *   min_peak_ratio (`float`): minimum ratio between the highest and second highest peak of the cross-correlation. Vectors with a less distinct match are discarded.
    *   min_peak_sharpness (`float`): minimum sharpness of the cross-correlation peak. Vectors with a flatter peak are discarded.
    *   max_magnitude (`float`): maximum allowed absolute magnitude of a flow vector, in pixels. Discards vectors from regions with larger displacements. Applied at both scales, in the pixels of each scale.
    *   max_deviation (`float`): maximum allowed deviation of a flow vector from the median of its 3x3 neighbourhood, in pixels. Vectors exceeding this are treated as outliers and discarded. Applied to each scale and to the combined flow. 0 disables the check.
    *   max_gradient (`float`): maximum allowed difference between neighbouring vectors of the combined flow, in pixels. 0 disables the check.
    *   min_patch_size (`int`): minimum size, in flow vectors, of a connected group of valid vectors in the combined flow. Smaller groups are discarded. 0 disables the check.

*   mesh (`dict`): dictionary of parameters for elastic mesh relaxation:
    *   dt (`float`): integration time step. Large values are faster but may cause the solver to overshoot past optimal solutions.
    *   gamma (`float`): viscosity/damping coefficient. Higher values suppress oscillations and improve stability, too high and convergence slows down.
    *   k0 (`float`): spring constant for inter-section springs, connecting mesh nodes between consecutive slices (across Z). Controls how tightly neighboring sections are pulled into alignment with each other.
    *   k (`float`): spring constant for intra-section springs, connecting mesh nodes within the same slice (across XY). Controls the internal rigidity of each section's mesh.
    *   num_iters (`int`): number of time steps to execute at once.
    *   max_iters (`int`): maximum number of iterations, acting as a safety cutoff if the mesh does not converge. Relaxation may stop earlier if `stop_v_max` is reached.
    *   stop_v_max (`float`): convergence threshold. Relaxation stops when the maximum vertex velocity drops below this value. Smaller values yield a more converged mesh at the cost of more iterations.
    *   dt_max (`float`): maximum total simulated time allowed for integration.
    *   start_cap (`float`): velocity clipping applied at the start of relaxation. Limits per-iteration displacements to prevent large initial jumps.
    *   final_cap (`float`): velocity clipping applied in later iterations. 
    *   prefer_orig_order (`bool`): if `true`, preserves the original ordering of mesh vertices during relaxation, preventing mesh folding.
    *   remove_drift (`bool`): if `true`, attempts to remove cumulative drift during mesh relaxation. **Note**: in practice this has been observed to introduce drift rather than remove it. Whether that is user error or data specific is unclear.

*   warp (`dict`): dictionary of parameters for applying the displacement map to the image data:
    *   work_size (`int`): tile size in pixels used when processing the image during warping. Larger values are faster but require more memory.
    *   overlap (`int`): overlap in pixels between adjacent work tiles, used to avoid boundary artifacts.

Definitions shown here are derived from SOFIMA documentation.

Values in `flow` are argument for [sofima.flow_field.flow_field](https://github.com/google-research/sofima/blob/efd7fd6348a692d721c609bdb2aea632365bf474/flow_field.py#L474).

Values in `clean` are arguments for [sofima.flow_utils.clean_flow](https://github.com/google-research/sofima/blob/efd7fd6348a692d721c609bdb2aea632365bf474/flow_utils.py#L37) and [sofima.flow_utils.reconcile_flows](https://github.com/google-research/sofima/blob/efd7fd6348a692d721c609bdb2aea632365bf474/flow_utils.py#L81).

Values in `mesh` are arguments for [sofima.mesh.IntegrationConfig](https://github.com/google-research/sofima/blob/efd7fd6348a692d721c609bdb2aea632365bf474/mesh.py#L283). Any of its fields can be given, except `stride` which is inherited from the `flow` dictionary.

`clean` and `mesh` parameters can be changed after `z prep` with `z flow --set name=value [name=value ...] [--stacks STACK ...]`. Parameter names are unique across both steps. The values are written into the `clean_config` and `mesh_config` of the dataset configs, and the steps they affect are run again for the stacks whose parameters changed. The parameters used are also recorded in the `flow_final` and `mesh_solved` stores, so editing the dataset configs by hand has the same effect.

Values in `warp` are argument for [sofima.warp.ndimage_warp](https://github.com/google-research/sofima/blob/efd7fd6348a692d721c609bdb2aea632365bf474/warp.py#L189).