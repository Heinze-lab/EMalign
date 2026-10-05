from sofima import warp
from ..io.store import write_data

def render_slice_z(destination, z, data, inv_map, data_bbox, flow_bbox, stride, return_render=False, parallelism=1):

    aligned = warp.warp_subvolume(data, data_bbox, inv_map, flow_bbox, stride, data_bbox, 'lanczos', parallelism=parallelism)

    if return_render:
        return aligned[0,0,...]
    else:
        return write_data(destination, aligned[0,0,...], z)