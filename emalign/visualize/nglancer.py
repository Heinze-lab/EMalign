'''
Functions related to neuroglancer viewer.
'''
import numpy as np


def start_nglancer_viewer(bind_address='localhost',
                          bind_port=33333,
                          layout='xy'):
    '''Start neuroglancer viewer on bind_port.

    Args:
        bind_address (str, optional): Address to bind the viewer to. Defaults to 'localhost'.
        bind_port (int, optional): Port on which to broadcast the viewer. Defaults to 33333.

    Returns:
        viewer (`neuroglancer.Viewer`): Neuroglancer viewer object corresponding to the currently active viewer.
            Use viewer.get_viewer_url() to print the URL.
    '''

    import neuroglancer

    neuroglancer.set_server_bind_address(bind_address=bind_address, bind_port=bind_port)
    viewer = neuroglancer.Viewer()

    with viewer.txn() as s:
        s.layout = layout

    return viewer


def data_to_LocalVolume(
                      data,
                      spatial_dims,
                      voxel_offset,
                      voxel_size,
                      vtype,
                      transpose=True
                       ):
    
    '''Convert array into neuroglancer.LocalVolume.
    
    Based on funlib implementation.

    Args:
        data (array-like): Array to turn into LocalVolume. Types and dimensions will be determined from array.
        spatial_dims (int): Number of spatial dimensions, up to 3.
        voxel_offset (list of `int`): Volume offset in voxels. Must be the same dimension as spatial_dims.
        voxel_size (list of `int`): Resolution in world unit (nm by default). Must be the same dimension as spatial_dims.
        vtype (str): Volume type. One of: segmentation, image

    Returns:
        local_volume (`neuroglancer.LocalVolume`): Array in local volume format to be added as a layer to a neuroglancer viewer.
    '''
    import neuroglancer

    if data.dtype == bool:
        data = data.astype(np.uint8)
    if transpose:
        data = data.T

    spatial_dim_names = ['t', 'x', 'y', 'z']
    channel_dim_names = ['b^', 'c^']

    dims = len(data.data.shape)
    channel_dims = dims - spatial_dims
    voxel_offset = [0] * channel_dims + list(voxel_offset)

    attrs = {
             'names': (channel_dim_names[-channel_dims:] if channel_dims > 0 else [])
             + spatial_dim_names[-spatial_dims:],
             'units': [''] * channel_dims + ['nm'] * spatial_dims,
             'scales': [1] * channel_dims + list(voxel_size),
            }

    dimensions = neuroglancer.CoordinateSpace(**attrs)
    local_volume = neuroglancer.LocalVolume(
                                            data=data,
                                            voxel_offset=voxel_offset,
                                            dimensions=dimensions,
                                            volume_type=vtype
                                           )
    return local_volume


def add_layers(arrays,
               viewer,
               names=[],
               voxel_offsets=[],
               voxel_sizes=[],
               vtypes=[],
               shaders=[],
               visible=True,
               transpose=True,
               viewer_layout='xy',
               clear_viewer=True):
    '''Add layers to a currently active neuroglancer viewer.

    Args:
        arrays (list of array-like): List of arrays to display. An extra dimension will be added if array.ndim == 2.
        viewer (neuroglancer.Viewer): Currently active neuroglancer viewer object.
        names (list of `str`, optional): List of names for neuroglancer layers corresponding to arrays. 
            If empty, names will be numbers within range(len(arrays)). Defaults to [].
        voxel_offsets (list, optional): List of voxel offsets corresponding to arrays
            If empty, will default to [0,0,0] for all arrays. Defaults to [].
        voxel_sizes (list, optional): List of voxel sizes corresponding to arrays. 
            If empty, will default to [1,1,1] for all arrays. Defaults to [].
        vtypes (list, optional): Volume types corresponding to arrays. One of: segmentation, image. 
            If empty, will be determined from array data type. Defaults to [].
        visible (bool, optional): Whether to start with these arrays as visible. Defaults to True.
        clear_viewer (bool, optional): Whether to clear the viewer of currently visible layers. Defaults to True.
    '''
    
    if not names:
        names = list(map(str, range(len(arrays))))
    if not voxel_offsets:
        voxel_offsets = [[0,0,0]]*len(arrays)
    if not voxel_sizes:
        voxel_sizes = [[1,1,1]]*len(arrays)
    if not vtypes:
        vtypes = [None]*len(arrays)
    if not shaders:
        shaders = [None]*len(arrays)

    layers = {}
    for i, arr in enumerate(arrays):
        name = names[i]
        voxel_offset = voxel_offsets[i]
        voxel_size = voxel_sizes[i]
        vtype = vtypes[i]


        if vtype is not None:
            assert vtype in ['segmentation', 'image']
        elif arr.dtype == np.uint8:
            if arr.max() == 1:
                # Mask
                vtype = 'segmentation'
            else:
                # Image
                vtype = 'image'
        elif arr.dtype == np.uint64 or arr.dtype == bool:
            vtype = 'segmentation'
        else:
            vtype = 'image'

        arr = arr[None, ...] if arr.ndim == 2 else arr
        layers[name] = data_to_LocalVolume(arr,
                                           len(arr.shape),
                                           voxel_offset,
                                           voxel_size,
                                           vtype,
                                           transpose
                                           )

    if clear_viewer:
        with viewer.txn() as s:
            s.layers.clear()
    
    with viewer.txn() as s:
        for name, layer in layers.items():
            s.layers.append(name=name, layer=layer)
            s.layers[name].visible = visible
            if shaders[i] is not None:
                s.layers[name].shader = shaders[i]

        s.layout=viewer_layout