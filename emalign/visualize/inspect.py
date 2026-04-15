from concurrent import futures
from glob import glob

from ..io.tif import load_tif
from .nglancer import *


def check_stacks_to_invert(stack_list, 
                           num_workers=1, 
                           bind_address='localhost', 
                           bind_port=33333):
    
    viewer = start_nglancer_viewer(bind_address=bind_address,
                                   bind_port=bind_port)
    print('Neuroglancer viewer: ' + viewer.get_viewer_url())
    print('Please wait for images to load (CTRL+C to cancel).')

    to_invert = {}
    with futures.ThreadPoolExecutor(num_workers) as tpe:
        fs = {}
        for stack_path in sorted(stack_list):
            stack_name = stack_path.split('/')[-2]
            fs[stack_name] = tpe.submit(load_tif, 
                                        glob(stack_path + '*.tif')[0], 1, {})

        for i, (stack_name, f) in enumerate(fs.items()):
            arr = f.result()[0]
            add_layers([arr],
                        viewer,
                        names=[stack_name],
                        clear_viewer=True)
            
            answer = input(f'{str(i).zfill(2)}/{len(fs)} - Invert {stack_name}? (y/n) ').strip(' ')

            while answer not in ['y', 'n', '']:
                answer = input(f'{str(i).zfill(2)}/{len(fs)} - Please provide a valid answer for {stack_name}: (y/n) ')

            if answer == 'y' or answer == '':
                to_invert.update({stack_name: True})
            elif answer == 'n':
                to_invert.update({stack_name: False})
    return to_invert