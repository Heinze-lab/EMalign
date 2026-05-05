from setuptools import setup, find_packages

VERSION = '0.1' 
DESCRIPTION = 'Align image tiles acquired with SBEM into an image stack. Using SOFIMA (by Google Research)'
LONG_DESCRIPTION = ''

# Setting up
setup(
        name='emalign', 
        version=VERSION,
        author='Valentin Gillet',
        author_email='valentin.gillet@biol.lu.se',
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[
            'numpy',
            'pandas',
            'networkx',
            'opencv-python',
            'tensorstore',
            'scipy',
            'pymongo',
            'tqdm',
            'sofima @ git+https://github.com/google-research/sofima.git@5c5642cded4b33ec7e975afc3d85f15c0126fb2e'
        ],
        extras_require={
            'neuroglancer': [
                'neuroglancer'
            ]
        },
        keywords=['python', 'alignment']
    )
