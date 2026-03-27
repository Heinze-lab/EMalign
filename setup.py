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
            'tqdm'
        ],
        extras_require={
            'neuroglancer': ['neuroglancer']
        },
        keywords=['python', 'alignment']
    )
