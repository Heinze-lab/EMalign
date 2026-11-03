project = 'emalign'
author = 'Valentin Gillet'
copyright = '2026, Valentin Gillet'

extensions = [
    'myst_nb',
    'sphinx_copybutton',
    'autoapi.extension',       # API reference from docstrings, without importing the package
    'sphinx.ext.napoleon',     # NumPy/Google-style docstrings
    'sphinx.ext.intersphinx',
    'sphinx.ext.githubpages',
]

autoapi_dirs = ['../emalign']
autoapi_root = 'api'
autoapi_add_toctree_entry = True

nb_execution_mode = 'off'
myst_enable_extensions = ['colon_fence']
exclude_patterns = [
    '_build', 
    '**.ipynb_checkpoints', 
    'requirements.txt',
    'tutorials/raw_data_earwig']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    'github_url': 'https://github.com/Heinze-lab/EMalign',
    'external_links': [
        {'name': '← Project home', 'url': 'https://heinze-lab.github.io/vem-toolkit.github.io/'},
    ],
}
