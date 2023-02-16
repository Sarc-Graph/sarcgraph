# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os

sys.path.insert(0, os.path.abspath("./sarcgraph"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "SarcGraph"
copyright = "2023, Saeed Mohammadzadeh"
author = "Saeed Mohammadzadeh"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "sphinx_material",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = []

autodoc_default_flags = ["members"]
autodoc_member_order = "bysource"
autodoc_mock_imports = ["numpy"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

copybutton_prompt_text = r">>> |\$ "
copybutton_prompt_is_regexp = True

# html_theme = "sphinx_material"
# html_static_path = ["_static"]
html_theme = "sphinx_material"
html_theme_options = {
    "color_primary": "indigo",
    "color_accent": "deep_orange",
    "repo_name": "SarcGraph",
    "repo_url": "https://github.com/Sarc-Graph/SarcGraph-2.0",
    "globaltoc_depth": 3,
    "globaltoc_collapse": True,
    "version_dropdown": True,
    "version_info": {
        "Release": "/",
        "Development": "/dev",
    },
    "master_doc": False,
    # 'nav_links': [
    #     {'href': 'index', 'title': 'Home', 'internal': True},
    #     {'href': 'installation', 'title': 'Installation', 'internal': True},
    #     {'href': 'api', 'title': 'API Reference', 'internal': True},
    #     {'href': 'contributing', 'title': 'Contributing', 'internal': True},
    # ],
}

# -- Added by me -------------------------------------------------------------
source_dir = "docs"
