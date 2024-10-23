# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
import sys

sys.path.insert(0, os.path.abspath(".."))  # isort:skip

# -- Project information -----------------------------------------------------

project = "Geological Subsurface Toolbox"
copyright = "2024, Deltares"
author = "Deltares"

# The full version, including alpha/beta/rc tags
import geost

version = geost.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "pydata_sphinx_theme",
    "sphinx.ext.napoleon",
    "sphinx_design",
    "myst_parser",
]


def setup(app):
    app.add_css_file("custom.css")  # may also be an URL


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "*bro"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"
html_logo = "_static/geost_logo.png"
html_favicon = "_static/geost_logo.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "external_links": [],
    "show_toc_level": 1,
    "show_nav_level": 2,
    "navbar_align": "left",
    "use_edit_page_button": False,
    "header_links_before_dropdown": 6,
    "pygments_light_style": "tango",
    "pygments_dark_style": "one-dark",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Deltares-research/geost",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/geost",
            "icon": "fa-solid fa-cubes",
            "type": "fontawesome",
        },
    ],
}
