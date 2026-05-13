# Installation
```{toctree}
---
maxdepth: 2
caption: Getting Started
hidden:
---

Installation <self>
Introduction to GeoST <getting_started/introduction>
```

## Installation (user)
GeoST is distributed via the Python package index (PyPi) and Conda-forge. In a Python >= 3.13 environment, install the latest stable release in your environment with either [pip](https://pypi.org/project/pip/), [Pixi](https://pixi.prefix.dev/latest/) or [Conda](https://conda.org/):

```
pip install geost
```
```
pixi add geost
```
When using Conda, make sure to specify the conda-forge channel:
```
conda install -c conda-forge geost
```
Or when using [miniforge](https://github.com/conda-forge/miniforge), which uses the conda-forge channel by default:
```
conda install geost
```

## Installation for developers
GeoST uses [Pixi](https://github.com/prefix-dev/pixi) for package and workflow management.

With pixi installed, navigate to the folder of the cloned repository and run the following
to install all GeoST dependencies and the package itself in editable mode:

```
pixi install
```

See the [Pixi documentation](https://pixi.sh/latest/) for more information. Next open
the Pixi shell by running:

```
pixi shell
```

Finally install the pre-commit hooks that enable automatic checks upon committing changes:

```
pre-commit install
```
