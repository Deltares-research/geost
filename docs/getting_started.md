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

GeoST is purely written in Python but it requires `Python 3.12` or higher. The package is distributed via the Python Package Index (Pypi). Installing it into any environment is as
easy as:

```{powershell}
pip install geost
```

## Latest experimental version
The latest or expirimental version can be installed via the [GitHub repository](https://github.com/Deltares-research/geost) of GeoST. Simply install the main branch by:

```{powershell}
pip install git+https://github.com/Deltares-research/geost.git
```

## Installation for developers
GeoST uses [Pixi](https://github.com/prefix-dev/pixi) for package management and workflows.

With pixi installed, navigate to the folder of the cloned repository and run the following 
to install all GeoST dependencies and the package itself in editable mode:

```{powershell}
pixi install
```

See the [Pixi documentation](https://pixi.sh/latest/) for more information.