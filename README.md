# GeoST - Geological Subsurface Toolbox
[![PyPI version](https://img.shields.io/pypi/v/geost.svg)](https://pypi.org/project/geost)
[![License: LGPLv3](https://img.shields.io/pypi/l/geost)](https://choosealicense.com/licenses/lgpl-3.0)
[![Lifecycle: experimental](https://lifecycle.r-lib.org/articles/figures/lifecycle-experimental.svg)](https://lifecycle.r-lib.org/articles/stages.html)
[![Build: status](https://img.shields.io/github/actions/workflow/status/deltares-research/geost/ci.yml)](https://github.com/Deltares-research/geost/actions)
[![codecov](https://codecov.io/gh/Deltares-research/geost/graph/badge.svg?token=HCNGLWTQ2H)](https://codecov.io/gh/Deltares-research/geost)
[![Formatting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/charliermarsh/ruff)

The Geological Subsurface Toolbox (GeoST) package is designed to be an easy-to-use Python interface for working with subsurface point data in The Netherlands (boreholes, well logs and CPT's). It provides selection, analysis and export methods that can be applied generically to the loaded data. It is designed to connect with other Deltares developments such as [iMod](https://github.com/Deltares/imod-python) and [DataFusionTools](https://publicwiki.deltares.nl/display/TKIP/Data+Fusion+Tools).

The internal BoreholeCollection, LogCollection and CptCollection classes use [Pandas](https://pandas.pydata.org/) for storing data and header information and  [Pandera](https://pandera.readthedocs.io/en/stable/) for data validation. For spatial functions [Geopandas](https://geopandas.org/en/stable/) is used. The package also supports reading/writing parquet and geoparquet files through Pandas and Geopandas respectively.

GeoST is a work-in-progress and aims to support an increasing number of data sources.

## Installation (user)
In a Python >= 3.12 environment, install the latest stable release using pip:

    pip install geost

Or the latest (experimental) version of the main branch directly from GitHub using:

    pip install git+https://github.com/Deltares-research/geost.git

## Installation (developer)
GeoST uses [Pixi](https://github.com/prefix-dev/pixi) for package management and workflows.

With pixi installed, navigate to the folder of the cloned repository and run the following to install all GeoST dependencies and the package itself in editable mode:

    pixi install

See the [Pixi documentation](https://pixi.sh/latest/) for more information. Next open
the Pixi shell by running:

    pixi shell

Finally install the pre-commit hooks that enable automatic checks upon committing changes:

    pre-commit install


## Documentation
All documentation can be found on our [GitHub pages](https://deltares-research.github.io/geost)

## Examples
For an overview of examples, go directly to the [examples on our GitHub pages](https://deltares-research.github.io/geost/examples.html).

We collect additional examples that make use of GeoST and other Subsurface Toolbox developments in
the [Deltares sst-examples repository](https://github.com/Deltares-research/sst-examples).

## Contributing
You can contribute by testing, raising issues and making pull requests. Some general guidelines:

- Use new branches for developing new features or bugfixes. Use prefixes such as feature/ bugfix/ experimental/ to indicate the type of branch
- Add unit tests (and test data) for new methods and functions using pytest.
- Add Numpy-style docstrings
- Use pre-commit (see installation for developers on this page)
