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


## Examples
For an overview of examples, see the [examples on our GitHub pages](https://deltares-research.github.io/geost/examples.html).

We collect additional examples that make use of GeoST and other Subsurface Toolbox developments in
the [Deltares sst-examples repository](https://github.com/Deltares-research/sst-examples).

## Supported data and file formats
**From local files**:
- Tabular data of borehole, CPT, etc. (.parquet, .csv)
- Geological boreholes xml (BHR-G)
- Geotechnical boreholes xml (BHR-GT)
- Pedological boreholes xml (BHR-P)
- Cone Penetration Test xml/gef (CPT)
- Pedological soilprofile descriptions xml (SFR)
- BORIS (TNO borehole description software) xml

**Directly from the [BRO REST-API](https://www.bro-productomgeving.nl/bpo/latest/url-s-publieke-rest-services)**:
- BHR-G
- BHR-GT
- BHR-P
- CPT
- SFR

**BRO models**:
- GeoTOP: from local NetCDF or directly via [OPeNDAP server](https://dinodata.nl/opendap/)

*Planned*:
- BRO/PDOK geopackages: [BHR-G](https://service.pdok.nl/bzk/bro-geologisch-booronderzoek/atom/index.xml), [BHR-GT](https://service.pdok.nl/bzk/bro-geotechnischbooronderzoek/atom/v1_0/index.xml), [BHR-P](https://service.pdok.nl/bzk/brobhrpvolledigeset/atom/v1_1/index.xml), [CPT](https://service.pdok.nl/bzk/brocptvolledigeset/atom/v1_0/index.xml), [SFR](https://service.pdok.nl/bzk/bodem/bro-wandonderzoek/atom/index.xml)
- Well logs LAS/ASCII
- REGIS II
- Dino xml geological boreholes
- BHR-G gef

## Features
After loading data from one of the supported formats it will automatically be validated. If the validation is succesful, a Collection object will be returned depending on your input data type (mixed CPT/well log/borehole collections are not allowed). A collection object consists of two main attributes: the **header** and **data**. The header contains a table with one entry per object and provides information about the name, location, surface level, and borehole/log/cpt start and end depths. The data attribute is a table that includes the data for every described layer (boreholes) or measurement (well logs, cpt's).

The collection object comes with a comprehensive set of methods that can be applied generically while ensuring that the header and data remain synchronized:

- Selection/slicing methods (e.g., objects within bounding box, within or close to geometries, based on depth and other conditions)
- Export methods (e.g. to csv, parquet, geopackage, VTK, DataFusionTools, Kingdom , etc)
- Datafusion methods (e.g. combining collections*, combining with data from maps, conversion of description protocols*)
- Miscellaneous methods (e.g. changing vertical/horizontal position reference system)

For a better overview of basic functionality, see the [Basics Tutorial](https://github.com/Deltares-research/geost/tree/main/tutorials).

## Contributing

You can contribute by testing, raising issues and making pull requests. Some general guidelines:

- Use new branches for developing new features or bugfixes. Use prefixes such as feature/ bugfix/ experimental/ to indicate the type of branch
- Add unit tests (and test data) for new methods and functions using pytest.
- Add Numpy-style docstrings
- Use Black formatting with default line length (88 characters)
- Update requirement.txt en environment.yml files if required
