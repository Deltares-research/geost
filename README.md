# GeoST - Geological Subsurface Toolbox
[![License: MIT](https://img.shields.io/pypi/l/imod)](https://choosealicense.com/licenses/mit)
[![Lifecycle: experimental](https://lifecycle.r-lib.org/articles/figures/lifecycle-experimental.svg)](https://lifecycle.r-lib.org/articles/stages.html)
[![Build: status](https://img.shields.io/github/actions/workflow/status/deltares-research/geost/ci.yml)](https://github.com/Deltares-research/geost/actions)
[![codecov](https://codecov.io/gh/Deltares-research/geost/graph/badge.svg?token=HCNGLWTQ2H)](https://codecov.io/gh/Deltares-research/geost)
[![Formatting: ruff](https://camo.githubusercontent.com/18c26428c337f9d641fa09b629a3a03b514e8ac84b57974a0ed7d1b38e14e060/68747470733a2f2f696d672e736869656c64732e696f2f656e64706f696e743f75726c3d68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f61737472616c2d73682f727566662f6d61696e2f6173736574732f62616467652f76322e6a736f6e)](https://github.com/charliermarsh/ruff)

The Geological Subsurface Toolbox (geost) package is designed to be an easy-to-use Python interface for working with subsurface point data in The Netherlands (boreholes, well logs and CPT's). It provides selection, analysis and export methods that can be applied generically to the loaded data. It is designed to connect with other Deltares developments such as [iMod](https://gitlab.com/deltares/imod) and [DataFusionTools](https://bitbucket.org/DeltaresGEO/datafusiontools/src/master/).

The internal BoreholeCollection, LogCollection and CptCollection classes use [Pandas](https://pandas.pydata.org/) for storing data and header information. It utilizes a custom, lightweight validation module inspired by the [Pandera](https://pandera.readthedocs.io/en/stable/) API. For spatial functions [Geopandas](https://geopandas.org/en/stable/) is used. The package also supports reading/writing parquet and geoparquet files through Pandas and Geopandas respectively. 

GeoST is a work-in-progress and currently support a limited number of data sources.

## Installation (user)
In a Python > 3.12 environment, install the latest stable release using pip:

    pip install geost

Or the latest (experimental) version of the main branch directly from GitHub using:

    pip install git+https://github.com/Deltares-research/geost.git

## Installation (developer)
We use [Pixi](https://github.com/prefix-dev/pixi) for package management and workflows.

With pixi installed, navigate to the folder of the cloned repository and run:

    pixi install

To install all GeoST dependencies. Next install GeoST in editable mode by running the 
pixi task 'install':

    pixi run install

See the [Pixi documentation](https://pixi.sh/latest/) for more information.

## Supported borehole and CPT formats
- From local files
    - geost .parquet file (complete)
    - Dino csv geological boreholes (complete)
    - Dino XML geological boreholes (planned)
    - BRO XML geotechnical boreholes (planned)
    - BRO XML soil boreholes (planned)
    - GEF boreholes (planned)
    - GEF CPT's (complete)
    - BRO XML CPT's (planned)
    - BRO geopackage CPT's (planned)
    - Well log LAS files (planned)
    - Well log ASCII files (planned)
- Directly from the BRO (REST API) (all planned)
    - CPT
    - BHR-P
    - BHR-GT
    - BHR-G

## Features
After loading data from one of the supported formats it will automatically be validated. If the validation is succesful, a Collection object will be returned depending on your input data type (mixed CPT/well log/borehole collections are not allowed). A collection object consists of two main attributes: the **header** and **data**. The header contains a table with one entry per object and provides information about the name, location, surface level, and borehole/log/cpt start and end depths. The data attribute is a table that includes the data for every described layer (boreholes) or measurement (well logs, cpt's).

The collection object comes with a comprehensive set of methods that can be applied generically while ensuring that the header and data remain synchronized:

- Selection/slicing methods (e.g., objects within bounding box, within or close to geometries, based on depth and other conditions)   
- Export methods (e.g. to csv, parquet, geopackage, VTK, DataFusionTools, Kingdom* , etc)
- Datafusion methods (e.g. combining collections*, combining with data from maps, conversion of description protocols*)
- Miscellaneous methods (e.g. changing vertical/horizontal position reference system)

For a better overview of basic functionality, see the [Basics Tutorial](https://gitlab.com/deltares/tgg-projects/subsurface-toolbox/pysst/-/blob/main/tutorials/basics.ipynb?ref_type=heads).

## Contributing

You can contribute by testing, raising issues and making pull requests. Some general guidelines:

- Use new branches for developing new features or bugfixes. Use prefixes such as feature/ bugfix/ experimental/ to indicate the type of branch
- Add unit tests (and test data) for new methods and functions. We use pytest.
- Add Numpy-style docstrings
- Use Black formatting with default line lenght (88 characters)
- Update requirement.txt en environment.yml files if required

## License
MIT license (Note: may change to a copyleft license in the future, depending on Deltares management decisions)

