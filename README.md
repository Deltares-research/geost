# pysst
[![License: MIT](https://img.shields.io/pypi/l/imod)](https://choosealicense.com/licenses/mit)
[![Lifecycle: experimental](https://lifecycle.r-lib.org/articles/figures/lifecycle-experimental.svg)](https://lifecycle.r-lib.org/articles/stages.html)
[![Build: status](https://gitlab.example.com/deltares/tgg-projects/subsurface-toolbox/pysst/badges/main/pipeline.svg)](https://gitlab.com/deltares/tgg-projects/subsurface-toolbox/pysst/-/pipelines)
[![Coverage](https://gitlab.example.com/deltares/tgg-projects/subsurface-toolbox/pysst/badges/main/coverage.svg)](https://gitlab.com/deltares/tgg-projects/subsurface-toolbox/pysst/-/pipelines)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

The Python Subsurface Toolbox (pysst) package is designed to handle all common formats of subsurface point data (Boreholes and CPT's). It provides selection, analysis and export methods that can be applied generically to the loaded data. It is designed to connect with other Deltares developments such as iMod Suite and DataFusionTools.

The internal BoreholeCollection and CptCollection dataclasses use [Pandas](https://pandas.pydata.org/) for storing data and header information. It utilizes a custom, lightweight validation module inspired by the [Pandera](https://pandera.readthedocs.io/en/stable/) API. For spatial functions [Geopandas](https://geopandas.org/en/stable/) is used. The package also supports reading/writing parquet and geoparquet files through Pandas and Geopandas respectively. 


## Supported borehole and CPT formats
- From local files
    - pysst .parquet file (complete)
    - Dino csv geological boreholes (complete)
    - Dino XML geological boreholes (planned)
    - BRO XML geotechnical boreholes (planned)
    - BRO XML soil boreholes (planned)
    - GEF boreholes (planned)
    - GEF CPT's (implemented through pygef)
    - BRO XML CPT's (planned)
    - BRO geopackage CPT's (planned)
- Directly from BRO (via SOAP webservice / REST API) (all planned)
    - CPT
    - BHR-P
    - BHR-GT
    - BHR-G

## Features
After loading data from one of the supported formats it will automatically be validated. If the validation is succesful, either a BoreholeCollection or CptCollection object will be returned depending on your input data type (mixed CPT/Borehole collections are not allowed). A collection object consists of two main attributes: the **header table** and the **data table**. The header attribute is a table that contains one entry per object and provides some general information about the name, location, surface level, and borehole/cpt start and end depths. The data attribute is a table that includes every layer and the associated data.

The collection object comes with a comprehensive set of methods that apply generically to boreholes and CPT's:

- Selection methods    
    - select_within_bbox                *Returns subselection of objects within a bounding box*
    - select_with_points                *Returns subselection of objects within a given distance from points*
    - select_with_lines                 *Returns subselection of objects within a given distance from lines*
    - select_within_polygons            *Returns subselection of objects within polygons* 
    - select_with_present_values        *Returns subselection of objects based on filtering values in the given data columns* 
    - select_in_depth_range             *Returns subselection of objects based on depth constraints* 
- Export methods
    - to_csv                            *Writes the data table to a csv file*
    - to_parquet                        *Writes the data table to a parquet file*
    - to_shape                          *Write the header table to a shapefile or geopackage for viewing locations in GIS*
    - to_geoparquet                     *Write the header table to a geoparquet file for viewing locations in GIS*
    - to_ipf                            *Write the header and data table to IPF (iMod Point File) for use in iMod Suite and legacy iMod*
    - to_vtm                            *Write data to vtm (multiple vtk) for viewing in 3D*
    - to_geodataclass                   *Write data to geodataclass for use in Deltares DataFusionTools*
- misc and other methods
    - append                            *Append another instance of the same type (so BoreholeCollection + BoreholeCollection or CptCollection + CptCollection)*
    - get_area_labels                   *Get labels of polygons that data objects are inside of. e.g. to find in which geomorphological units boreholes are located*
    - change_vertical_reference         *Change vertical reference system. e.g. from NAP to relative depth.*

Borehole and Cpt collections each have analysis methods that are specific to the data type. Please refer to the documentation for more information

## Usage examples
This example shows how you could load some borehole data
```
from pysst import read_sst_cores

boreholes = read_sst_cores(r'c:\path\to\boreholes.parquet')
```

The instance 'boreholes' provides you with several methods to further select data from polygons, lines and points. In the below example a polygon is used for selection
```
import geopandas as gpd

study_area = gpd.read_file(r'c:\path\to\polygon_shapefile.shp)

boreholes_selected = boreholes.select_within_polygons(study area)
```

The new instance 'boreholes_selected' contains only the boreholes within the study area polygon. You can make additional selections, do analyses on the data and export the result in various ways. Let's say we want to know the cumulative thickness of clay layers ("K" in the "lith" column) within the first 5 m for every borehole and export that information such that we can visualise it in GIS:
```
# Use slice_depth_interval method with 'depth' as vertical reference to get the first 5 m of every borehole 
# (by default it references NAP, then we would slice off everyhting below 5 m NAP, which is not what we want)
boreholes_selected_first_5m = boreholes_selected.slice_depth_interval(lower_boundary=5, vertical_reference="depth" )

# Get the cumulative thickness and add this data as a new column to the header (so we can export it later)
boreholes_selected_first_5m.get_cumulative_layer_thickness("lith", "K", include_in_header=True)

# Now export the result to a geoparquet for viewing in GIS (shapefile and geopackage are also possible)
boreholes_selected_first_5m.to_geoparquet(r'c:\path\to\output.geoparquet)  
```

Suppose we want to use a set of boreholes with the Deltares DataFusionTools and also export the files to vtm for 3D viewing in paraview. The full script would be:
```
from pysst import read_sst_cores

boreholes = read_sst_cores(r'c:\path\to\boreholes.parquet')                                     # Load data
boreholes_in_study_area = boreholes.select_within_bbox(150000, 160000, 420000, 430000)          # Select boreholes in study area
boreholes_in_study_area.to_vtm(r'c:\path\to\boreholes.vtm', ["lith", "organic_admixture"])      # Create vtm file of boreholes with specified columns
dftobject = boreholes_in_study_area.to_datafusiontools(["lith", "organic_admixture"])           # Create DFT "Data" objects using specified columns
```
The "dftobject" can now be used within DataFusionTools. See e.g. the [DFT tutorial](https://datafusiontools.readthedocs.io/en/latest/index.html)

## Contributing

You can contribute by testing, raising issues and making pull requests. Some general guidelines:

- Use new branches for developing new features or bugfixes. Use prefixes such as feature/ bugfix/ experimental/ to indicate the type of branch
- Add unit tests (and test data) for new methods and functions. We use pytest.
- Add Numpy-style docstrings
- Use Black formatting with default line lenght (88 characters)
- Update requirement.txt en environment.yml files if required

## License
MIT license

