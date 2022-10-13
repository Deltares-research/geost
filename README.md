# pysst
The Python Subsurface Toolbox (pysst) package is designed to handle all common formats of subsurface point data (Boreholes and CPT's). It provides analysis and export methods that can be applied generically to the loaded data. 


## Supported borehole and CPT formats
- pysst .parquet file (complete)
- Dino XML geological boreholes (planned)
- BRO XML geotechnical boreholes (planned)
- BRO XML soil boreholes (planned)
- GEF boreholes (planned)
- GEF CPT's (implemented through pygef)
- BRO XML CPT's (planned)
- BRO geopackage CPT's (planned)

## Basic usage
This example shows how you could load some borehole data
```
from pysst import read_sst_cores

boreholes = read_sst_cores(r'c:\path\to\boreholes.parquet')
```

The instance 'boreholes' provides you with several methods to further select data from polygons, lines and points. In the below example a polygon is used for selection
```
import geopandas as gpd

study_area = gpd.read_file(r'c:\path\to\polygon_shapefile.shp)

boreholes_selected = boreholes.select_from_polygons(study area)
```

The new instance 'boreholes_selected' contains only the boreholes within the study area polygon. You can export the result in various ways
```
boreholes_selected.to_parquet(r'c:\path\to_output.parquet)   # Write data to parquet file (or csv file, for that use the method to_csv)
boreholes_selected.to_ipf(r'c:\path\to_output.ipf)   # Write to iMod ipf file
boreholes_selected.to_shape(r'c:\path\to_output.shp)   # Write to shapefile or geopackage (only point locations and metadata)
boreholes_selected.to_vtk(r'c:\path\to_output.vtk)   # Write to vtk file
```


## Collaboration

You can contribute by testing, raising issues and making pull requests. Some general guidelines:

- Use new branches for developing new features or bugfixes
- Add unit tests (and test data) for new methods and functions. We use pytest.
- Add Numpy-style docstrings
- Use Black formatting 

## License
MIT license

