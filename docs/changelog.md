# Release notes

## v0.4.2

Replaced core header and data structures (e.g. `PointHeader`, `LayeredData`) in `Collection` instances by accessors on GeoDataFrames (header) and DataFrames (data). Now the header and data attributes of Collections have direct access to all Geopandas and Pandas methods. See the [GeoST accessors](./user_guide/accessors.ipynb) section in the User guide for detailed explanation of the new accessors. Other notable changes are Python 3.14 support and BHRGT grain size sample support in `geost.bro_api_read`

**Added**
- **Added** Python 3.14 support
- **Added** `geost.bro_api_read` now supports geotechnical borehole grainsize sample data: use "BHR-GT-samples" as object_type argument.
- **Added** `geost.bro_api_read` now supports selection bounding boxes and geometries in any coordinate reference system.
- **Added** `.gsthd` for Geopandas GeoDataFrames, see also [GeoST accessors](./user_guide/accessors.ipynb#header-gsthd-accessor).
- **Added** `.gstda` for Pandas DataFrames, see also [GeoST accessors](./user_guide/accessors.ipynb#data-gstda-accessor).
- **Added** `VoxelModel.slice_depth_interval` to slice voxelmodels between specific depth intervals using single values or 1D/2D elevation grids.
- **Added** `VoxelModel.most_common` to determine the most common unit (i.e. value) and corresponding thickness at every x,y-location.
- **Added** `VoxelModel.value_counts` to determine the value counts of unique values in a
variable, in the total variable or along a specific dimension.
- **Added** `VoxelModel.from_opendap` for generic voxel models.
- **Added** option to skip data validation in Collections entirely by setting `geost.config.validation.SKIP = True`
- **Added** Added basic analysis functions for BHR-GT sample data

**Other**
- Move `BoreholeCollection` and `CptCollection` to top-level import of package.
- Make Pyvista imports lazy to only import when used.
- Removed unused points in PyVista.UnstructuredGrid exports, reducing file size when saved to vtk file.
- Significantly sped up creation of geometries from x/y arrays in `geost.utils.dataframe_to_geodataframe` (e.g. used for creating headers from data)

[**Full Changelog**](https://github.com/Deltares-research/geost/compare/0.3.0...0.4.2)

## v0.3.0

Milestone update introducing improved integration with the BRO, 3D viewing/VTK features,
better data validation and configuration options.

**Added**
- **Added** XML file parsing functionality for BHR-P, BHR-G, BHR-GT, CPT and SFR objects
- **Added** `geost.bro_api_read` function to retrieve objects through the BRO REST API
- **Added** `VoxelModel.get_thickness` method
- **Added** `geost.config` module for global settings. See also the [Validation](./user_guide/validation.ipynb) section in the user guide.
- **Added** `LayeredData.to_pyvista_grid`, `DiscrteData.to_pyvista_grid` methods
- **Added** New user guide sections on XML parsing and data validation
- **Added** Example that showcases using the new `geost.bro_api_read` function
- **Added** Example that showcases creating thickness maps based on Voxel models
- **Added** Example that showcases new PyVista/VTK/3D export and viewing features

**Other**
- Renamed `VoxelModel.to_vtk` method to `VoxelModel.to_pyvista_grid` and changed behaviour (returns PyVista objects instead of writing a VTK file)
- Renamed `to_vtk` and `to_vtm` methods of LayeredData, DiscreteData and Collections to `to_pyvista_cylinders` and changed behaviour (returns PyVista objects instead of writing a VTK file)
- Changed to and integrated [Pandera](https://pandera.readthedocs.io/en/stable/) for dataframe validation
- License changed to GNU Lesser General Public License v3 (LGPLv3)

[**Full Changelog**](https://github.com/Deltares-research/geost/compare/0.2.9...0.3.0)

## v0.2.9

**Added**
- **Added** `VoxelModel.to_vtk` method
- **Added** `vtk.voxelmodel_to_pyvista_unstructured`

**Fixed**
- **Fixed** replace validation decorators with print statements by dedicated UserWarnings in data validation and other functions

[**Full Changelog**](https://github.com/Deltares-research/geost/compare/0.2.8...0.2.9)

## v0.2.8

**Added**
- **Added** generic reader for Geopackage files
- **Added** begin of reader for the "BRO CPT kenset" geopackage
- **Added** keyword arguments 'min_thickness' and 'min_depth' for method 'get_layer_top'

**Fixed**
- **Fixed** fix unintended loading of data in `GeoTop.from_opendap`
- **Fixed** use correct transformation of coordinates between lat,lon and x,y

**Other**
- Removed unnecessary transpose in `sample_along_line` for more generic behaviour

[**Full Changelog**](https://github.com/Deltares-research/geost/compare/0.2.6...0.2.8)

## v0.2.6

**Added**
- **Added** GeoST documentation and deployment of these docs on Github pages
- **Added** Voxel model support, including an implementation for GeoTOP (WIP)
- **Added** *CptCollection*, *DiscreteData* objects, including i/o and some basic analysis methods (WIP)
- **Added** Function *add_voxelmodel_variable* to add voxel model data to point data
- **Added** Data folder with example datasets, integrated with pooch
- **Added** Collection geopackage export of header and data
- **Added** Collection pickle export


**Fixed**
* **Fixed** *find_top_sand* function if no sand is present
* **Fixed** *find_area_labels* can now return multiple labels when passing an iterable of column names
* **Fixed** Newly added columns to the header are now preserved upon making selections and slices

[**Full Changelog**](https://github.com/Deltares-research/geost/compare/0.2.4...0.2.6)

## v0.2.4

**Added**
- **Added** export of BoreholeCollections to the Kingdom seismic interpretation software
- **Added** option to pass file location (instead of a Geodataframe) to the get_area_labels method

**Fixed**
* **Fixed** cumulative thickness returning NaN instead of 0 when queried layer is not present
* **Fixed** MergeError when calling a method that adds a column to the header object multiple times
* **Fixed** multiple Pandas setting-on-copy-warnings
* **Fixed** QGis3D export not using the collection's CRS

[**Full Changelog**](https://github.com/Deltares-research/geost/compare/0.2.2...0.2.4)

## v0.2.2

**Added**
- **Added** support of MacOS and Linux operating systems
- **Added** dynamic package versioning
- **Added** validation check to prevent duplicated columns

**Fixed**
- **Fixed** icons in readme
- **Fixed** select_by_values slow copy behaviour for large datasets

[**Full Changelog**](https://github.com/Deltares-research/geost/compare/0.2.1...0.2.2)

## v0.2.1

**Added**

**Fixed**
- **Fixed** adjustment of z-coordinates when using read_borehole_table

[**Full Changelog**](https://github.com/Deltares-research/geost/compare/0.2.0...0.2.1)

## v0.2.0

- Initial release of GeoST on PyPi
