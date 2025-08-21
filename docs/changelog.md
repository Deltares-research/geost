# Release notes

## v0.3.0

Milestone update introducing improved integration with the BRO, 3D viewing/VTK features,
better data validation and configuration options.

[**Full Changelog**](https://github.com/Deltares-research/geost/compare/0.2.9...0.3.0)

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


## v0.2.9

[**Full Changelog**](https://github.com/Deltares-research/geost/compare/0.2.8...0.2.9)

**Added**
- **Added** `VoxelModel.to_vtk` method
- **Added** `vtk.voxelmodel_to_pyvista_unstructured`

**Fixed**
- **Fixed** replace validation decorators with print statements by dedicated UserWarnings in data validation and other functions


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


## v0.2.6

[**Full Changelog**](https://github.com/Deltares-research/geost/compare/0.2.4...0.2.6)

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


## v0.2.4

[**Full Changelog**](https://github.com/Deltares-research/geost/compare/0.2.2...0.2.4)

**Added**
- **Added** export of BoreholeCollections to the Kingdom seismic interpretation software
- **Added** option to pass file location (instead of a Geodataframe) to the get_area_labels method

**Fixed**
* **Fixed** cumulative thickness returning NaN instead of 0 when queried layer is not present
* **Fixed** MergeError when calling a method that adds a column to the header object multiple times
* **Fixed** multiple Pandas setting-on-copy-warnings
* **Fixed** QGis3D export not using the collection's CRS


## v0.2.2

[**Full Changelog**](https://github.com/Deltares-research/geost/compare/0.2.1...0.2.2)

**Added**
- **Added** support of MacOS and Linux operating systems
- **Added** dynamic package versioning
- **Added** validation check to prevent duplicated columns

**Fixed**
- **Fixed** icons in readme
- **Fixed** select_by_values slow copy behaviour for large datasets


## v0.2.1

[**Full Changelog**](https://github.com/Deltares-research/geost/compare/0.2.0...0.2.1)

**Added**

**Fixed**
- **Fixed** adjustment of z-coordinates when using read_borehole_table


## v0.2.0

- Initial release of GeoST on PyPi
