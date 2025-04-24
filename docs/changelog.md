# Release notes

## v0.2.7

[**Full Changelog**](https://github.com/Deltares-research/geost/compare/0.2.4...0.2.5)

**Added**
- **Added** generic reader for Geopackage files
- **Added** begin of reader for the "BRO CPT kenset" geopackage

**Fixed**
- **Fixed** fix unintended loading of data in `GeoTop.from_opendap`

**Other**
- **Other** remove unnecessary transpose in `sample_along_line` for more generic behaviour


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

**Other**
- **Updated** Function docstrings
- **Updated** Pixi tasks for docs management


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

**Other**
- **Updated** readme
- **Removed** requirements.txt

## v0.2.2

[**Full Changelog**](https://github.com/Deltares-research/geost/compare/0.2.1...0.2.2)

**Added**
- **Added** support of MacOS and Linux operating systems
- **Added** dynamic package versioning
- **Added** validation check to prevent duplicated columns

**Fixed**
- **Fixed** icons in readme
- **Fixed** select_by_values slow copy behaviour for large datasets

**Other**
- **Updated** pixi environment

## v0.2.1

[**Full Changelog**](https://github.com/Deltares-research/geost/compare/0.2.0...0.2.1)

**Added**

**Fixed**
- **Fixed** adjustment of z-coordinates when using read_borehole_table

**Other**
- **Updated** readme
- **Updated** license

## v0.2.0

- Initial release of GeoST on PyPi