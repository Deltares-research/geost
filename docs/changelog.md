```{currentmodule}
geost
```
# Release notes

## v0.2.4

[**Full Changelog**](https://github.com/Deltares-research/geost/compare/0.2.2...0.2.4)

#### Added
- **Added** export of BoreholeCollections to the Kingdom seismic interpretation software
- **Added** option to pass file location (instead of a Geodataframe) to the get_area_labels method

#### Fixed
* **Fixed** cumulative thickness returning NaN instead of 0 when queried layer is not present
* **Fixed** MergeError when calling a method that adds a column to the header object multiple times
* **Fixed** multiple Pandas setting-on-copy-warnings
* **Fixed** QGis3D export not using the collection's CRS

#### Other
- **Updated** readme
- **Removed** requirements.txt

## v0.2.2

[**Full Changelog**](https://github.com/Deltares-research/geost/compare/0.2.1...0.2.2)

#### Added
- **Added** support of MacOS and Linux operating systems
- **Added** dynamic package versioning
- **Added** validation check to prevent duplicated columns

#### Fixed
- **Fixed** icons in readme
- **Fixed** select_by_values slow copy behaviour for large datasets

#### Other
- **Updated** pixi environment

## v0.2.1

[**Full Changelog**](https://github.com/Deltares-research/geost/compare/0.2.0...0.2.1)

#### Added

#### Fixed
- **Fixed** adjustment of z-coordinates when using read_borehole_table

#### Other
- **Updated** readme
- **Updated** license

## v0.2.0

- Initial release of GeoST on PyPi