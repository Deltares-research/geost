---
html_theme.sidebar_secondary.remove:
---

# Examples

On this page you will find links to examples and projects that ultilize GeoST for 
managing subsurface data.

````{grid} 1 2 2 4
:gutter: 3
```{grid-item-card} Retrieve BRO data
:text-align: center
:link: ./examples/retrieve_bro_data.html
:link-type: url
:img-bottom: ./_static/dike_section.png
Extracting data from XML files from BRO or other sources.
```
```{grid-item-card} BRO soil cores using the BRO API
:text-align: center
:link: ./examples/bro_soil_cores.html
:link-type: url
:img-bottom: ./_static/example_bro_soil.png
A simple analysis of soil cores that are directly retrieved from the BRO.
```
```{grid-item-card} Thickness maps from GeoTOP
:text-align: center
:link: ./examples/geotop_thickness_maps.html
:img-bottom: ./_static/geotop_thickness_example.png
Create thickness maps of GeoTOP units with only a few lines of code.
```
```{grid-item-card} Combining GeoTOP and CPTs
:text-align: center
:link: ./examples/combine_geotop_with_cpts.html
:img-bottom: ./_static/cpts_usp.png
Combining information from a voxelmodel (GeoTOP) with point data (CPTs).
```
```{grid-item-card} GeoST + PyVista: 3D export features
:text-align: center
:link: ./examples/boreholes_geotop_3d.html
:img-bottom: ./_static/geotop_cpt_ic_model.png
Showcasing GeoST's VTK export and 3D viewing features powered by PyVista.
```
````

## External examples

Explore the use of GeoST subsurface modelling and data analysis workflows

````{grid} 1 2 2 4
```{grid-item-card} GeoST + scikit-learn (simple)
:text-align: center
:link: https://github.com/Deltares-research/sst-examples/blob/main/predict_sand_thickness/predict_sand_thickness_simple.ipynb
:img-bottom: ./_static/example_sand_thickness.png
Using geological borehole data to predict the thickness of sand using decision tree 
algorithms provided by scikit-learn.
```
```{grid-item-card} GeoST + scikit-learn (advanced)
:text-align: center
:link: https://github.com/Deltares-research/sst-examples/blob/main/predict_sand_thickness/predict_sand_thickness_advanced.ipynb
:img-bottom: ./_static/example_sand_thickness.png
Using geological borehole data to predict the thickness of sand using decision tree 
algorithms provided by scikit-learn.
```
````