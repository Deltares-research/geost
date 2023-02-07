from pysst import read_sst_cores

all_dino_cores = read_sst_cores(
    r"c:\Users\onselen\Lokale data\DINO_Extractie_bovennaaronder_d20230201.parquet"
)

# Make a pre-selection based on the study area. Using a fast selection method like select_within_bbox
# Is advised to reduce the amount of points, which speeds up later analysis and selections.
all_dino_cores = all_dino_cores.select_within_bbox(155000, 180000, 495000, 515000)

# Create collection of dino cores that contain gyttja, detritus and dy.
cores_with_gy = all_dino_cores.select_by_values(
    "lith", ["GY", "DET", "DY", "V"], how="or"
)

# Get cumulative thicknesses of selected layers, add the data to header for later export
cores_with_gy.get_cumulative_layer_thickness(
    "lith", ["GY", "DET", "DY", "V"], include_in_header=True
)
# Get the total thickness of the cover layer
cores_with_gy.cover_layer_thickness(include_in_header=True)

# Get top of layers, add the data to header for later export. First we change the vertical reference system
# such that the result will be in meters below the surface
cores_with_gy.change_vertical_reference("depth")
cores_with_gy.get_layer_top("lith", ["GY", "DET", "DY", "V"], include_in_header=True)

# Export collections to shapefiles
cores_with_gy.to_shape(
    r"n:\Projects\11207000\11207168\B. Measurements and calculations\Jaar 2\Geologie gyttja\Shapefiles\cores_with_gy.shp"
)
