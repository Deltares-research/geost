from pysst import read_sst_cores, read_gef_cpts, read_sst_cpts
from pysst.utils import excel_to_parquet

# Read from local DINO borehole parquet file and select
all_dino_cores = read_sst_cores(
    r"c:\Users\onselen\Lokale data\DINO_Extractie_bovennaaronder_d20230201.parquet"
)
cores = all_dino_cores.select_within_bbox(128500, 130000, 504000, 504500)

# Read cpts and turn into boreholecollection with lithologies
cpts = read_sst_cpts(
    r"n:\Projects\11206500\11206761\B. Measurements and calculations\3D-SSM Purmerend Casus\Veldwerk\Resultaten\Purmerend DualEM\Model training\Training_CPTs\CPT_data.parquet"
)
cpts.add_lithology()
cpts_as_borehole = cpts.as_boreholecollection()

# Read excel-entered hand boreholes, convert to pysst readable parquet and create collection
excel_to_parquet(
    r"n:\Projects\11206500\11206761\B. Measurements and calculations\3D-SSM Purmerend Casus\Veldwerk\Resultaten\Boringen\boreholes_Purmerend_final_fill_k.xlsx"
)
cores_hand = read_sst_cores(
    r"n:\Projects\11206500\11206761\B. Measurements and calculations\3D-SSM Purmerend Casus\Veldwerk\Resultaten\Boringen\boreholes_Purmerend_final_fill_k.parquet"
)

# Turn all into one object 'cores'
cores.append(cpts_as_borehole)
cores.append(cores_hand)

# Export to DFT Data objects as pickle file. Use columns lith, zm and ah as features
# (= lithology, sand median class and organic admixture). Use encode is false to
# maintain categorical string data (Random Forest model can handle these!)
cores.to_datafusiontools(
    ["lith"],
    r"c:\Users\onselen\OneDrive - Stichting Deltares\Projects\DFT_hackaton\dino_boreholes.pickle",
    encode=False,
)

# Also export the points to geoparquet to view locations in QGIS (>= 3.26)
cores.to_geoparquet(
    r"c:\Users\onselen\OneDrive - Stichting Deltares\Projects\DFT_hackaton\dino_boreholes.geoparquet"
)

# Export to vtm for visualisation
cores.to_vtm(
    r"c:\Users\onselen\OneDrive - Stichting Deltares\Projects\DFT_hackaton\dino_boreholes.vtm",
    ["lith"],
    radius=5,
)
