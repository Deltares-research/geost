from pysst import read_sst_cores, read_gef_cpt
from pysst import excel_to_parquet, csv_to_parquet

# excel_to_parquet(
#     r"n:\Projects\11206500\11206761\B. Measurements and calculations\3D-SSM Purmerend Casus\Veldwerk\Resultaten\Boringen\boreholes.xlsx"
# )

# collection = read_sst_cores(
#     r"n:\Projects\11206500\11206761\B. Measurements and calculations\3D-SSM Purmerend Casus\Veldwerk\Resultaten\Boringen\boreholes.parquet"
# )

# collection2 = read_sst_cores(
#     r"n:\Projects\11206500\11206761\B. Measurements and calculations\3D-SSM Purmerend Casus\Veldwerk\Resultaten\Boringen\boreholes.parquet",
# )

# collection.append(collection2)

# cpts = read_gef_cpt(
#     r"n:\Projects\11206500\11206761\B. Measurements and calculations\3D-SSM Purmerend Casus\Veldwerk\Data\van HHNK\sonderingen"
# )


dino = read_sst_cores(
    r"c:\Users\onselen\OneDrive - Stichting Deltares\Development\DinoCore\DINO_Extractie_bovennaaronder_d20220405.parquet"
)

excel_to_parquet()
