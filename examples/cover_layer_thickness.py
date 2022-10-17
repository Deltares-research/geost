from pysst import read_sst_cores
from pysst.borehole import BoreholeCollection
import geopandas as gpd
import numpy as np

#############################################

# all_cores = read_sst_cores(
#     r"c:\Users\onselen\OneDrive - Stichting Deltares\Development\DinoCore\DINO_Extractie_bovennaaronder_d20220405.parquet"
# )
core_area = gpd.read_file(
    r"c:\Users\onselen\OneDrive - Stichting Deltares\Projects\Deklaagdikte Marc\dijk_shapes\Dijken_buffer_dissolved.gpkg"
)

# cores_in_area = all_cores.select_from_polygons(core_area) # 15244 stuks
# cores_in_area.to_parquet(
#     r"c:\Users\onselen\OneDrive - Stichting Deltares\Projects\Deklaagdikte Marc\boringen\boringen_binnendijk.parquet"
# )

deklaag_cores = read_sst_cores(
    r"c:\Users\onselen\OneDrive - Stichting Deltares\Projects\Deklaagdikte Marc\boringen\boringen_binnendijk.parquet"
)

# thickness = deklaag_cores.cover_layer_thickness()
# header = deklaag_cores.header.copy()
# header["dl_dik"] = thickness["cover_thickness"]
# header.to_file(
#     r"c:\Users\onselen\OneDrive - Stichting Deltares\Projects\Deklaagdikte Marc\resultaten\boringen_met_dl.gpkg"
# )

points = gpd.read_file(
    r"c:\Users\onselen\OneDrive - Stichting Deltares\Projects\Deklaagdikte Marc\wellen\wellen_nl_RD.gpkg"
)
points = gpd.sjoin(points, core_area)[points.columns]
gmm = gpd.read_file(
    r"c:\Users\onselen\Lokale data\bro-geomorfologischekaart.gpkg",
    layer="view_geomorphological_area",
)

thicknesses_mean_800 = []
thicknesses_mean_3_closest = []
landforms = []


def get_thicknesses(boreholes, points):
    distances = boreholes.header.geometry.apply(
        lambda d: points.distance(d)
    ).sort_values(by=0)

    local_cover_thicknesses = boreholes.cover_layer_thickness()
    # [local_cover_thicknesses["cover_thickness"].loc[idx] if np.isfinite(local_cover_thicknesses["cover_thickness"].loc[idx]) else np.nan for idx in distances.index]
    cover_thicknesses_3_closest = local_cover_thicknesses.loc[distances[0:3].index]
    thickness_3_closest = cover_thicknesses_3_closest["cover_thickness"].mean()
    thickness_mean_800 = local_cover_thicknesses["cover_thickness"].mean()
    return (thickness_3_closest, thickness_mean_800)


for i, point in points.iterrows():
    print(i)
    cores = deklaag_cores.select_from_points(point, buffer=800)
    point_simple = gpd.GeoDataFrame(geometry=[point.geometry])
    landform = gpd.sjoin(point_simple, gmm)["landformsubgroup_code"]
    cores_landforms = gpd.sjoin(cores.header, gmm)["landformsubgroup_code"]
    if not len(landform) == 0:
        landforms.append(landform)
        cores_indices = cores_landforms[cores_landforms.values == landform.values].index
        ok_boreholes = BoreholeCollection(
            cores.data.loc[cores.data["nr"].isin(cores.header.loc[cores_indices]["nr"])]
        )
        if ok_boreholes.n_points > 0:
            thickness_3_closest, thickness_mean_800 = get_thicknesses(
                ok_boreholes, point_simple
            )

            thicknesses_mean_3_closest.append(thickness_3_closest)
            thicknesses_mean_800.append(thickness_mean_800)

        elif cores.n_points > 0:
            thickness_3_closest, thickness_mean_800 = get_thicknesses(
                cores, point_simple
            )
            thicknesses_mean_3_closest.append(thickness_3_closest)
            thicknesses_mean_800.append(thickness_mean_800)
        else:
            thicknesses_mean_3_closest.append(np.nan)
            thicknesses_mean_800.append(np.nan)
    elif cores.n_points > 0:
        thickness_3_closest, thickness_mean_800 = get_thicknesses(cores, point_simple)
        thicknesses_mean_3_closest.append(thickness_3_closest)
        thicknesses_mean_800.append(thickness_mean_800)
        landforms.append("")
    else:
        thicknesses_mean_3_closest.append(np.nan)
        thicknesses_mean_800.append(np.nan)
        landforms.append("")

points["thick_3"] = thicknesses_mean_3_closest
points["thick_800"] = thicknesses_mean_800
points["landform"] = [lf.values[0] if type(lf) != str else lf for lf in landforms]

points["thick"] = [
    t3 if t3 != 0 and np.isfinite(t3) else t800
    for t3, t800 in zip(thicknesses_mean_3_closest, thicknesses_mean_800)
]
points.to_file(
    r"c:\Users\onselen\OneDrive - Stichting Deltares\Projects\Deklaagdikte Marc\resultaten\wellen_met_dl.gpkg"
)
