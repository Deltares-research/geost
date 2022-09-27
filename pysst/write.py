import numpy as np
import pyvista as pv
import geopandas as gpd
from shapely.geometry import Point


def borehole_to_vtk(
    table, entries, borehole_size: int = 1, vertical_factor: float = 1.0
) -> None:
    boreholes = table.groupby("nr")

    for i, borehole in boreholes:
        # define corner coordinates
        x_array = borehole.x.values
        y_array = borehole.y.values
        x_array = np.hstack([x_array, x_array[0]])
        y_array = np.hstack([y_array, y_array[0]])

        x_array_corner1 = x_array - borehole_size
        x_array_corner4 = x_array - borehole_size
        x_array_corner2 = x_array + borehole_size
        x_array_corner3 = x_array + borehole_size
        y_array_corner1 = y_array - borehole_size
        y_array_corner2 = y_array - borehole_size
        y_array_corner3 = y_array + borehole_size
        y_array_corner4 = y_array + borehole_size

        z_array = np.hstack([(borehole.mv + borehole.top).values, borehole.end.iloc[0]])

        corner1 = np.vstack([x_array_corner1, y_array_corner1, z_array]).transpose()
        corner2 = np.vstack([x_array_corner2, y_array_corner2, z_array]).transpose()
        corner3 = np.vstack([x_array_corner3, y_array_corner3, z_array]).transpose()
        corner4 = np.vstack([x_array_corner4, y_array_corner4, z_array]).transpose()

        vertices = np.vstack([corner1, corner2, corner3, corner4])
