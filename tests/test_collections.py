import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import pyvista as pv
import xarray as xr
from numpy.testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
)
from shapely.geometry import LineString, Point, Polygon

from geost.base import Collection


@pytest.fixture
def update_raster():
    x_coors = np.arange(127000, 128500, 500)
    y_coors = np.arange(503000, 501500, -500)
    data = np.ones((3, 3))
    array = xr.DataArray(data, {"x": x_coors, "y": y_coors})
    return array


class TestCollection:
    @pytest.mark.unittest
    def test_init_from_data(self, borehole_data):
        with pytest.warns() as record:
            collection = Collection(borehole_data)

        assert (
            str(record[0].message)
            == "Header is None, setting the header from the given data."
        )
        # TODO: `record[1]` has unexpected validation warning is thrown, check
        assert ("Setting the header without an active geometry column.") in str(
            record[1].message
        )

        assert isinstance(collection, Collection)
        assert isinstance(collection.header, gpd.GeoDataFrame)
        assert isinstance(collection.data, pd.DataFrame)
        assert not collection.header_has_geometry
        assert not collection.has_inclined
        assert collection.vertical_reference == 5709
        assert collection._nr == "nr"

        with pytest.raises(
            KeyError,
            match="Data table must contain a column identifying the survey IDs.",
        ):
            Collection(borehole_data.rename(columns={"nr": "invalid"}))

    @pytest.mark.unittest
    def test_init_from_header_and_data(self, borehole_data):
        header = borehole_data.drop_duplicates("nr").reset_index(drop=True)
        assert isinstance(
            header, pd.DataFrame
        )  # Just to be sure it is a DataFrame and not a GeoDataFrame

        with pytest.warns() as record:
            col1 = Collection(borehole_data, header=header)

        # TODO: `record[0]` has unexpected validation warning is thrown, check
        assert ("Setting the header without an active geometry column.") in str(
            record[0].message
        )

        assert isinstance(col1, Collection)
        assert isinstance(col1.header, gpd.GeoDataFrame)
        assert isinstance(col1.data, pd.DataFrame)
        assert not col1.header_has_geometry
        assert col1._nr == "nr"

        header = gpd.GeoDataFrame(
            header, geometry=gpd.points_from_xy(header["x"], header["y"]), crs=28992
        )
        col2 = Collection(borehole_data, header=header)
        assert col2.header_has_geometry
        assert (
            not col1.header_has_geometry
        )  # Make sure the attribute is not shared between instances

        with pytest.raises(
            KeyError,
            match="Header table must contain a column identifying the survey IDs",
        ):
            Collection(borehole_data, header=header.rename(columns={"nr": "invalid"}))

        with pytest.raises(
            ValueError,
            match="Column identifying survey IDs in data and header must have the same name",
        ):
            Collection(
                borehole_data, header=header.rename(columns={"nr": "bro_id"})
            )  # bro_id is a valid survey ID name, but different names in header and data raise an error

    @pytest.mark.unittest
    def test_init_empty(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            collection = Collection()  # Should not raise warnings

        assert isinstance(collection, Collection)
        assert collection.header.empty
        assert collection.data.empty
        assert not collection.header_has_geometry
        assert collection._nr is None

        collection = Collection(pd.DataFrame())
        assert isinstance(collection, Collection)
        assert collection.header.empty
        assert collection.data.empty
        assert not collection.header_has_geometry
        assert collection._nr is None

        collection = Collection(pd.DataFrame(), header=pd.DataFrame())
        assert isinstance(collection, Collection)
        assert collection.header.empty
        assert collection.data.empty
        assert not collection.header_has_geometry
        assert collection._nr is None

    @pytest.mark.unittest
    def test_init_only_header(self, borehole_data):
        header = borehole_data.drop_duplicates("nr").reset_index(drop=True)
        with pytest.raises(
            ValueError,
            match="Header was provided but data is None. A header cannot exist without a corresponding data table",
        ):
            Collection(header=header)

    @pytest.mark.unittest
    def test_get(self, borehole_collection):
        selection = borehole_collection.get("A")
        assert selection.header.iloc[0, 0] == "A"
        assert_array_equal(selection.data["nr"].unique(), ["A"])

        selection = borehole_collection.get(["A", "B"])
        assert list(selection.header["nr"].unique()) == ["A", "B"]
        assert_array_equal(selection.data["nr"].unique(), ["A", "B"])

    @pytest.mark.unittest
    def test_add_header_column_to_data(self, borehole_collection):
        borehole_collection.header["test_data"] = [
            i for i in range(len(borehole_collection.header))
        ]
        borehole_collection.add_header_column_to_data("test_data")
        assert_array_equal(
            borehole_collection.data["test_data"], np.repeat([0, 1, 2, 3, 4], 5)
        )

    @pytest.mark.unittest
    def test_change_vertical_reference(self, borehole_collection):
        assert (
            1 == 2
        )  # Make test fail to not forget to update the implementation of this method
        assert borehole_collection.vertical_reference == 5709
        borehole_collection.change_vertical_reference("Ostend height")
        assert borehole_collection.vertical_reference == 5710

    @pytest.mark.unittest
    def test_change_horizontal_reference(self, borehole_collection):
        assert (
            1 == 2
        )  # Make test fail to not forget to update the implementation of this method
        assert borehole_collection.horizontal_reference == 28992
        borehole_collection.change_horizontal_reference(32631)
        assert borehole_collection.horizontal_reference == 32631
        assert_almost_equal(
            borehole_collection.data["x"][0], borehole_collection.header["x"][0]
        )
        assert_almost_equal(borehole_collection.data["x"][0], 523403.281803, decimal=5)
        assert_almost_equal(
            borehole_collection.data["y"][0], borehole_collection.header["y"][0]
        )
        assert_almost_equal(borehole_collection.data["y"][0], 5313546.187669, decimal=5)

    @pytest.mark.unittest
    def test_change_horizontal_reference_with_inclined(self, nlog_borehole_collection):
        assert nlog_borehole_collection.horizontal_reference == 28992
        nlog_borehole_collection.change_horizontal_reference(32631)
        assert nlog_borehole_collection.horizontal_reference == 32631
        assert_almost_equal(
            nlog_borehole_collection.header["x"][0], 532641.76, decimal=2
        )
        assert_almost_equal(
            nlog_borehole_collection.header["y"][0], 6.170701e6, decimal=0
        )
        assert_almost_equal(
            nlog_borehole_collection.data["x_bot"][0], 532650.77, decimal=2
        )
        assert_almost_equal(
            nlog_borehole_collection.data["y_bot"][0], 6.170700e6, decimal=0
        )

    @pytest.mark.unittest
    def test_reset_header(self, borehole_collection):
        borehole_collection.reset_header()
        assert borehole_collection.horizontal_reference == 28992

    @pytest.mark.unittest
    def test_select_within_bbox(self, borehole_collection):
        selected = borehole_collection.select_within_bbox(1.5, 1.5, 3.5, 5)
        assert isinstance(selected, Collection)
        assert all(selected.header["nr"].unique() == ["A", "D"])
        assert all(selected.data["nr"].unique() == ["A", "D"])

        with pytest.raises(
            TypeError,
            match="Method 'select_within_bbox' requires a header with a valid geometry column.",
        ):
            borehole_collection.header = borehole_collection.header.drop(
                columns="geometry"
            )
            borehole_collection.select_within_bbox(1.5, 1.5, 3.5, 5)

    @pytest.mark.unittest
    def test_select_with_points(self, borehole_collection):
        selection_points = [Point(1, 1), Point(4, 4), Point(1, 4), Point(4, 1)]
        selection_gdf = gpd.GeoDataFrame(
            {"id": [0, 1, 2, 3]}, geometry=selection_points
        )
        selection = borehole_collection.select_with_points(selection_gdf, 1.1)
        assert selection.n_points == 3

        selection_inverted = borehole_collection.select_with_points(
            selection_gdf, 1.1, invert=True
        )
        assert selection_inverted.n_points == 2

        with pytest.raises(
            TypeError,
            match="Method 'select_with_points' requires a header with a valid geometry column.",
        ):
            borehole_collection.header = borehole_collection.header.drop(
                columns="geometry"
            )
            borehole_collection.select_with_points(selection_gdf, 1.1)

    @pytest.mark.unittest
    def test_select_with_lines(self, borehole_collection):
        selection_lines = [LineString([[1, 1], [5, 5]]), LineString([[1, 5], [5, 1]])]
        selection_gdf = gpd.GeoDataFrame({"id": [0, 1]}, geometry=selection_lines)
        selection = borehole_collection.select_with_lines(selection_gdf, 0.6)
        assert selection.n_points == 2
        selection_inverted = borehole_collection.select_with_lines(
            selection_gdf, 0.6, invert=True
        )
        assert selection_inverted.n_points == 3

        with pytest.raises(
            TypeError,
            match="Method 'select_with_lines' requires a header with a valid geometry column.",
        ):
            borehole_collection.header = borehole_collection.header.drop(
                columns="geometry"
            )
            borehole_collection.select_with_lines(selection_gdf, 0.6)

    @pytest.mark.unittest
    def test_select_within_polygons(self, borehole_collection):
        selection_polygon = [Polygon(((2, 1), (5, 4), (4, 5), (1, 3)))]
        selection_gdf = gpd.GeoDataFrame({"id": [0]}, geometry=selection_polygon)

        selection = borehole_collection.select_within_polygons(selection_gdf)
        assert selection.n_points == 1

        selection_inverted = borehole_collection.select_within_polygons(
            selection_gdf, invert=True
        )
        assert selection_inverted.n_points == 4

        selection_buffer = borehole_collection.select_within_polygons(
            selection_gdf, buffer=0.7
        )
        assert selection_buffer.n_points == 2

        selection_inverted_buffer = borehole_collection.select_within_polygons(
            selection_gdf, buffer=0.7, invert=True
        )
        assert selection_inverted_buffer.n_points == 3

        with pytest.raises(
            TypeError,
            match="Method 'select_within_polygons' requires a header with a valid geometry column.",
        ):
            borehole_collection.header = borehole_collection.header.drop(
                columns="geometry"
            )
            borehole_collection.select_within_polygons(selection_gdf, 0.6)

    @pytest.mark.unittest
    def test_spatial_join(self, borehole_collection):
        label_gdf = gpd.GeoDataFrame(
            {"id": [1]}, geometry=[Polygon(((2, 1), (5, 4), (4, 5), (1, 4)))], crs=28992
        )
        # Return variant
        output = borehole_collection.spatial_join(label_gdf, "id")
        assert isinstance(output, gpd.GeoDataFrame)
        assert "id" in output.columns
        assert output.shape == (2, 7)

        # In-place variant
        borehole_collection.spatial_join(label_gdf, "id", include_in_header=True)
        assert "id" in borehole_collection.header.columns
        assert borehole_collection.header.shape == (5, 7)

        with pytest.raises(
            ValueError,
            match="The 'how' parameter is not allowed when include_in_header is True.",
        ):
            borehole_collection.spatial_join(
                label_gdf, "id", how="left", include_in_header=True
            )

        with pytest.raises(
            TypeError,
            match="Method 'spatial_join' requires a header with a valid geometry column.",
        ):
            borehole_collection.header = borehole_collection.header.drop(
                columns="geometry"
            )
            borehole_collection.spatial_join(label_gdf, "id")

    @pytest.mark.unittest
    def test_select_by_elevation(self, borehole_collection):
        selected = borehole_collection.select_by_elevation(top_min=0, top_max=0.25)
        assert isinstance(selected, Collection)
        assert_array_equal(selected.header["nr"], ["A", "C", "D"])
        assert_array_equal(selected.data["nr"].unique(), ["A", "C", "D"])
        assert selected.data.shape == (15, 8)

        selected = borehole_collection.select_by_elevation(end_max=-3, end_min=-4)
        assert_array_equal(selected.header["nr"], ["A", "B", "E"])
        assert_array_equal(selected.data["nr"].unique(), ["A", "B", "E"])
        assert selected.data.shape == (15, 8)

        # Test same selection with end column missing, will be computed from data and added
        # to header, result must be the same.
        borehole_collection.header.drop(columns="end", inplace=True)
        selected = borehole_collection.select_by_elevation(end_max=-3, end_min=-4)
        assert_array_equal(selected.header["nr"], ["A", "B", "E"])
        assert_array_equal(selected.data["nr"].unique(), ["A", "B", "E"])
        assert selected.data.shape == (15, 8)

    @pytest.mark.unittest
    def test_select_by_length(self, borehole_collection):
        sel = borehole_collection.select_by_length(min_length=3.5, max_length=5.0)
        assert_array_equal(sel.header["nr"], ["A", "B"])
        assert_array_equal(sel.data["nr"].unique(), ["A", "B"])
        assert sel.data.shape == (10, 8)

        # Test same selection with end column missing, will be computed from data and added
        # to header, result must be the same.
        borehole_collection.header.drop(columns="end", inplace=True)
        sel = borehole_collection.select_by_length(min_length=3.5, max_length=5.0)
        assert_array_equal(sel.header["nr"], ["A", "B"])
        assert_array_equal(sel.data["nr"].unique(), ["A", "B"])
        assert sel.data.shape == (10, 8)

    @pytest.mark.unittest
    def test_select_by_values(self, borehole_collection, cpt_collection):
        selected = borehole_collection.select_by_values("lith", ["V", "K"], how="or")
        assert_array_equal(selected.data["nr"].unique(), ["A", "B", "C", "D"])
        assert selected.data.shape == (20, 8)
        selected = borehole_collection.select_by_values("lith", ["V", "K"], how="and")
        assert_array_equal(selected.data["nr"].unique(), ["B", "D"])
        assert selected.data.shape == (10, 8)

        selected = borehole_collection.select_by_values(
            "lith", ["V", "K"], how="and", invert=True
        )
        assert_array_equal(selected.data["nr"].unique(), ["A", "C", "E"])
        assert selected.data.shape == (15, 8)

        selected = cpt_collection.select_by_values("qc", slice(15, 20))
        assert_array_equal(selected.data["nr"].unique(), ["b"])
        assert selected.data.shape == (10, 9)

    @pytest.mark.unittest
    def test_slice_depth_interval(self, borehole_collection, cpt_collection):
        # Test slicing with respect to depth below the surface.
        upper, lower = 0.6, 2.4
        sliced = borehole_collection.slice_depth_interval(upper, lower)
        assert isinstance(sliced, Collection)
        assert sliced.header.shape == (5, 6)
        assert sliced.data.shape == (14, 8)

        # Test slicing with respect to a vertical reference plane.
        nap_upper, nap_lower = -2, -3
        sliced = borehole_collection.slice_depth_interval(
            nap_upper, nap_lower, relative_to_vertical_reference=True
        )
        assert sliced.header.shape == (5, 6)
        assert sliced.data.shape == (11, 8)

        # Test slices that return empty objects.
        empty_slice = borehole_collection.slice_depth_interval(-2, -1)
        assert empty_slice.header.empty
        assert empty_slice.data.empty

        with pytest.raises(
            KeyError,
            match="Method 'slice_depth_interval' requires depth information in the DataFrame.",
        ):
            borehole_collection.data.drop(columns=["bottom"], inplace=True)
            borehole_collection.slice_depth_interval(0.6, 2.4)

        sliced = cpt_collection.slice_depth_interval(0.6, 4.4)
        assert isinstance(sliced, Collection)
        assert sliced.header.shape == (2, 6)
        assert sliced.data.shape == (11, 9)

        with pytest.raises(
            KeyError,
            match="Method 'slice_depth_interval' requires depth information in the DataFrame.",
        ):
            cpt_collection.data.drop(columns=["depth"], inplace=True)
            cpt_collection.slice_depth_interval(0.6, 4.4)

    @pytest.mark.unittest
    def test_slice_by_values(self, borehole_collection, cpt_collection):
        sliced = borehole_collection.slice_by_values("lith", "Z")
        assert_array_equal(sliced.header["nr"], ["A", "C", "D", "E"])
        assert np.all(sliced.data["lith"] == "Z")
        assert sliced.data.shape == (10, 8)
        assert sliced.n_points == 4

        sliced = borehole_collection.slice_by_values("lith", "Z", invert=True)
        assert_array_equal(sliced.header["nr"], ["A", "B", "C", "D"])
        assert ~np.any(sliced.data["lith"] == "Z")
        assert sliced.data.shape == (15, 8)
        assert sliced.n_points == 4

        sliced = cpt_collection.slice_by_values("qc", slice(15, 20))
        assert_array_equal(sliced.header["nr"], ["b"])
        assert sliced.data["qc"].between(15, 20, inclusive="both").all()
        assert sliced.data.shape == (8, 9)

    @pytest.mark.unittest
    def test_select_by_condition(self, borehole_collection):
        # same selection with this method is used in test_data_objects
        selected = borehole_collection.select_by_condition(
            borehole_collection.data["lith"] == "V"
        )
        assert isinstance(selected, Collection)
        assert_array_equal(selected.header["nr"], ["B", "D"])
        assert (selected.data["lith"] == "V").all()
        assert selected.data.shape == (4, 8)

        selected = borehole_collection.select_by_condition(
            borehole_collection.data["lith"] == "V", invert=True
        )
        assert_array_equal(selected.header["nr"], ["A", "B", "C", "D", "E"])
        assert not (selected.data["lith"] == "V").any()
        assert selected.data.shape == (21, 8)

        selected = borehole_collection.select_by_condition(
            (borehole_collection.data["lith"] == "V")
            & (borehole_collection.data["bottom"] <= 2.5)
        )
        assert isinstance(selected, Collection)
        assert_array_equal(selected.header["nr"], ["B", "D"])
        assert (selected.data["lith"] == "V").all()
        assert (selected.data["bottom"] <= 2.5).all()
        assert selected.data.shape == (3, 8)

    @pytest.mark.unittest
    def test_to_qgis3d(self, borehole_collection):
        outfile = Path("temp.gpkg")
        borehole_collection.to_qgis3d(outfile)
        assert outfile.is_file()
        outfile.unlink()

    @pytest.mark.unittest
    def test_to_geoparquet(self, borehole_collection, tmp_path):
        outfile = tmp_path / r"test_export.geoparquet"
        borehole_collection.to_geoparquet(outfile)
        assert outfile.is_file()

    @pytest.mark.unittest
    def test_to_shape(self, borehole_collection, tmp_path):
        outfile = tmp_path / r"test_export.shp"
        borehole_collection.to_shape(outfile)
        assert outfile.is_file()

    @pytest.mark.unittest
    def test_to_geopackage(self, borehole_collection, tmp_path):
        outfile = tmp_path / r"test_export.gpkg"
        borehole_collection.to_geopackage(outfile)
        assert outfile.is_file()
        layers = gpd.list_layers(outfile)
        assert_array_equal(layers["name"], ["header", "data"])
        assert_equal(layers["geometry_type"][0], "Point")
        assert_array_equal(layers["geometry_type"].isna(), [False, True])

    @pytest.mark.unittest
    def test_to_parquet(self, borehole_collection, tmp_path):
        outfile = tmp_path / r"test_export.gpkg"
        borehole_collection.to_parquet(outfile)
        assert outfile.is_file()

    @pytest.mark.unittest
    def test_to_csv(self, borehole_collection, tmp_path):
        outfile = tmp_path / r"test_export.gpkg"
        borehole_collection.to_csv(outfile)
        assert outfile.is_file()

    @pytest.mark.unittest
    def test_to_pickle(self, borehole_collection, tmp_path):
        outfile = tmp_path / r"test_export.pkl"
        borehole_collection.to_pickle(outfile)
        assert outfile.is_file()

    @pytest.mark.unittest
    def test_to_pyvista_cylinders(self, borehole_collection):
        # More detailed tests are in TestLayeredData in test_data_objects.py
        cylinders = borehole_collection.to_pyvista_cylinders("lith", radius=0.1)
        assert isinstance(cylinders, pv.MultiBlock)

    @pytest.mark.unittest
    def test_to_pyvista_grid(self, borehole_collection):
        # More detailed tests are in TestLayeredData in test_data_objects.py
        grid = borehole_collection.to_pyvista_grid("lith")
        assert isinstance(grid, pv.UnstructuredGrid)

    @pytest.mark.parametrize(
        "collection, column, value, expected_column, expected_thickness",
        [
            (
                "borehole_collection",
                "lith",
                "Z",
                "Z_thickness",
                [2.2, np.nan, 2.6, 0.5, 3.0],
            ),
            (
                "borehole_collection",
                "lith",
                ["L", "Z"],
                "L,Z_thickness",
                [2.2, np.nan, 2.6, 0.5, 3.0],
            ),
            (
                "cpt_collection",
                "qc",
                slice(0.7, 13),
                "qc[0.7:13]_thickness",
                [2, 1],
            ),
        ],
        ids=["string", "list", "slice"],
    )
    def test_get_cumulative_thickness(
        self, collection, column, value, expected_column, expected_thickness, request
    ):
        collection = request.getfixturevalue(collection)

        thickness = collection.get_cumulative_thickness(column, value)
        assert isinstance(thickness, pd.Series)

        collection.get_cumulative_thickness(column, value, include_in_header=True)
        assert_array_almost_equal(
            collection.header[expected_column], expected_thickness
        )

    @pytest.mark.parametrize(
        "collection, column, value, expected_column, expected_tops",
        [
            (
                "borehole_collection",
                "lith",
                "V",
                "V_top",
                [np.nan, 1.2, np.nan, 0.5, np.nan],
            ),
            (
                "borehole_collection",
                "lith",
                ["Z", "V"],
                "Z,V_top",
                [1.5, 1.2, 2.9, 0.5, 0.0],
            ),
            (
                "cpt_collection",
                "qc",
                slice(0.7, 18),
                "qc[0.7:18]_top",
                [8.0, 0.0],
            ),
        ],
        ids=["string", "list", "slice"],
    )
    def test_get_layer_top(
        self, collection, column, value, expected_column, expected_tops, request
    ):
        collection = request.getfixturevalue(collection)

        tops = collection.get_layer_top(column, value)
        assert isinstance(tops, pd.Series)

        collection.get_layer_top(column, value, include_in_header=True)
        assert_array_almost_equal(collection.header[expected_column], expected_tops)

    @pytest.mark.parametrize(
        "collection, column, value, expected_column, expected_base",
        [
            (
                "borehole_collection",
                "lith",
                "V",
                "V_base",
                [np.nan, 3.1, np.nan, 2.5, np.nan],
            ),
            (
                "borehole_collection",
                "lith",
                ["Z", "V"],
                "Z,V_base",
                [3.7, 3.1, 5.5, 3.0, 3.0],
            ),
            (
                "cpt_collection",
                "qc",
                slice(0.7, 18),
                "qc[0.7:18]_base",
                [10, 5],
            ),
        ],
        ids=["string", "list", "slice"],
    )
    def test_get_layer_base(
        self, collection, column, value, expected_column, expected_base, request
    ):
        collection = request.getfixturevalue(collection)

        base = collection.get_layer_base(column, value)
        assert isinstance(base, pd.Series)

        collection.get_layer_base(column, value, include_in_header=True)
        assert_array_almost_equal(collection.header[expected_column], expected_base)

    @pytest.mark.unittest
    def test_to_kingdom(self, borehole_collection):
        outfile = Path("temp_kingdom.csv")
        tdfile = Path(outfile.parent, f"{outfile.stem}_TDCHART{outfile.suffix}")
        borehole_collection.to_kingdom(outfile)
        assert outfile.is_file()
        assert tdfile.is_file()
        outfile.unlink()
        tdfile.unlink()
