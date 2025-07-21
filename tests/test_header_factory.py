from unittest.mock import Mock

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point

from geost.base import LineHeader, PointHeader
from geost.header_factory import HeaderFactory, header_factory


class TestHeaderFactory:
    """Test suite for HeaderFactory class."""

    @pytest.fixture
    def factory(self):
        """Create a fresh HeaderFactory instance for each test."""
        return HeaderFactory()

    @pytest.fixture
    def point_gdf(self):
        """Create a simple point GeoDataFrame for testing."""
        data = {
            "nr": ["test1", "test2"],
            "x": [1.0, 2.0],
            "y": [1.0, 2.0],
            "surface": [0.0, 0.0],
            "end": [-10.0, -10.0],
        }
        geometry = [Point(1.0, 1.0), Point(2.0, 2.0)]
        gdf = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:28992")
        return gdf

    @pytest.fixture
    def line_gdf(self):
        """Create a simple line GeoDataFrame for testing."""
        data = {
            "nr": ["line1", "line2"],
            "x": [1.0, 2.0],
            "y": [1.0, 2.0],
        }
        geometry = [LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])]
        gdf = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:28992")
        return gdf

    @pytest.fixture
    def mixed_geometry_gdf(self):
        """Create a GeoDataFrame with mixed geometry types."""
        data = {
            "nr": ["test1", "test2"],
            "x": [1.0, 2.0],
            "y": [1.0, 2.0],
        }
        geometry = [Point(1.0, 1.0), LineString([(0, 0), (1, 1)])]
        gdf = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:28992")
        return gdf

    @pytest.mark.unittest
    def test_init(self, factory):
        """Test HeaderFactory initialization."""
        assert isinstance(factory._types, dict)
        assert len(factory._types) == 0

    @pytest.mark.unittest
    def test_register_header(self, factory):
        """Test registering header types."""
        # Test registering Point header
        factory.register_header("Point", PointHeader)
        assert "Point" in factory._types
        assert factory._types["Point"] == PointHeader

        # Test registering Line header
        factory.register_header("Line", LineHeader)
        assert "Line" in factory._types
        assert factory._types["Line"] == LineHeader

        # Test overwriting existing registration
        mock_header = Mock()
        factory.register_header("Point", mock_header)
        assert factory._types["Point"] == mock_header

    @pytest.mark.unittest
    def test_get_geometry_type_single_point(self, point_gdf):
        """Test getting geometry type for Point geometries."""
        geometry_type = HeaderFactory.get_geometry_type(point_gdf)
        assert geometry_type == "Point"

    @pytest.mark.unittest
    def test_get_geometry_type_single_line(self, line_gdf):
        """Test getting geometry type for LineString geometries."""
        geometry_type = HeaderFactory.get_geometry_type(line_gdf)
        assert geometry_type == "LineString"

    @pytest.mark.unittest
    def test_get_geometry_type_mixed(self, mixed_geometry_gdf):
        """Test getting geometry type for mixed geometries."""
        geometry_type = HeaderFactory.get_geometry_type(mixed_geometry_gdf)
        assert geometry_type == "Multiple geometries"

    @pytest.mark.unittest
    def test_get_geometry_type_empty(self):
        """Test getting geometry type for empty GeoDataFrame."""
        empty_gdf = gpd.GeoDataFrame(
            {"nr": [], "x": [], "y": []}, geometry=[], crs="EPSG:28992"
        )
        geometry_type = HeaderFactory.get_geometry_type(empty_gdf)
        assert geometry_type is None

    @pytest.mark.unittest
    def test_create_from_point_success(self, factory, point_gdf):
        """Test successful creation of PointHeader."""
        factory.register_header("Point", PointHeader)

        header = factory.create_from(point_gdf, vertical_reference="NAP")

        assert isinstance(header, PointHeader)
        assert len(header) == 2

    @pytest.mark.unittest
    def test_create_from_line_success(self, factory, line_gdf):
        """Test successful creation of LineHeader."""
        factory.register_header("LineString", LineHeader)

        header = factory.create_from(line_gdf, vertical_reference="NAP")

        assert isinstance(header, LineHeader)
        assert len(header) == 2

    @pytest.mark.unittest
    def test_create_from_unregistered_type(self, factory, point_gdf):
        """Test creation with unregistered geometry type raises ValueError."""
        # Don't register Point header
        with pytest.raises(ValueError, match="No Header type available for Point"):
            factory.create_from(point_gdf, vertical_reference="NAP")

    @pytest.mark.unittest
    def test_create_from_mixed_geometries(self, factory, mixed_geometry_gdf):
        """Test creation with mixed geometries raises ValueError."""
        factory.register_header("Point", PointHeader)
        factory.register_header("LineString", LineHeader)

        with pytest.raises(
            ValueError, match="No Header type available for Multiple geometries"
        ):
            factory.create_from(mixed_geometry_gdf, vertical_reference="NAP")

    @pytest.mark.unittest
    def test_create_from_with_kwargs(self, factory, point_gdf):
        """Test that kwargs are passed through to header constructor."""
        factory.register_header("Point", PointHeader)

        # Test with vertical_reference kwarg
        header = factory.create_from(point_gdf, vertical_reference="EPSG:5709")
        assert header.vertical_reference.to_epsg() == 5709

    @pytest.mark.unittest
    def test_create_from_empty_gdf(self, factory):
        """Test creation with empty GeoDataFrame."""
        factory.register_header("Point", PointHeader)

        empty_gdf = gpd.GeoDataFrame(
            {"nr": [], "x": [], "y": [], "surface": [], "end": []},
            geometry=[],
            crs="EPSG:28992",
        )

        # Empty GeoDataFrame should raise ValueError due to None geometry type
        with pytest.raises(ValueError, match="No Header type available for None"):
            factory.create_from(empty_gdf, vertical_reference="NAP")
