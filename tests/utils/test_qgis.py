from xml.etree import ElementTree as ET

import pandas as pd
import pytest

from geost.utils.qgis import write_qml


@pytest.fixture
def lithology_dataframe():
    return pd.DataFrame(
        {
            "lithology_nr": [1, 2, 3],
            "lithology": ["sand", "clay", "silt"],
            "description": ["granular", "cohesive", "fine-grained"],
            "red": [230, 120, 190],
            "green": [210, 150, 170],
            "blue": [120, 170, 120],
        }
    )


@pytest.mark.unittest
@pytest.mark.parametrize(
    "value_column, label_column, description_column, expected_labels",
    [
        ("lithology_nr", None, None, ["1", "2", "3"]),
        ("lithology", None, None, ["sand", "clay", "silt"]),
        (
            "lithology_nr",
            None,
            "lithology",
            ["1 - sand", "2 - clay", "3 - silt"],
        ),
        (
            "lithology",
            None,
            "description",
            ["sand - granular", "clay - cohesive", "silt - fine-grained"],
        ),
    ],
)
def test_write_qml_vector_from_dataframe(
    tmp_path,
    lithology_dataframe,
    value_column,
    label_column,
    description_column,
    expected_labels,
):
    path = tmp_path / "vector.qml"

    assert (
        write_qml(
            path,
            lithology_dataframe,
            value_column=value_column,
            color_columns=("red", "green", "blue"),
            label_column=label_column,
            description_column=description_column,
            field="lithology_code",
        )
        is None
    )

    root = ET.parse(str(path)).getroot()
    renderer = root.find("renderer-v2")
    assert renderer is not None
    assert renderer.get("type") == "categorizedSymbol"
    assert renderer.get("attr") == "lithology_code"
    categories = renderer.findall("categories/category")
    expected_values = lithology_dataframe[value_column].astype(str).tolist()
    assert [category.get("value") for category in categories] == expected_values
    assert [category.get("label") for category in categories] == expected_labels

    first_symbol = renderer.find("symbols/symbol")
    assert first_symbol is not None
    options = {
        option.get("name"): option.get("value")
        for option in first_symbol.findall("layer/Option/Option")
    }
    assert options["color"] == "230,210,120,255"


@pytest.mark.unittest
def test_write_qml_raster_from_dataframe(tmp_path, lithology_dataframe):
    path = tmp_path / "raster.qml"

    assert (
        write_qml(
            path,
            lithology_dataframe,
            value_column="lithology_nr",
            color_columns=("red", "green", "blue"),
            layer_type="raster",
            band=2,
        )
        is None
    )

    root = ET.parse(str(path)).getroot()
    renderer = root.find("pipe/rasterrenderer")
    assert renderer is not None
    assert renderer.get("type") == "paletted"
    assert renderer.get("band") == "2"
    entries = renderer.findall("colorPalette/paletteEntry")
    assert [entry.get("value") for entry in entries] == ["1", "2", "3"]
    assert [entry.get("color") for entry in entries] == [
        "#e6d278",
        "#7896aa",
        "#beaa78",
    ]
    assert [entry.get("label") for entry in entries] == ["1", "2", "3"]


@pytest.mark.unittest
def test_write_qml_vector_duplicate_labels_keep_distinct_colors(
    tmp_path, lithology_dataframe
):
    lithology_dataframe.loc[1, ["lithology", "description"]] = ["sand", "granular"]
    path = tmp_path / "duplicate_labels.qml"

    write_qml(
        path,
        lithology_dataframe,
        value_column="lithology_nr",
        color_columns=("red", "green", "blue"),
        label_column="lithology",
        description_column="description",
        field="lithology_code",
    )

    root = ET.parse(str(path)).getroot()
    renderer = root.find("renderer-v2")
    assert renderer is not None
    categories = renderer.findall("categories/category")
    assert categories[0].get("label") == categories[1].get("label")
    assert categories[0].get("value") != categories[1].get("value")
    symbols = renderer.findall("symbols/symbol")
    colors = []
    for symbol in symbols[:2]:
        options = {
            option.get("name"): option.get("value")
            for option in symbol.findall("layer/Option/Option")
        }
        colors.append(options["color"])
    assert colors == ["230,210,120,255", "120,150,170,255"]
    assert colors[0] != colors[1]


@pytest.mark.unittest
def test_write_qml_rejects_invalid_rgb(tmp_path):
    lithology_dataframe = pd.DataFrame(
        {
            "lithology_nr": [1],
            "lithology": ["sand"],
            "description": ["granular"],
            "red": [256],
            "green": [210],
            "blue": [120],
        }
    )

    with pytest.raises(
        ValueError, match="RGB values must be integers between 0 and 255"
    ):
        write_qml(
            tmp_path / "invalid.qml",
            lithology_dataframe,
            value_column="lithology_nr",
            color_columns=("red", "green", "blue"),
            field="lithology_code",
        )
