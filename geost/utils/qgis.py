from __future__ import annotations

from numbers import Real
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd


def write_qml(
    path: str | Path,
    df: pd.DataFrame,
    *,
    value_column: str,
    color_columns: tuple[str, str, str],
    label_column: str | None = None,
    description_column: str | None = None,
    label_separator: str = " - ",
    layer_type: str = "vector",
    field: str | None = None,
    band: int = 1,
) -> None:
    """
    Write QGIS vector or raster symbology.

    Parameters
    ----------
    path : str or pathlib.Path
        Path and filename where the generated QGIS style file will be written.
    df : pandas.DataFrame
        Table containing the category values and RGB color values.
    value_column : str
        Column or index containing category values. These values choose colors
        and are shown as labels unless ``label_column`` is supplied.
    color_columns : tuple of str
        Names of the red, green, and blue columns, in that order. Their values
        set the color for each corresponding row in the table.
    label_column : str, optional
        Column or index containing the text shown in the legend. Defaults to
        ``value_column``.
    description_column : str, optional
        Optional column containing text to add after each legend label.
    label_separator : str, optional
        Text placed between the legend label and its description. Defaults to
        ``" - "``.
    layer_type : str, optional
        Choose ``"vector"`` (default) to color features by an attribute, or
        ``"raster"`` to color cells whose numeric values match ``value_column``.
    field : str, optional
        Name of the target vector field containing values from
        ``value_column``. QGIS uses it to choose each feature's color. If
        omitted, the name in ``value_column`` is used. Not used for rasters.
    band : int, optional
        Raster band containing values to match with the palette colors.
        Defaults to 1 and is only used for raster styles.

    Raises
    ------
    ValueError
        If an option is invalid, required columns are missing, the table is
        empty, or any RGB value is outside the range from 0 to 255.

    """
    if layer_type not in ("vector", "raster"):
        raise ValueError("layer_type must be 'vector' or 'raster'.")
    if layer_type == "vector":
        if not isinstance(field, str) or not field.strip():
            raise ValueError("field must be a non-empty attribute field name.")
    else:
        if field is not None:
            raise ValueError("field is only applicable to vector styles.")
        if not isinstance(band, int) or isinstance(band, bool) or band < 1:
            raise ValueError("band must be a positive integer.")

    if df.empty:
        raise ValueError("Cannot export empty metadata.")

    values = _get_metadata_values(df, value_column)
    rgb = df.loc[:, list(color_columns)].to_numpy(dtype=float)
    if rgb.shape != (len(values), 3):
        raise ValueError(
            "RGB values must contain one red, green, and blue value per category."
        )
    if (
        not np.isfinite(rgb).all()
        or ((rgb < 0) | (rgb > 255)).any()
        or (rgb != np.floor(rgb)).any()
    ):
        raise ValueError("RGB values must be integers between 0 and 255.")
    rgb = rgb.astype(int)

    label_values = (
        values if label_column is None else _get_metadata_values(df, label_column)
    )
    if description_column is None:
        labels = [str(value) for value in label_values]
    else:
        descriptions = _get_metadata_values(df, description_column)
        labels = [
            (
                str(value)
                if pd.isna(description)
                else f"{value}{label_separator}{description}"
            )
            for value, description in zip(label_values, descriptions)
        ]

    if layer_type == "raster" and any(
        not isinstance(value, Real) or isinstance(value, bool) for value in values
    ):
        raise ValueError("Raster styles only support numeric voxel numbers.")

    qgis = ET.Element(
        "qgis",
        version="3.34.0",
        styleCategories="Symbology",
    )
    if layer_type == "raster":
        _add_raster_style(qgis, values, rgb, labels, band)
    else:
        if field is None:
            field = value_column
        if not isinstance(field, str) or not field.strip():
            raise ValueError("field must be a non-empty attribute field name.")
        _add_vector_style(qgis, values, rgb, labels, field)

    ET.indent(qgis, space="  ")
    ET.ElementTree(qgis).write(
        str(Path(path)),
        encoding="utf-8",
        xml_declaration=True,
    )


def _get_metadata_values(df: pd.DataFrame, column: str) -> list[Any]:
    """
    Get metadata values from a DataFrame column or its named index.

    Parameters
    ----------
    df : pandas.DataFrame
        Table to read the requested column or index values from.
    column : str
        Name of the metadata column or index.

    Returns
    -------
    list
        Values from the requested column or index, in their original order.

    Raises
    ------
    ValueError
        If the requested column is not present and is not the named index.

    """
    if column in df.columns:
        return df[column].tolist()
    if df.index.name == column:
        return df.index.tolist()
    raise ValueError(f"Metadata column '{column}' was not found.")


def _add_vector_style(
    qgis: ET.Element,
    values: list[Any],
    colors_rgb: np.ndarray,
    labels: list[str],
    field: str,
) -> None:
    """
    Add categorized polygon symbols to a QGIS style document.

    Parameters
    ----------
    qgis : xml.etree.ElementTree.Element
        Root QGIS style element to which the vector renderer is added.
    values : list
        Values used to create vector categories, in metadata row order.
    colors_rgb : numpy.ndarray
        RGB colors corresponding to each value, in the same row order.
    labels : list of str
        Text labels displayed for each category in the QGIS layer legend.
    field : str
        Name of the vector attribute whose values select these categories.

    """
    renderer = ET.SubElement(
        qgis,
        "renderer-v2",
        type="categorizedSymbol",
        attr=field,
        symbollevels="0",
        enableorderby="0",
    )
    categories = ET.SubElement(renderer, "categories")
    symbols = ET.SubElement(renderer, "symbols")

    for index, (value, label, color) in enumerate(zip(values, labels, colors_rgb)):
        red, green, blue = color
        if isinstance(value, str):
            value_type = "string"
        elif isinstance(value, Real):
            value_type = "double" if isinstance(value, (float, np.floating)) else "long"
        else:
            raise ValueError("Vector category values must be strings or numbers.")

        ET.SubElement(
            categories,
            "category",
            value=str(value),
            type=value_type,
            symbol=str(index),
            render="true",
            label=label,
        )
        symbol = ET.SubElement(
            symbols, "symbol", name=str(index), type="fill", alpha="1"
        )
        layer = ET.SubElement(
            symbol,
            "layer",
            {"class": "SimpleFill", "enabled": "1", "locked": "0"},
        )
        options = ET.SubElement(layer, "Option", type="Map")
        for name, option_value in (
            ("color", f"{red},{green},{blue},255"),
            ("style", "solid"),
            ("outline_style", "no"),
        ):
            ET.SubElement(
                options,
                "Option",
                name=name,
                value=option_value,
                type="QString",
            )


def _add_raster_style(
    qgis: ET.Element,
    values: list[Any],
    colors_rgb: np.ndarray,
    labels: list[str],
    band: int,
) -> None:
    """
    Add a paletted raster renderer to a QGIS style document.

    Parameters
    ----------
    qgis : xml.etree.ElementTree.Element
        Root QGIS style element to which the raster renderer is added.
    values : list
        Numeric raster values to associate with palette colors, in row order.
    colors_rgb : numpy.ndarray
        RGB colors corresponding to each raster value, in the same row order.
    labels : list of str
        Text labels displayed for the values in the QGIS raster legend.
    band : int
        Number of the raster band containing values to match with the palette.

    """
    pipe = ET.SubElement(qgis, "pipe")
    renderer = ET.SubElement(
        pipe,
        "rasterrenderer",
        type="paletted",
        band=str(band),
        opacity="1",
        alphaBand="-1",
        nodataColor="",
    )
    ET.SubElement(renderer, "rasterTransparency")
    palette = ET.SubElement(renderer, "colorPalette")

    for value, label, color in zip(values, labels, colors_rgb):
        red, green, blue = color
        ET.SubElement(
            palette,
            "paletteEntry",
            value=str(value),
            color=f"#{red:02x}{green:02x}{blue:02x}",
            alpha="255",
            label=label,
        )
