from pathlib import Path

import numpy as np
import pandas as pd
from lxml import etree

from geost.io.parsers.parser_utils import safe_coerce


class BorisXML:
    """
    Class with properties and reading/parsing methods for BORIS XML exports. BORIS is
    software developed by TNO to describe boreholes in the lab. The exported XML-format
    bears no resemblance to DINO or BRO XML standards.

    The reader is naive, which means that all data is taken from the XML as is, without
    running any checks on the parsed data.

    Note that currently only essential items to construct a BoreholeCollection are parsed.
    """

    attr_type_to_dtype = dict(
        code=str,
        median=float,
        percentage=float,
    )

    def __init__(self, xml: str | Path | etree._Element):
        if isinstance(xml, (str, Path)):
            self.root = etree.parse(xml).getroot()
        elif isinstance(xml, etree._Element):
            self.root = xml
        else:
            raise TypeError(
                "Input datatype for reader not allowed. Must be an lxml Element "
                "tree or path to an xml file."
            )

        self.layer_dataframes = list()

        for point_survey in self.root.iterchildren():
            self.parse_pointsurvey(point_survey)
        self.layer_dataframe = pd.concat(
            self.layer_dataframes, ignore_index=True
        ).convert_dtypes(convert_integer=False)

    def parse_pointsurvey(self, point_survey: etree._Element) -> None:
        # Get header and layer data as dictionaries
        header_data = self.parse_headerdata(point_survey)
        layer_dataframe = self.parse_layerdata(
            point_survey.findall("borehole/lithoDescr/lithoInterval")
        )
        layer_dataframe.insert(
            1,
            "top",
            [0] + list(layer_dataframe["bottom"].iloc[:-1].values),
        )
        layer_dataframe["end"] += header_data["surface"]

        # Repeat header data for every described borehole interval
        for key, value in header_data.items():
            layer_dataframe.insert(0, key, np.full(len(layer_dataframe), value))

        # Append data of this borehole to
        self.layer_dataframes.append(layer_dataframe)

    def parse_headerdata(self, point_survey: etree._Element) -> dict:
        # Find elements that contain header data
        identification_element = point_survey.find("identification")
        location_element = point_survey.find("surveyLocation")
        elevation_element = point_survey.find("surfaceElevation/elevation")

        # Get positional reference system attributes
        # self.horizontal_reference = location_element.find("coordinates").get(
        #     "coordSystem"
        # )
        # self.vertical_reference = elevation_element.get("levelReference") or "NAP"

        # Get header data. Note that the 'end' column is later retrieved and added to
        # the header based on the last layer interval.
        nr = safe_coerce(identification_element.get("id"), str)
        x = safe_coerce(location_element.find("coordinates/coordinateX").text, float)
        y = safe_coerce(location_element.find("coordinates/coordinateY").text, float)
        surface = safe_coerce(elevation_element.get("levelValue"), float)

        # Create header data dict
        header_data = dict(surface=surface, y=y, x=x, nr=nr)
        return header_data

    def parse_layerdata(self, borehole_element: etree._Element) -> pd.DataFrame:
        *_, last = borehole_element
        end = safe_coerce(last.get("baseDepth"), float) * -1

        num_of_layers = len(borehole_element)
        layer_data = list()
        unique_tags = ["bottom"]
        for interval in borehole_element:
            interval_dict = dict()
            interval_dict["bottom"] = float(interval.get("baseDepth"))
            for data_element in interval.getchildren():
                if data_element.tag not in unique_tags:
                    unique_tags += [data_element.tag]
                if data_element.attrib:
                    attr_type, attr_value = (
                        data_element.attrib.keys()[0],
                        data_element.attrib.values()[0],
                    )
                    interval_dict[data_element.tag] = safe_coerce(
                        attr_value, self.attr_type_to_dtype[attr_type]
                    )
                else:
                    interval_dict[data_element.tag] = safe_coerce(
                        data_element.text, str
                    )
            layer_data.append(interval_dict)

        # Construct final dataframe
        layer_df = pd.DataFrame(columns=unique_tags, index=range(num_of_layers))
        for i, layer in enumerate(layer_data):
            layer_df.update(pd.DataFrame(layer, index=[i]))
        layer_df.insert(0, "end", np.full(num_of_layers, end))
        return layer_df
