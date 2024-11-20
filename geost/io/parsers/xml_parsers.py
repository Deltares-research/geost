import re
from pathlib import Path
from typing import NamedTuple, Union

import numpy as np
import pandas as pd
from lxml import etree

from geost.io.parsers.parser_utils import DDCoord, RDCoord, safe_coerce


class LayerSoilCore(NamedTuple):
    top: Union[int, float]
    bottom: Union[int, float]
    horizon: str
    standard_name: str
    pedologic_name: str
    # TODO: not all the below are always in the xml
    # om: str
    # ca: str
    # gravel: str
    # shells: str
    # clay_perc: Union[int, float]


class SoilCore:
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

        self.broid = None
        self.x = None
        self.y = None
        self.z = None
        self.enddepth = None
        self.quality = None
        self.crs = None
        self.reference_level = None

        self.__set_namespaces()
        self.__get_data()
        self._set_header_info()

        self.df = pd.DataFrame(self.parse_layers())
        attr_dict = {
            "nr": "broid",
            "x": "x",
            "y": "y",
            "surface": "z",
            "end": "enddepth",
            "codegroup": "codegroup",
            "soilclass": "soilclass",
        }
        for attr in ["nr", "x", "y", "surface", "end", "codegroup", "soilclass"][::-1]:
            self.df.insert(0, attr, self.__dict__[attr_dict[attr]])
        self.df = self.df.convert_dtypes()

    def __repr__(self):
        name = self.__class__.__name__
        repr_ = f"{name}(nr={self.broid}, end={self.enddepth}, quality={self.quality})"
        return repr_

    @property
    def ns(self):
        ns = {
            None: "http://www.broservices.nl/xsd/brocommon/3.0",
            "ns6": "http://www.opengis.net/swe/2.0",
            "ns5": "http://www.broservices.nl/xsd/srcommon/1.0",
            "ns8": "http://www.broservices.nl/xsd/bhrcommon/2.0",
            "ns7": "http://www.opengis.net/om/2.0",
            "ns2": "http://www.opengis.net/gml/3.2",
            "ns4": "http://www.broservices.nl/xsd/dsbhr/2.0",
            "ns3": "http://www.w3.org/1999/xlink",
        }
        return ns

    def __set_namespaces(self):
        for k, v in self.ns.items():
            v = "{" + v + "}"
            if k is None:
                setattr(self, "defns", v)
            else:
                setattr(self, k, v)

    @property
    def attr_paths(self):
        """
        Return a dictionary of attributes in the xml and corresponding datapaths.

        """
        attr_paths = {e.tag.split("}")[1]: e.tag for e in self.data}
        return attr_paths

    @property
    def header(self):
        index = [
            "nr",
            "x",
            "y",
            "surface",
            "end",
            "quality",
            "crs",
            "reference_level",
            "codegroup",
            "soilclass",
            "textclass",
            "textprofile",
            "ca_profile",
            "reworking",
            "gw_class",
        ]

        header = pd.Series(
            [
                self.broid,
                self.x,
                self.y,
                self.z,
                self.enddepth,
                self.quality,
                self.crs,
                self.reference_level,
                self.codegroup,
                self.soilclass,
                self.textclass,
                self.textprofile,
                self.ca_profile,
                self.reworking,
                self.gw_class,
            ],
            index=index,
        )
        return header

    @property
    def attrs(self):
        return list(self.attr_paths.keys())

    def get_main_element(self, attr):
        e = self.data.find(f"{self.attr_paths[attr]}")
        return e

    def __get_data(self):
        self.data = self.root.find(f"*/{self.ns4}BHR_O")

    def _set_header_info(self):
        self.broid = self.get_main_element("broId").text
        self.quality = self.get_main_element("qualityRegime").text

        self.rd_location()
        self.get_z()
        self.bored_interval()
        self.parse_soil_classification_data()

    @staticmethod
    def get_crs(loc):
        crs = re.search(r"EPSG::(?P<crs>\d+)", str(loc.values())).group("crs")
        return crs

    @property
    def _borehole_data(self):
        data = self.get_main_element("boreholeSampleDescription")
        return data

    def rd_location(self):
        """
        Return a namedtuple of the location in RD-coordinates.

        """
        rd_element = self.get_main_element("deliveredLocation")
        loc = rd_element.find(f"{self.ns8}location")
        self.crs = self.get_crs(loc)

        x, y = loc.find(f"{self.ns2}pos").text.split(" ")
        self.x = round(float(x), 0)
        self.y = round(float(y), 0)

        return RDCoord(self.x, self.y, int(self.crs))

    def latlon_location(self):
        """
        Return a namedtuple of the location in decimal degree coordinates.

        """
        latlon_element = self.get_main_element("standardizedLocation")
        loc = latlon_element.find(f"{self.defns}location")

        crs = self.get_crs(loc)

        lat, lon = loc.find(f"{self.ns2}pos").text.split(" ")

        return DDCoord(float(lat), float(lon), int(crs))

    def get_z(self):
        elem = self.get_main_element("deliveredVerticalPosition")

        z = elem.find(f"{self.ns8}offset").text
        ref = elem.find(f"{self.ns8}verticalDatum").text

        self.z = float(z)
        self.reference_level = ref

    def bored_interval(self):
        elem = self.get_main_element("boring")

        self.begindepth = float(elem.find(f"*/{self.ns8}beginDepth").text)
        self.enddepth = self.z - float(elem.find(f"*/{self.ns8}endDepth").text)

    def parse_soillayer_element(self, layer):
        """
        Parse a soillayer element from the xml element tree of a Pedological
        soil core.

        Parameters
        ----------
        layer : lxml.etree._Element
            Soillayer element to parse.

        Returns
        -------
        l : namedtuple
            Namedtuple of the information and values of the layer.

        """
        return LayerSoilCore(
            float(layer.find(f"{self.ns8}upperBoundary").text),
            float(layer.find(f"{self.ns8}lowerBoundary").text),
            layer.find(f"*/{self.ns8}horizonCode").text,
            layer.find(f"*/*/{self.ns8}standardSoilName").text,
            layer.find(f"*/*/{self.ns8}pedologicalSoilName").text,
            # TODO:Not all of the below are in the xml
            # layer.find(f"*/*/{self.ns8}organicMatterClass").text,
            # layer.find(f"*/*/{self.ns8}carbonateClass").text,
            # layer.find(f"*/*/{self.ns8}containsGravel").text,
            # layer.find(f"*/*/{self.ns8}containsShellMatter").text,
            # float(layer.find(f"*/*/*/{self.ns8}clayContent").text),
        )

    def parse_layers(self):
        layers = self._borehole_data.find(f"{self.ns8}result")
        for l in layers.iterfind(f"{self.ns8}soilLayer"):  # noqa: E741
            yield self.parse_soillayer_element(l)

    def parse_soil_classification_data(self):
        soilclass = self._borehole_data.find(f"{self.ns8}soilClassification")

        self.codegroup = soilclass.find(f"{self.ns8}codeGroup").text
        self.soilclass = soilclass.find(f"{self.ns8}soilClass").text
        # TODO: Not all of the below parameters are always in the xml
        # self.textclass = soilclass.find(f"{self.ns8}textureClass").text
        # self.textprofile = soilclass.find(f"{self.ns8}textureProfile").text
        # self.ca_profile = soilclass.find(f"{self.ns8}carbonateProfile").text
        self.textclass = None
        self.textprofile = None
        self.ca_profile = None
        self.reworking = soilclass.find(f"{self.ns8}reworkingClass").text
        self.gw_class = soilclass.find(f"{self.ns8}groundwaterTableClass").text


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
