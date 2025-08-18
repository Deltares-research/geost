from typing import Iterable, Iterator, List, TypeVar, Union

import requests
from lxml import etree

from geost.bro.bro_utils import get_bbox_criteria
from geost.projections import horizontal_reference_transformer

Coordinate = TypeVar("Coordinate", int, float)


class BroApi:
    apis = {
        "CPT": "/sr/cpt/v1",
        "BHR-P": "/sr/bhrp/v2",
        "BHR-GT": "/sr/bhrgt/v2",
        "BHR-G": "/sr/bhrg/v3",
        "SFR": "/sr/sfr/v2",
    }

    def __init__(self, server_url=r"https://publiek.broservices.nl"):
        self.session = requests.Session()
        self.server_url = server_url
        self.search_url = "/characteristics/searches"
        self.object_list = []

    def get_objects(
        self, bro_ids: str | Iterable, object_type: str = "CPT"
    ) -> list[bytes]:
        """
        Retrieve BRO objects via API requests to the public BRO REST-API.

        Parameters
        ----------
        bro_ids : str | Iterable
            Single BRO ID or Iterable of BRO ID's
        object_type : str, optional
            BRO object type. Can be CPT (Cone penetration tests), BHR-P (Soil cores),
            BHR-GT (Geotechnical cores), or BHR-G (Geological cores). By default "CPT".

        Returns
        -------
        list[bytes]
            List of bytestrings containing the XML data of the requested BRO objects.

        Raises
        ------
        Warning
            When a non-existing BRO ID was given.
        Warning
            When the server does not respond to the request (40x error).
        """
        if isinstance(bro_ids, str):
            bro_ids = [bro_ids]

        api_response = []
        for bro_id in bro_ids:
            response = self.session.get(
                f"{self.server_url}{self.apis[object_type]}/objects/{bro_id}"
            )
            if response.status_code == 200 and "rejection" not in response.text:
                api_response.append(response.text.encode("utf-8"))
            elif response.status_code != 200 or "rejection" in response.text:
                raise Warning(
                    f"Error {response.status_code}: Unable to request {bro_id} from ",
                    "database",
                )
        return api_response

    def search_objects_in_bbox(
        self,
        xmin: Coordinate,
        ymin: Coordinate,
        xmax: Coordinate,
        ymax: Coordinate,
        epsg: str = "28992",
        object_type: str = "CPT",
    ) -> List[str]:
        """
        Search for BRO objects of the given object type within the given bounding box.
        Returns a list of BRO objects that can be used to retrieve their data using the
        get_objects method.

        Parameters
        ----------
        xmin : Union[float, int]
            x-coordinate of the bbox lower left corner.
        ymin : Union[float, int]
            y-coordinate of the bbox lower left corner.
        xmax : Union[float, int]
            x-coordinate of the bbox upper right corner.
        ymax : Union[float, int]
            y-coordinate of the bbox upper right corner.
        epsg : str, optional
            Coordinate reference system of the given bbox coordinates, by default
            "28992" (= Rijksdriehoek New CRS).
        object_type : str, optional
            BRO object type. Can be CPT (Cone penetration tests), BHR-P (Soil cores),
            BHR-GT (Geotechnical cores), or BHR-G (Geological cores). By default "CPT".

        Returns
        -------
        List[str]
            List containing BRO ID's of objects found within the bbox.

        Raises
        ------
        Warning
            If the server does not respond or the search query was rejected.
        """
        response = self.__response_to_bbox(
            xmin, ymin, xmax, ymax, epsg=epsg, object_type=object_type
        )
        if response.status_code == 200 and "rejection" not in response.text:
            etree_root = etree.fromstring(response.text.encode("utf-8"))
            self.object_list += self.__objects_from_etree(etree_root)
        elif response.status_code == 400 or "groter dan 2000" in response.text:
            self.__search_objects_in_divided_bbox(
                xmin, ymin, xmax, ymax, epsg=epsg, object_type=object_type
            )
        else:
            raise Warning(
                "Selection is invalid and could not be retrieved from the database"
            )

    def __search_objects_in_divided_bbox(
        self,
        xmin: Coordinate,
        ymin: Coordinate,
        xmax: Coordinate,
        ymax: Coordinate,
        epsg: str = "28992",
        object_type: str = "CPT",
    ) -> List[str]:
        division_levels = int((xmax - xmin + ymax - ymin) / 1000)
        division_x = (xmax - xmin) / division_levels
        for division_level in range(division_levels):
            print(
                f"More than 2000 object requests in API call, dividing calls. Current call {division_level + 1}/{division_levels}"
            )
            xmin_divided = xmin + (division_level * division_x)
            xmax_divided = xmin + ((division_level + 1) * division_x)
            response = self.__response_to_bbox(
                xmin_divided,
                ymin,
                xmax_divided,
                ymax,
                epsg=epsg,
                object_type=object_type,
            )
            etree_root = etree.fromstring(response.text.encode("utf-8"))
            self.object_list += self.__objects_from_etree(etree_root)

    def __response_to_bbox(
        self,
        xmin: Coordinate,
        ymin: Coordinate,
        xmax: Coordinate,
        ymax: Coordinate,
        epsg: str = "28992",
        object_type: str = "CPT",
    ):
        transformer = horizontal_reference_transformer(epsg, 4326)
        xmin, ymin = transformer.transform(xmin, ymin)
        xmax, ymax = transformer.transform(xmax, ymax)
        criteria = get_bbox_criteria(xmin, ymin, xmax, ymax)
        api_url = self.server_url + self.apis[object_type] + self.search_url
        response = self.session.post(api_url, json=criteria)
        return response

    def __objects_from_etree(self, etree_root):
        namespace = etree_root.nsmap["brocom"]
        bro_ids = etree_root.findall(".//{" + f"{namespace}" + "}broId")
        bro_objects = [id.text for id in bro_ids]
        return bro_objects
