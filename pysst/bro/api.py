import requests
from lxml import etree
from typing import Union, Iterable
from pysst.projections import xy_to_ll
from pysst.bro.bro_utils import get_bbox_criteria


class BroApi:
    def __init__(self, server_url=r"https://publiek.broservices.nl"):
        self.session = requests.Session()
        self.server_url = server_url
        self.apis = {
            "CPT": "/sr/cpt/v1",
            "BHR-P": "/sr/bhrp/v2",
            "BHR-GT": "/sr/bhrgt/v2",
            "BHR-G": "/sr/bhrg/v2",
        }
        self.objects_url = "/objects"
        self.search_url = "/characteristics/searches"

    def get_objects(self, bro_ids: Union[str, Iterable], object_type: str = "CPT"):
        """
        Get BRO objects

        Parameters
        ----------
        bro_ids : Union[str, Iterable]
            Single BRO ID or Iterable of BRO ID's
        object_type : str, optional
            BRO object type. Can be CPT (Cone penetration tests), BHR-P (Soil cores), BHR-GT (Geotechnical cores), or BHR-G (Geological cores). By default "CPT"

        Yields
        ------
        lxml._Element
            Element tree containing data of the requested BRO object

        Raises
        ------
        Warning
            When a non-existing BRO ID was given
        Warning
            When the server does noet respond (40x error)
        """
        if isinstance(bro_ids, str):
            bro_ids = [bro_ids]
        for bro_id in bro_ids:
            response = self.session.get(
                self.server_url
                + self.apis[object_type]
                + self.objects_url
                + f"/{bro_id}"
            )
            if response.status_code == 200 and not "rejection" in response.text:
                yield etree.fromstring(response.text.encode("utf-8"))
            elif "rejection" in response.text:
                raise Warning(
                    f"{bro_id} is invalid and could not be retrieved from the database"
                )
            elif response.status_code != 200:
                raise Warning(
                    f"Error {response.status_code}: Unable to request {bro_id} from database"
                )

    def search_objects_in_bbox(
        self,
        xmin: Union[float, int] = 140500,
        xmax: Union[float, int] = 141000,
        ymin: Union[float, int] = 455000,
        ymax: Union[float, int] = 455500,
        epsg: str = "28992",
        object_type: str = "CPT",
    ):
        xmin_ll, ymin_ll = xy_to_ll(xmin, ymin, epsg)
        xmax_ll, ymax_ll = xy_to_ll(xmax, ymax, epsg)
        criteria = get_bbox_criteria(xmin_ll, xmax_ll, ymin_ll, ymax_ll)
        api_url = self.server_url + self.apis[object_type] + self.search_url
        response = self.session.post(api_url, json=criteria)
        if response.status_code == 200 and not "rejection" in response.text:
            etree_root = etree.fromstring(response.text.encode("utf-8"))
        else:
            raise Warning(
                f"Selection is invalid and could not be retrieved from the database"
            )

        namespaces = etree_root.nsmap
        selected_bro_elements = etree_root.findall("dispatchDocument", namespaces)
        bro_ids = []
        for selected_bro_element in selected_bro_elements:
            bro_el = selected_bro_element.getchildren()[0]
            bro_ids.append(bro_el.find("brocom:broId", namespaces).text)

        return bro_ids
