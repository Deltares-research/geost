import requests
from lxml import etree
from typing import Union, Iterable


class BroApi:
    def __init__(self, server_url=r"https://publiek.broservices.nl"):
        self.session = requests.Session()
        self.server_url = server_url
        self.cpt_api = "/sr/cpt/v1"
        self.objects_url = "/objects"
        self.search_url = "/characteristics/searches"

    def get_cpt_objects(self, bro_ids: Union[str, Iterable]):
        if isinstance(bro_ids, str):
            bro_ids = [bro_ids]
        for bro_id in bro_ids:
            response = self.session.get(
                self.server_url + self.cpt_api + self.objects_url + f"/{bro_id}"
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
