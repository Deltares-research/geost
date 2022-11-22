import requests
import xml.etree.ElementTree as ET


class BroApi:
    def __init__(self, server_url=r"https://publiek.broservices.nl"):
        self.session = requests.Session()
        self.server_url = server_url
        self.cpt_api = "/sr/cpt/v1"
        self.objects_url = "/objects"
        self.search_url = "/characteristics/searches"

    def get_CPT_objects(self, bro_ids):
        if isinstance(bro_ids, str):
            bro_ids = [bro_ids]
        for bro_id in bro_ids:
            response = self.session.get(
                self.server_url + self.cpt_api + self.objects_url + f"/{bro_ids}"
            )
            yield ET.fromstring(response.text)


# testing
# bro_api = BroApi()
# cpt = bro_api.get_CPT_objects("CPT000000038771")
# for cp in cpt:
#     cp
