import pytest
from pathlib import Path
from lxml.etree import _Element

from pysst.bro import BroApi


class TestBroApi:
    def test_response(self):
        api = BroApi()
        assert api.session.get(api.server_url + api.cpt_api).status_code == 200

    def test_get_cpts(self):
        api = BroApi()
        cpts = api.get_cpt_objects(
            ["CPT000000038771", "CPT000000000787", "CPT000000125133"]
        )
        for cpt in cpts:
            assert isinstance(cpt, _Element)
