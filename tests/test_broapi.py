import pytest
from pathlib import Path
from lxml.etree import _Element

from pysst.bro import BroApi


class TestBroApi:
    def test_response(self):
        api = BroApi()
        assert api.session.get(api.server_url + api.cpt_api).status_code == 200

    def test_get_single_valid_cpt(self):
        api = BroApi()
        cpts = api.get_cpt_objects("CPT000000038771")
        for cpt in cpts:
            assert isinstance(cpt, _Element)

    def test_get_valid_cpts(self):
        api = BroApi()
        cpts = api.get_cpt_objects(
            ["CPT000000038771", "CPT000000000787", "CPT000000125133"]
        )
        for cpt in cpts:
            assert isinstance(cpt, _Element)

    def test_get_invalid_cpt(self):
        api = BroApi()
        cpts = api.get_cpt_objects("CPT0000000doesnotexist")
        with pytest.raises(Warning) as excinfo:
            for cpt in cpts:
                pass
        assert (
            "CPT0000000doesnotexist is invalid and could not be retrieved from the database"
            in str(excinfo.value)
        )
