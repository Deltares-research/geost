import pytest
from pathlib import Path
from lxml.etree import _Element

from pysst.bro import BroApi


class TestBroApi:
    def test_response(self):
        api = BroApi()
        for key in api.apis:
            assert api.session.get(api.server_url + api.apis[key]).status_code == 200

    def test_get_single_valid_cpt(self):
        api = BroApi()
        cpts = api.get_objects("CPT000000038771", object_type="CPT")
        for cpt in cpts:
            assert isinstance(cpt, _Element)

    def test_get_valid_cpts(self):
        api = BroApi()
        cpts = api.get_objects(
            ["CPT000000038771", "CPT000000000787", "CPT000000125133"], object_type="CPT"
        )
        for cpt in cpts:
            assert isinstance(cpt, _Element)

    def test_get_invalid_cpt(self):
        api = BroApi()
        cpts = api.get_objects("CPT0000000doesnotexist", object_type="CPT")
        with pytest.raises(Warning) as excinfo:
            for cpt in cpts:
                pass
        assert (
            "CPT0000000doesnotexist is invalid and could not be retrieved from the database"
            in str(excinfo.value)
        )

    def test_search_cpts_in_bbox(self):
        api = BroApi()
        cpts = api.search_objects_in_bbox(
            112400, 112500, 442600, 442800, object_type="CPT"
        )
        assert cpts == ["CPT000000000787", "CPT000000029403"]
