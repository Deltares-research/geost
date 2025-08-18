from typing import Literal

import pytest
from lxml.etree import _Element

from geost.bro import BroApi


class TestBroApi:
    @pytest.mark.unittest
    def test_response(self):
        api = BroApi()
        for key in api.apis:
            # Exclude BRO Geological boreholes until added to BRO database
            # (They currently return a 503 status code)
            if key != "BHR-G":
                assert (
                    api.session.get(api.server_url + api.apis[key]).status_code == 200
                )

    @pytest.mark.parametrize(
        "objects, object_type",
        (
            (["CPT000000038771", "CPT000000000787", "CPT000000125133"], "CPT"),
            (["BHR000000123271", "BHR000000319277", "BHR000000051413"], "BHR-P"),
            (["BHR000000347437", "BHR000000347438", "BHR000000347434"], "BHR-GT"),
            (["BHR000000396339", "BHR000000396340", "BHR000000396341"], "BHR-G"),
            (["SFR000000002177", "SFR000000002172", "SFR000000002655"], "SFR"),
        ),
        ids=["CPT", "BHR-P", "BHR-GT", "BHR-G", "SFR"],
    )
    def test_get_valid_objects(self, objects, object_type):
        api = BroApi()
        objects = api.get_objects(objects, object_type=object_type)
        assert isinstance(objects, list)
        assert len(objects) == 3
        for obj in objects:
            assert isinstance(obj, bytes)

    @pytest.mark.parametrize(
        "bro_id, object_type, expected_exception",
        (
            ("NonExistingBroId", "CPT", "Unable to request NonExistingBroId"),
            ("NonExistingBroId", "BHR-GT", "Unable to request NonExistingBroId"),
        ),
    )
    def test_get_invalid_objects(
        self,
        bro_id: Literal["NonExistingBroId"],
        object_type: Literal["CPT"] | Literal["BHR-GT"],
        expected_exception: Literal["Unable to request NonExistingBroId"],
    ):
        api = BroApi()
        with pytest.raises(Warning) as excinfo:
            api.get_objects(bro_id, object_type=object_type)
            assert expected_exception in str(excinfo.value)

    @pytest.mark.unittest
    def test_search_and_get_cpts_in_bbox(self):
        # Note: if test fails, check if BRO database was updated first
        api = BroApi()
        api.search_objects_in_bbox(112400, 442750, 112500, 442850, object_type="CPT")
        minimum_present_objects = ["CPT000000000787", "CPT000000029403"]
        present_objects = [
            obj for obj in api.object_list if obj in minimum_present_objects
        ]
        assert len(present_objects) == len(minimum_present_objects)

    @pytest.mark.unittest
    def test_search_and_get_bhrps_in_bbox(self):
        # Note: if test fails, check if BRO database was updated first
        api = BroApi()

        api.search_objects_in_bbox(141500, 455100, 141700, 455300, object_type="BHR-P")
        minimum_present_objects = [
            "BHR000000085497",
            "BHR000000247842",
            "BHR000000120513",
            "BHR000000206176",
        ]
        present_objects = [
            obj for obj in api.object_list if obj in minimum_present_objects
        ]
        assert len(present_objects) == len(minimum_present_objects)

    @pytest.mark.unittest
    def test_search_and_get_bhrgts_in_bbox(self):
        # Note: if test fails, check if BRO database was updated first
        api = BroApi()
        # Find CPTs
        api.search_objects_in_bbox(141300, 452700, 142300, 453500, object_type="BHR-GT")
        minimum_present_objects = [
            "BHR000000353592",
            "BHR000000353583",
            "BHR000000353598",
            "BHR000000353600",
        ]
        present_objects = [
            obj for obj in api.object_list if obj in minimum_present_objects
        ]
        assert len(present_objects) == len(minimum_present_objects)

    @pytest.mark.unittest
    def test_get_too_large_volume(self, capsys: pytest.CaptureFixture[str]):
        api = BroApi()
        _ = api.search_objects_in_bbox(82000, 425000, 90000, 432000)
        captured = capsys.readouterr()
        assert "More than 2000 object requests in API call" in captured.out
