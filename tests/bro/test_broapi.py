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

    @pytest.mark.unittest
    def test_get_valid_cpts(self):
        api = BroApi()
        cpt_objects = api.get_objects(
            ["CPT000000038771", "CPT000000000787", "CPT000000125133"], object_type="CPT"
        )
        assert isinstance(cpt_objects, list)
        assert len(cpt_objects) == 3
        for cpt in cpt_objects:
            assert isinstance(cpt, bytes)

    @pytest.mark.unittest
    def test_get_valid_bhrps(self):
        api = BroApi()
        bhrp_objects = api.get_objects(
            ["BHR000000342629", "BHR000000195688", "BHR000000206176"],
            object_type="BHR-P",
        )
        assert len(bhrp_objects) == 3
        for bhrp in bhrp_objects:
            assert isinstance(bhrp, bytes)

    @pytest.mark.unittest
    def test_get_valid_bhrgts(self):
        api = BroApi()
        bhrgt_objects = api.get_objects(
            ["BHR000000347437", "BHR000000347438", "BHR000000347434"],
            object_type="BHR-GT",
        )
        assert len(bhrgt_objects) == 3
        for bhrgt in bhrgt_objects:
            assert isinstance(bhrgt, bytes)

    @pytest.mark.xfail(reason="BRO Geological boreholes not yet available from API")
    def test_get_valid_bhrgs(self):
        api = BroApi()
        bhrg_objects = api.get_objects(
            ["BHR000000342629", "BHR000000195688", "BHR000000206176"],
            object_type="BHR-G",
        )
        assert len(bhrg_objects) == 3
        for bhr_data in bhrg_objects:
            assert isinstance(bhr_data, _Element)

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
