import pytest
from pathlib import Path
from lxml.etree import _Element

from pysst.bro import BroApi


class TestBroApi:
    @pytest.mark.unittest
    def test_response(self):
        api = BroApi()
        for key in api.apis:
            assert api.session.get(api.server_url + api.apis[key]).status_code == 200

    @pytest.mark.unittest
    def test_get_valid_cpts(self):
        api = BroApi()
        cpt_datas = api.get_objects(
            ["CPT000000038771", "CPT000000000787", "CPT000000125133"], object_type="CPT"
        )
        for cpt_data in cpt_datas:
            assert isinstance(cpt_data, _Element)

    @pytest.mark.unittest
    def test_get_valid_bhrps(self):
        api = BroApi()
        bhr_datas = api.get_objects(
            ["BHR000000342629", "BHR000000195688", "BHR000000206176"],
            object_type="BHR-P",
        )
        for bhr_data in bhr_datas:
            assert isinstance(bhr_data, _Element)

    @pytest.mark.unittest
    def test_get_valid_bhrgts(self):
        api = BroApi()
        bhr_datas = api.get_objects(
            ["BHR000000347437", "BHR000000347438", "BHR000000347434"],
            object_type="BHR-GT",
        )
        for bhr_data in bhr_datas:
            assert isinstance(bhr_data, _Element)

    @pytest.mark.unittest
    def test_get_valid_bhrgs(self):
        pass
        # BHR-G objects are not yet available in the BRO

        # api = BroApi()
        # bhr_datas = api.get_objects(
        #     ["BHR000000342629", "BHR000000195688", "BHR000000206176"],
        #     object_type="BHR-G",
        # )
        # for bhr_data in bhr_datas:
        #     assert isinstance(bhr_data, _Element)

    @pytest.mark.unittest
    def test_get_invalid_cpt(self):
        api = BroApi()
        cpt_datas = api.get_objects("CPT0000000doesnotexist", object_type="CPT")
        with pytest.raises(Warning) as excinfo:
            for cpt_data in cpt_datas:
                pass
        assert (
            "CPT0000000doesnotexist is invalid and could not be retrieved from the database"
            in str(excinfo.value)
        )

    @pytest.mark.unittest
    def test_get_invalid_bhr(self):
        api = BroApi()
        bhr_datas = api.get_objects("BHR0000000doesnotexist", object_type="BHR-GT")
        with pytest.raises(Warning) as excinfo:
            for bhr_data in bhr_datas:
                pass
        assert (
            "BHR0000000doesnotexist is invalid and could not be retrieved from the database"
            in str(excinfo.value)
        )

    @pytest.mark.unittest
    def test_search_and_get_cpts_in_bbox(self):
        # Note: if test fails, check if BRO database was updated first
        api = BroApi()
        # Find CPTs
        cpts = api.search_objects_in_bbox(
            112400, 112500, 442750, 442850, object_type="CPT"
        )
        assert cpts == ["CPT000000000787", "CPT000000029403"]
        # Get CPTs
        cpt_datas = api.get_objects(cpts, object_type="CPT")
        for cpt_data in cpt_datas:
            assert isinstance(cpt_data, _Element)

    @pytest.mark.unittest
    def test_search_and_get_bhrps_in_bbox(self):
        # Note: if test fails, check if BRO database was updated first
        api = BroApi()
        # Find CPTs
        bhrps = api.search_objects_in_bbox(
            141500, 141700, 455100, 455300, object_type="BHR-P"
        )
        assert bhrps == [
            "BHR000000085497",
            "BHR000000247842",
            "BHR000000120513",
            "BHR000000206176",
        ]
        # Get CPTs
        bhr_datas = api.get_objects(bhrps, object_type="BHR-P")
        for bhr_data in bhr_datas:
            assert isinstance(bhr_data, _Element)

    @pytest.mark.unittest
    def test_search_and_get_bhrgts_in_bbox(self):
        # Note: if test fails, check if BRO database was updated first
        api = BroApi()
        # Find CPTs
        bhrgts = api.search_objects_in_bbox(
            141300, 142300, 452700, 453500, object_type="BHR-GT"
        )
        assert bhrgts == [
            "BHR000000353592",
            "BHR000000353583",
            "BHR000000353598",
            "BHR000000353600",
        ]
        # Get CPTs
        bhr_datas = api.get_objects(bhrgts, object_type="BHR-GT")
        for bhr_data in bhr_datas:
            assert isinstance(bhr_data, _Element)
