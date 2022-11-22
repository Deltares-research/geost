import pytest
from pathlib import Path

from pysst.bro import BroApi


class TestBroApi:
    def test_response(self):
        api = BroApi()
        assert api.session.get(api.server_url + api.cpt_api).status_code == 200
