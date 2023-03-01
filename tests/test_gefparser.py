import pytest
from pathlib import Path
from pysst.io.parsers import CptGefFile


class TestCptGefParser:
    @pytest.fixture
    def test_cpt_files(self):
        file_path = Path(__file__).parent
        test_cpt_files = list(file_path.glob('data\cpt\*.gef'))
        return test_cpt_files

    @pytest.mark.unittest
    def test_read_files(self, test_cpt_files):
        for f in test_cpt_files:
            cpt = CptGefFile(f)
            print(cpt, "okay")
            assert isinstance(cpt, CptGefFile)
