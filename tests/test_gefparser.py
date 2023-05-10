import pytest
from pathlib import Path
from typing import NamedTuple
from pysst.io.parsers import CptGefFile


class CptInfo(NamedTuple):
    nr: str
    x: float
    y: float
    z: float
    end: float
    ncols: int
    system: str
    reference: str
    error: float
    nrecords: int


class TestCptGefParser:
    filepath = Path(Path(__file__).parent, 'data', 'cpt')

    @pytest.fixture
    def test_cpt_files(self):
        test_cpt_files = list(self.filepath.glob('*.gef'))
        return test_cpt_files

    @pytest.fixture
    def cpt_a(self):
        cpt = CptGefFile(
            self.filepath/r'83268_DKMP003_(DKMP_D03)_wiertsema.gef'
        )
        return cpt

    @pytest.fixture
    def cpt_b(self):
        cpt = CptGefFile(self.filepath/r'AZZ158_gem_rotterdam.gef')
        return cpt

    @pytest.fixture
    def cpt_c(self):
        cpt = CptGefFile(self.filepath/r'CPT000000157983_IMBRO.gef')
        return cpt

    @pytest.fixture
    def cpt_d(self):
        cpt = CptGefFile(self.filepath/r'CPT10_marine_sampling.gef')
        return cpt
    
    @pytest.fixture
    def test_info_cpt_a(self):
        return CptInfo(
            'DKMP_D03', 176161.1, 557162.1, -0.06, 37.317295, 11, '31000', 'NAP', 0.01, 1815
            )
    
    @pytest.fixture
    def test_info_cpt_b(self):
        return CptInfo(
            'AZZ158', 0.0, 0.0, 5.05, 59.5, 6, '0', 'NAP', None, 2980
            )

    @pytest.fixture
    def test_info_cpt_c(self):
        return CptInfo(
            'CPT000000157983', 176416.1, 557021.9, -5.5, 28.84, 9, '28992', 'NAP', None, 1310
            )
    
    @pytest.fixture
    def test_info_cpt_d(self):
        return CptInfo(
            'YANGTZEHAVEN CPT 10', 61949.0, 443624.0, -17.69, 5.75, 4, '31000', 'NAP', None, None
            )

    @pytest.mark.unittest
    def test_read_files(self, test_cpt_files):
        for f in test_cpt_files:
            cpt = CptGefFile(f)
            assert isinstance(cpt, CptGefFile)
    
    @pytest.mark.integrationtest
    @pytest.mark.parametrize('cpt, test_info', [
        ('cpt_a', 'test_info_cpt_a'),
        ('cpt_b', 'test_info_cpt_b'),
        ('cpt_c', 'test_info_cpt_c'),
        ('cpt_d', 'test_info_cpt_d'),
    ])
    def test_cpt_parsing_result(self, cpt, test_info, request):
        cpt = request.getfixturevalue(cpt)
        test_info = request.getfixturevalue(test_info)

        assert cpt.nr == test_info.nr
        assert cpt.x == test_info.x
        assert cpt.y == test_info.y
        assert cpt.z == test_info.z
        assert cpt.enddepth == test_info.end
        assert cpt.ncolumns == test_info.ncols
        assert cpt.coord_system == test_info.system
        assert cpt.reference_system == test_info.reference
        assert cpt.delta_z == test_info.error

        critical_cols_from_file = ['length', 'qc', 'fs']
        assert all(col in cpt.columns for col in critical_cols_from_file)

        if cpt.nr != 'YANGTZEHAVEN CPT 10':  # TODO: Fix Yangtzehaven parsing bug.
            assert len(cpt.df) == test_info.nrecords
  