import pytest
from pathlib import Path
from pysst.io.parsers import CptGefFile


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

    @pytest.mark.unittest
    def test_read_files(self, test_cpt_files):
        for f in test_cpt_files:
            cpt = CptGefFile(f)
            assert isinstance(cpt, CptGefFile)
    # TODO: create single cpt test func instead of all parametrize tests separate
    @pytest.mark.parametrize('cpt, header', [
        ('cpt_a', ['DKMP_D03', 176161.1, 557162.1, -0.06, 37.317295]), # TODO: create fixtures for cpt attributes to assert
        ('cpt_b', ['AZZ158', 0.0, 0.0, 5.05, 59.5]),
        ('cpt_c', ['CPT000000157983', 176416.1, 557021.9, -5.5, 28.84]),
        ('cpt_d', ['YANGTZEHAVEN CPT 10', 61949.0, 443624.0, -17.69, 5.75]),
    ])
    def test_cpt_header_info(self, cpt, header, request):
        cpt = request.getfixturevalue(cpt)

        nr, x, y, z, end = header

        assert cpt.nr == nr
        assert cpt.x == x
        assert cpt.y == y
        assert cpt.z == z
        assert cpt.enddepth == end

    @pytest.mark.parametrize('cpt', ['cpt_a', 'cpt_b', 'cpt_c', 'cpt_d'])
    def test_critical_columns_present(self, cpt, request):
        cpt = request.getfixturevalue(cpt)

        expected_cols = ['length', 'qc', 'fs']

        assert all(col in cpt.columns for col in expected_cols)

    @pytest.mark.parametrize('cpt, n', [
        ('cpt_a', 11),
        ('cpt_b', 6),
        ('cpt_c', 9),
        ('cpt_d', 4),
    ])
    def test_number_of_columns(self, cpt, n, request):
        cpt = request.getfixturevalue(cpt)
        assert cpt.ncolumns == n

    @pytest.mark.parametrize('cpt, system', [
        ('cpt_a', '31000'),
        ('cpt_b', '0'),
        ('cpt_c', '28992'),
        ('cpt_d', '31000'),
    ])
    def test_coordinate_system(self, cpt, system, request):
        cpt = request.getfixturevalue(cpt)
        assert cpt.ncolumns == system

    @pytest.mark.parametrize('cpt, elevation', [
        ('cpt_a', ('NAP', 0.01)),
        ('cpt_b', ('NAP', None)),
        ('cpt_c', ('NAP', None)),
        ('cpt_d', ('NAP', None)),
    ])
    def test_elevation_reference(self, cpt, elevation, request):
        cpt = request.getfixturevalue(cpt)

        reference, error = elevation

        assert cpt.reference_system == reference
        assert cpt.delta_z == error

    @pytest.mark.parametrize('cpt, nrecords', [
        ('cpt_a', 1815),
        ('cpt_b', 2980),
        ('cpt_c', 1310),
        # ('cpt_d', ), # TODO: fix data parsing of cpt_d
    ])
    def test_number_of_data_records(self, cpt, nrecords, request):
        cpt = request.getfixturevalue(cpt)
        assert len(cpt.df) == nrecords

    def cpta_test():


class TestFoo:

    @pytest.fixture
    def a(self):
        return 2
    
    @pytest.fixture
    def b(self):
        return 3

  