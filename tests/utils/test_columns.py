import pytest

from geost.utils import columns


@pytest.mark.parametrize(
    "input_col, prefix, suffix, expected",
    [
        ([1, 2, 3], "", "_sum", "1,2,3_sum"),
        ("Z", "mean_", "_thickness", "mean_Z_thickness"),
        (slice(1, 10), "range_", "", "range_1:10"),
        ({"a", "b", "c"}, "", "", "a,b,c"),
        ({"invalid": "invalid"}, None, None, None),
    ],
    ids=["string", "list", "slice", "set", "invalid_dict"],
)
def test_column_name_from(input_col, prefix, suffix, expected, request):
    test_id = request.node.callspec.id
    if "invalid" in test_id:
        with pytest.raises(TypeError):
            columns.column_name_from(input_col, prefix=prefix, suffix=suffix)
    else:
        assert (
            columns.column_name_from(input_col, prefix=prefix, suffix=suffix)
            == expected
        )
