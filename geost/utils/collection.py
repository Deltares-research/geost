from typing import Iterable, Literal

import pandas as pd

import geost
from geost import Collection


def concat(
    collections: Iterable[Collection], *, join: Literal["inner", "outer"] = "outer"
) -> Collection:
    """
    Concatenate multiple Collection instances into one Collection instance. The header
    and data tables of the given Collection instances will be concatenated separately
    and then combined into a new Collection instance.

    Parameters
    ----------
    collections : Iterable[:class:`~geost.base.Collection`]
        Collection instances to concatenate.
    join : {"inner", "outer"}, optional
        How to handle columns that are not shared between the header and data tables of
        the given Collection instances. "inner" will keep only the columns that are
        shared between all tables, while "outer" will keep all columns and fill missing
        values with NaN. The default is "outer".

    Returns
    -------
    :class:`~geost.base.Collection`
        New Collection instance resulting from the concatenation of the given
        Collection instances.

    Note
    ----
    The names of positional columns (e.g. "nr", "x", "y", "surface", "depth") in the header
    and data tables of the given Collection instances will be set to the positional column
    names of the first Collection in the iterable. For example, if the first Collection
    has "nr" as positional column for the survey name, but the second Collection has "id",
    the concatenated Collection will have "nr" as positional column for survey name.

    """
    # Check collections can be concatenated
    if not all(isinstance(collection, Collection) for collection in collections):
        raise ValueError("All inputs must be instances of the Collection class.")
    if not all(collection.crs == collections[0].crs for collection in collections):
        raise ValueError("All collections must have the same CRS to be concatenated.")
    if not all(
        collection.vertical_datum == collections[0].vertical_datum
        for collection in collections
    ):
        raise ValueError(
            "All collections must have the same vertical datum to be concatenated."
        )

    # Get list of headers and data with renamed positional columns to match those of the
    # first collection in the iterable
    headers = [
        collection.header.gst._rename_positional_columns_like(collections[0].header)
        for collection in collections
    ]
    datas = [
        collection.data.gst._rename_positional_columns_like(collections[0].data)
        for collection in collections
    ]

    # Concat and drop duplicates
    concatenated_header = pd.concat(
        headers, ignore_index=True, join=join
    ).drop_duplicates(subset=collections[0]._nr)
    concatenated_data = pd.concat(datas, ignore_index=True, join=join).drop_duplicates()

    return Collection(
        concatenated_data,
        header=concatenated_header,
        has_inclined=any(collection.has_inclined for collection in collections),
        vertical_datum=collections[0].vertical_datum,
    )
