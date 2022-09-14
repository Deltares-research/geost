import pandas as pd
import pandera as pa
from pandera.typing import Series
from typing import Union, Optional


class EntriesSchema(pa.SchemaModel):
    """
    Check PointdataCollection.entries attribute for correct datatypes
    """

    nr: Series[str]
    x: Series[float]
    y: Series[float]
    mv: Series[float]
    end: Series[float]

    @pa.dataframe_check
    def check_mv(cls, df: pd.DataFrame) -> Series[bool]:
        return all(df["mv"] > df["end"])


class PointdataSchema(pa.SchemaModel):
    """
    Check dataframe for correct datatypes and coerce if required before creating PointdataCollection instance
    """

    nr: Series[str] = pa.Field(coerce=True)
    x: Series[float] = pa.Field(coerce=True)
    y: Series[float] = pa.Field(coerce=True)
    mv: Series[float] = pa.Field(coerce=True)
    end: Series[float] = pa.Field(coerce=True)


class BoreholeSchema(PointdataSchema):
    """
    Check dataframe for correct datatypes and coerce if required before creating BoreholeCollection instance
    """

    lith: Series[str] = pa.Field(coerce=True)
