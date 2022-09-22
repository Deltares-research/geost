import pandas as pd
import pandera as pa
from pandera.typing import Series
from typing import Union, Optional


class EntriesdataSchema(pa.SchemaModel):
    pass


class PointdataSchema(pa.SchemaModel):
    """
    Check dataframe for correct datatypes and coerce if required before creating PointdataCollection instance.
    The columns in this schema are required for both boreholes and CPTs.
    """

    nr: Series[str] = pa.Field(coerce=True)
    x: Series[float] = pa.Field(coerce=True)
    y: Series[float] = pa.Field(coerce=True)
    mv: Series[float] = pa.Field(coerce=True)
    end: Series[float] = pa.Field(coerce=True)
    top: Series[float] = pa.Field(coerce=True)
    bottom: Series[float] = pa.Field(coerce=True)

    @pa.dataframe_check
    def check_borehole_maaiveld_higher_than_end(cls, df: pd.DataFrame) -> Series[bool]:
        return all(df["mv"] > df["end"])

    @pa.dataframe_check
    def check_layer_top_higher_than_bottom(cls, df: pd.DataFrame) -> Series[bool]:
        return all(df["top"] > df["bottom"])


class BoreholeSchema(PointdataSchema):
    """
    Check dataframe for correct datatypes and coerce if required before creating BoreholeCollection instance.
    """

    lith: Series[str] = pa.Field(coerce=True)


class CptSchema(PointdataSchema):
    """
    Check dataframe for correct datatypes and coerce if required before creating CptCollection instance.
    """

    lith: Series[str] = pa.Field(coerce=True)
