from pathlib import Path
from typing import Iterable, List, Literal

import rioxarray as rio
import xarray as xr

from geost.enums import UnitEnum
from geost.models import VoxelModel

from .bro_utils import coordinates_to_cellcenters, flip_ycoordinates


class Lithology(UnitEnum):  # pragma: no cover
    anthropogenic = 0
    organic = 1
    clay = 2
    loam = 3
    fine_sand = 5
    medium_sand = 6
    coarse_sand = 7
    gravel = 8
    shells = 9


class AntropogenicUnits(UnitEnum):  # pragma: no cover
    AAOP = 1000
    AAES = 1005


class HoloceneUnits(UnitEnum):  # pragma: no cover
    NIGR = 1010
    NINB = 1045
    NASC = 1020
    ONAWA = 1030
    NAZA = 1040
    NAWAZU = 1048
    NAWAAL = 1049
    NAWA = 1050
    BHEC = 1060
    OEC = 1070
    NAWOBE = 1080
    KK1 = 1085
    NIFL = 1089
    NIHO = 1090
    NAZA2 = 1095
    NAWO = 1100
    NWNZ = 1110
    NAWOVE = 1120
    KK2 = 1125
    NIBA = 1130
    NA = 2000
    EC = 2010
    NI = 2020
    KK = 2030


class OlderUnits(UnitEnum):  # pragma: no cover
    BXKO = 3000
    BXSI = 3010
    BXSI1 = 3011
    BXWI = 3020
    BXSI2 = 3012
    BXWIKO = 3025
    BXWISIKO = 3030
    BXDE = 3040
    BXDEKO = 3045
    BXSC = 3050
    BXLM = 3060
    BXBS = 3090
    BX = 3100
    KRWY = 4000
    KRBXDE = 4010
    KRZU = 4020
    KROE = 4030
    KRTW = 4040
    KR = 4050
    BEOM = 4055
    BEWY = 4060
    BERO = 4070
    BE = 4080
    KW1 = 4085
    KW = 4090
    WB = 4100
    EE = 4110
    EEWB = 4120
    DR = 5000
    DRGI = 5010
    GE = 5020
    DN = 5030
    URTY = 5040
    PE = 5050
    UR = 5060
    ST = 5070
    AP = 5080
    SY = 5090
    PZ = 5100
    WA = 5110
    PZWA = 5120
    MS = 5130
    KI = 5140
    OO = 5150
    IE = 5160
    VI = 5170
    BR = 5180
    VE = 5185
    RUBO = 5190
    RU = 5200
    TOZEWA = 5210
    TOGO = 5220
    TO = 5230
    DOAS = 5240
    DOIE = 5250
    DO = 5260
    LA = 5270
    HT = 5280
    HO = 5290
    MT = 5300
    GU = 5310
    VA = 5320
    AK = 5330


class ChannelBeltUnits(UnitEnum):  # pragma: no cover
    AEC = 6000
    ABEOM = 6005
    ANAWA = 6010
    ANAWO = 6020
    BEC = 6100
    BNAWA = 6110
    BNAWO = 6120
    CEC = 6200
    CNAWA = 6210
    CNAWO = 6220
    DEC = 6300
    DNAWA = 6310
    DNAWO = 6320
    EEC = 6400
    ENAWA = 6410
    ENAWO = 6420


class StratGeotop:
    """
    Class for making selections of stratigraphic units in GeoTop containing class attributes
    four main groups of stratigraphic units:

    - holocene : alias of :class:`~geost.bro.bro_geotop.HoloceneUnits`
    - channel : alias of :class:`~geost.bro.bro_geotop.ChannelBeltUnits`
    - older : alias of :class:`~geost.bro.bro_geotop.OlderUnits`
    - antropogenic : alias of :class:`~geost.bro.bro_geotop.AntropogenicUnits`

    Each of these class attributes is a :class:`~geost.enums.UnitEnum` which contains
    abbreviation (e.g. "NAWA") and corresponding number for geological units.

    Example usage
    -------------
    Select a group of "tidal" units by name:

    >>> tidal = StratGeotop.select_units(["NAWA", "ANAWA", "NAWO"])

    Or select a group of values:

    >>> selection = StratGeotop.select_values([1100, 6010, 6100])

    """

    holocene = HoloceneUnits
    channel = ChannelBeltUnits
    older = OlderUnits
    antropogenic = AntropogenicUnits

    @classmethod
    def select_units(cls, units: str | Iterable[str]):
        """
        Select units by name.

        Parameters
        ----------
        units : str | Iterable[str]
            Name as string or array_like object of strings with units to select.

        Returns
        -------
        List[enum]
            List of the selected units.

        Examples
        --------
        Select units based on name:

        >>> nawa = StratGeotop.select_units("NAWA")

        Or select a group of "tidal" units:

        >>> tidal = StratGeotop.select_units(["NAWA", "ANAWA", "NAWO"])

        """
        units = cls.holocene.cast_to_list(units)
        selection = (
            cls.holocene.select_units(units)
            + cls.channel.select_units(units)
            + cls.older.select_units(units)
            + cls.antropogenic.select_units(units)
        )
        return selection

    @classmethod
    def select_values(cls, values: int | Iterable[int]):
        """
        Select units by value.

        Parameters
        ----------
        units : str | Iterable[str]
            Number or array_like object of numbers with units to select.

        Returns
        -------
        List[enum]
            List of the selected units.

        Examples
        --------
        Select units based on value:

        >>> selection = StratGeotop.select_values(1100)

        Or select a group of values:

        >>> selection = StratGeotop.select_units([1100, 6010, 6100])

        """
        values = cls.holocene.cast_to_list(values)
        selection = (
            cls.holocene.select_values(values)
            + cls.channel.select_values(values)
            + cls.older.select_values(values)
            + cls.antropogenic.select_values(values)
        )
        return selection

    @classmethod
    def to_dict(cls, key: Literal["unit", "value"] = "value") -> dict:
        """
        Create a mapping dictionary from StratGeotop with "unit": "value" or "value": "unit"
        mapping.

        Parameters
        ----------
        key : Literal["unit", "value"], optional
            Determine what to use for "key" ("unit" or "value"). The default is "value".

        Returns
        -------
        dict

        """
        return (
            cls.holocene.to_dict(key)
            | cls.channel.to_dict(key)
            | cls.older.to_dict(key)
            | cls.antropogenic.to_dict(key)
        )


class GeoTop(VoxelModel):
    @classmethod
    def from_netcdf(
        cls,
        nc_path: str | Path,
        data_vars: List[str] = None,
        bbox: tuple[float, float, float, float] = None,
        lazy: bool = True,
        **xr_kwargs,
    ):
        """
        Read the BRO GeoTop subsurface model from a netcdf dataset into a GeoTop
        VoxelModel instance. The complete dataset of GeoTop can be downloaded from:
        https://dinodata.nl/opendap.

        Parameters
        ----------
        nc_path : str | Path
            Path to the netcdf file of GeoTop.
        data_vars : ArrayLike
            List or array-like object specifying which data variables to return.
        bbox : tuple (xmin, ymin, xmax, ymax), optional
            Specify a bounding box (xmin, ymin, xmax, ymax) to return a selected area of
            GeoTop. The default is None.
        lazy : bool, optional
            If True, netcdf loads lazily. Use False for speed improvements for larger
            areas but that still fit into memory. The default is False.
        **xr_kwargs
            Additional keyword arguments xarray.open_dataset. See relevant documentation
            for details.

        Returns
        -------
        GeoTop
            GeoTop instance of the netcdf file.

        """
        cellsize = 100
        dz = 0.5
        crs = 28992

        if lazy and "chunks" not in xr_kwargs:
            xr_kwargs["chunks"] = "auto"

        ds = xr.open_dataset(nc_path, **xr_kwargs)
        ds = coordinates_to_cellcenters(ds, cellsize, dz)

        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            ds = ds.sel(
                x=slice(xmin - (0.5 * cellsize), xmax + (0.5 * cellsize)),
                y=slice(ymin - (0.5 * cellsize), ymax + (0.5 * cellsize)),
            )

        if data_vars is not None:
            ds = ds[data_vars]

        if not lazy:
            print("Load data")
            ds = ds.load()

        ds = flip_ycoordinates(ds)
        ds = ds.rio.write_crs(crs)

        return cls(ds)

    @classmethod
    def from_opendap(
        cls,
        url: str = r"https://dinodata.nl/opendap/GeoTOP/geotop.nc",
        data_vars: List[str] = None,
        bbox: tuple = None,
        lazy: bool = True,
        **xr_kwargs,
    ):
        """
        Download an area of GeoTop directly from the OPeNDAP data server into a GeoTop
        VoxelModel instance.

        Parameters
        ----------
        url : str
            Url to the netcdf file on the OPeNDAP server. See:
            https://www.dinoloket.nl/modelbestanden-aanvragen
        data_vars : ArrayLike
            List or array-like object specifying which data variables to return.
        bbox : tuple (xmin, ymin, xmax, ymax), optional
            Specify a bounding box (xmin, ymin, xmax, ymax) to return a selected area of
            GeoTop. The default is None but for practical reasons, specifying a bounding
            box is advised (TODO: find max downloadsize for server).
        lazy : bool, optional
            If True, netcdf loads lazily. Use False for speed improvements for larger
            areas but that still fit into memory. The default is False.
        **xr_kwargs
            Additional keyword arguments xarray.open_dataset. See relevant documentation
            for details.

        Returns
        -------
        GeoTop
            GeoTop instance for the selected area.

        """
        return cls.from_netcdf(url, data_vars, bbox, lazy, **xr_kwargs)
