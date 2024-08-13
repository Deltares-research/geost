# class CptCollection(PointDataCollection):
#     """
#     Class for collections of CPT data.

#     Users must use the reader functions in
#     :py:mod:`~geost.read` to create collections. The following readers generate CPT
#     objects:

#     :func:`~geost.read.read_sst_cpts`, :func:`~geost.read.read_gef_cpts`

#     Args:
#         data (pd.DataFrame): Dataframe containing borehole/CPT data.

#         vertical_reference (str): Vertical reference, see
#          :py:attr:`~geost.base.PointDataCollection.vertical_reference`

#         horizontal_reference (int): Horizontal reference, see
#          :py:attr:`~geost.base.PointDataCollection.horizontal_reference`

#         header (pd.DataFrame): Header used for construction. see
#          :py:attr:`~geost.base.PointDataCollection.header`
#     """

#     def __init__(
#         self,
#         data: pd.DataFrame,
#         vertical_reference: str = "NAP",
#         horizontal_reference: int = 28992,
#         header: Optional[pd.DataFrame] = None,
#         is_inclined: bool = False,
#     ):
#         super().__init__(
#             data,
#             vertical_reference,
#             horizontal_reference,
#             header=header,
#             is_inclined=is_inclined,
#         )

#     def add_ic(
#         self,
#     ):  # Move to cpt analysis functions, use something like 'apply' function in classes
#         """
#         Calculate soil behaviour type index (Ic) for all CPT's in the collection.

#         The data is added to :py:attr:`~geost.base.PointDataCollection.header`.
#         """
#         self.data["ic"] = calc_ic(self.data["qc"], self.data["friction_number"])

#     def add_lithology(
#         self,
#     ):  # Move to cpt analysis functions, use something like 'apply' function in classes
#         """
#         Interpret lithoclass for all CPT's in the collection.

#         The data is added to :py:attr:`~geost.base.PointDataCollection.header`.
#         """
#         if "ic" not in self.data.columns:
#             self.add_ic()
#         self.data["lith"] = calc_lithology(
#             self.data["ic"], self.data["qc"], self.data["friction_number"]
#         )

#     def as_boreholecollection(self):  # No change
#         """
#         Export CptCollection to BoreholeCollection. Requires the "lith" column to be
#         present. Use the method :py:meth:`~geost.borehole.CptCollection.add_lithology`

#         Returns
#         -------
#         Instance of :class:`~geost.borehole.BoreholeCollection`
#         """
#         if "lith" not in self.data.columns:
#             raise IndexError(
#                 r"The column \"lith\" is required to convert to BoreholeCollection"
#             )

#         borehole_converted_dataframe = self.data[
#             ["nr", "x", "y", "surface", "end", "top", "bottom", "lith"]
#         ]
#         cptcollection_as_bhcollection = BoreholeCollection(
#             borehole_converted_dataframe,
#             vertical_reference=self.vertical_reference,
#             header=self.header,
#         )
#         return cptcollection_as_bhcollection
