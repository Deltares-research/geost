from pathlib import Path

import geopandas as gpd
import pandas as pd

from geost.utils import create_connection


class Geopackage:
    """
    Read and inspect the contents of a geopackage file. The class is designed to be used
    as a context manager to ensure the connection to the geopackage is closed after use.

    Parameters
    ----------
    file : str | Path
        Valid string path to the Geopackage file.

    Examples
    --------
    Check the available layers in a geopackage file:

    >>> print(Geopackage("my_geopackage.gpkg").layers())

    Read the first five records of a table in the geopackage for a quick inspection:

    >>> with Geopackage("my_geopackage.gpkg") as gp:
    ...     print(gp.table_head("my_table"))

    For use without a context manager:

    >>> gp = Geopackage("my_geopackage.gpkg")
    >>> gp.get_connection()
    >>> table = gp.read_table("my_table") # Returns a DataFrame of the table.

    """

    def __init__(self, file: str | Path):
        self.file = file
        self.connection = None

    def __enter__(self):
        self.get_connection()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _get_cursor(self):
        """
        Get a sqlite3 cursor object for the geopackage connection to execute database
        queries with. This method is used internally to execute queries and should not
        be used directly.

        """
        return self.connection.cursor()

    def layers(self) -> pd.DataFrame:
        """
        Return a Pandas DataFrame containing the layers and associated geometry types in
        the geopackage.

        """
        return gpd.list_layers(self.file)

    def get_connection(self):
        """
        Create a manual sqlite3 connection to the geopackage file. Establishing and closing
        this connection is automatically handled when the class is used as a context manager.

        """
        self.connection = create_connection(self.file)

    def close(self):
        """
        Close the connection to the geopackage file.

        """
        if self.connection:
            self.connection = None

    def get_column_names(self, table: str) -> list:
        """
        Get the column names of a table in the geopackage.

        Parameters
        ----------
        table : string
            Name of the table to get the column names for.

        Returns
        -------
        columns : list
            List of the column names for the table.

        """
        cursor = self._get_cursor()
        cursor.execute(f"SELECT * FROM {table}")
        columns = [col[0] for col in cursor.description]
        return columns

    def table_head(self, table: str) -> pd.DataFrame:
        """
        Select the first five records from a table the in geopackage for quick inspection.

        Parameters
        ----------
        table : string
            Name of the table to select the first records from.

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame of the first five records.

        """
        cursor = self._get_cursor()
        cursor.execute(f"SELECT * FROM {table} LIMIT 5")
        data = cursor.fetchall()
        return pd.DataFrame(data, columns=self.get_column_names(table))

    def read_table(self, table: str) -> pd.DataFrame:
        """
        Read all data in a specified table of the geopackage.

        Parameters
        ----------
        table : str
            Name of the table to read. Check the available tables with `Geopackage.layers()`.

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame of the table.

        """
        columns = self.get_column_names(table)
        table = self.query(f"SELECT * FROM {table}", columns)
        return table

    def query(self, query: str, outcolumns: list = None) -> pd.DataFrame:
        """
        Use a custom query on the geopackage to retrieve desired tables or joined information
        from multiple tables.

        Parameters
        ----------
        query : str
            Full string of the SQL query to retrieve the desired table with.
        outcolumns : list, optional
            Specify column names to be used for the output table. The default is
            None.

        Returns
        -------
        pd.DataFrame
            Result DataFrame of the query.

        """
        cursor = self._get_cursor()
        cursor.execute(query)
        data = cursor.fetchall()

        if outcolumns is None:
            outcolumns = [col[0] for col in cursor.description]

        return pd.DataFrame(data, columns=outcolumns)
