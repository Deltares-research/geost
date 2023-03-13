"""
This code is part of the core functionality of Deltares DataFusionTools. The 'Data' 
class is used to represent all spatial data, including boreholes and CPT's. In order to 
avoid a dependency on DataFusionTools (and its many dependencies that pysst doesn't 
need), the code was copied here to allow for the export of Pysst objects to 
DataFusionTools Data objects.

source:

https://bitbucket.org/DeltaresGEO/datafusiontools/src/master/

DataFusionTools is released under GNU GPL v3
"""

from dataclasses import dataclass
from typing import List, Optional, Union
import numpy as np
import pandas as pd


@dataclass
class Geometry:
    x: float
    y: float
    z: Optional[float] = None
    label: Optional[float] = None


@dataclass
class Variable:
    label: str
    value: np.array


@dataclass
class Data:
    """
    Class that contains values of one location

    :param location: The location the data
    :param variables: Variables that are contain in the data
    :param independent_variable: Independent variable that is connected to all the data
    """

    location: Geometry
    independent_variable: Union[Variable, None] = None
    variables: List[Variable] = []

    def get_variable(self, name: str):
        """
        Function that returns variable based on its name
        """
        for variable in self.variables:
            if variable.label == name:
                return variable
        return None

    def update_variable(self, name: str, value: np.ndarray):
        """
        Function that updates variable
        """
        index = None
        for counter, variable in enumerate(self.variables):
            if name == variable.label:
                index = counter
                continue
        if index is None:
            raise ValueError(f"Name provided {name} is defined in this class instance.")
        self.variables[index].value = value

    @property
    def variables(self):
        return self._variables

    @variables.setter
    def variables(self, values: List[Variable]):
        if self.independent_variable is not None:
            for variable in values:
                if len(variable.value) != len(self.independent_variable.value):
                    raise ValueError(
                        f"Length of variable {variable.label} is not the same as that of the independent variable."
                    )
        self._variables = values


########################################################################################
################################# PYSST ADDITIONS ######################################
########################################################################################


def export_to_dftgeodata(data, columns):
    # Prepare and encode labelled data

    data_encoded = pd.get_dummies(data.loc[:, columns])
    encoded_columns = data_encoded.columns
    data_prepared = data.drop(columns, axis=1)
    data_prepared = pd.concat([data_prepared, data_encoded], axis=1)

    objects = data_prepared.groupby("nr")
    geodataclasses = []

    for nr, obj in objects:
        variables = []
        location = Geometry(obj.x.iloc[0], obj.y.iloc[0], z=obj.mv.iloc[0])
        independent_var_depth = Variable(
            value=(obj.top - (obj.top - obj.bottom) / 2).values, label="height"
        )
        for variable_column in encoded_columns:
            variables.append(
                Variable(value=obj[variable_column].values, label=variable_column)
            )
        geodataclasses.append(Data(location, independent_var_depth, variables))

    return geodataclasses
