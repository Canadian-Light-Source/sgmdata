# General imports
import glob
# from IPython.testing.globalipapp import get_ipython
import asyncio
import h5py
import pandas
# from dask.distributed import Client
# from distributed import Client
import numpy as np
from lmfit import Model
from collections import OrderedDict
# Plotting function imports
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import HoverTool, CustomJS, BooleanFilter, LinearColorMapper, LogColorMapper, ColorBar
from bokeh.io import show
### FOLLOWING JUST FOR MY COMP
import os
import sys
import inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
### JUST FOR MY COMP DONE
import sgmdata
from sgmdata.load import SGMData
import math
from sgmdata.utilities.util import plot1d


def file_retrieval(path):
    """
    Purpose: Gets files that match a specified pattern from a specified directory.
    Parameters:
        path(str): path to directory and pattern to look for in directory.
    Returns:
        list_of_files(list): a list of the files that match the given pattern.
    """
    if type(path) != str:
        if type(path) == int or type(path) == float or type(path) == bool:
            raise TypeError("Argument must be a path to an hdf5 file, in string form. Cannot take data of type " +
                            str(type(path)) + ". Please try again with a path to an hdf5 file in string form.")
        else:
            raise TypeError("Argument must be a single hdf5 file. Function cannot take compound data type containing "
                            "hdf5 files. Please try again with a single hdf5 file.")
    if ".hdf5" not in path[-6:]:
        raise ValueError("Argument must be an hdf5 file. An hdf5 file was not passed in. Please try again with an hdf5"
                         " file.")
    list_of_files = []
    for filename in glob.iglob(path, recursive=True):
        list_of_files.append(filename)
    if len(list_of_files) == 0:
        raise ValueError("There are no files that match the given pattern. Please try again with a different pattern.")
    return list_of_files


def check_sample_fitness(list_of_files):
    """
    Purpose: Create an SGMData object from hdf5 files. Interpolate the data in the SGMData object. Return this
    interpolated data.
    Parameters:
        list_of_files(list of str): the names of the hdf5 files to create an SGMData object out of.
    Returns:
        interp_list(list): a list of pandas dataframes. Contains the interpolated version of the data in the files
        specified in list_of_files.
    """
    ### Test data for SGMScan
    sgm_data = sgmdata.load.SGMData(list_of_files)
    # print(sgm_data)
    sgm_data.interpolate()

    data = sgm_data
    print(data._repr_console_())
    # keys = data.scans.keys()
    # print(sgm_data.scans)
    # for item in keys:
    #     print(data.scans[item]._repr_console_())
        # data.scans[item]._repr_console_()


x = file_retrieval('C:/Users/roseh/Desktop/Internship/SignalToNoiseConvergence/h5Files/*Co-nitrate*.hdf5')
# x = file_retrieval('C:/Users/roseh/Desktop/Internship/MyCode/h5Files/*bee*.hdf5')
y = check_sample_fitness(x)