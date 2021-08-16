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
    sgm_data = sgmdata.load.SGMData(list_of_files)
    file = list(sgm_data.__dict__['scans'].keys())
    print(file)
    # i = 0
    # while i < len(file):
    #     print("\n" + str(file[i]))
    #     for item in sgm_data.__dict__['scans'][file[i]].__dict__.keys():
    #         print(item)
    #     i += 1

    # if len(sgm_data.__dict__['scans']) == 0:
    #     raise ValueError("hdf5 file must contain scans to be able to predict the number of scans required. The hdf5 "
    #                      "file you have provided does not contain any scans. PLease try again with an hdf5 file that"
    #                      " does contain scans.")
    # has_sdd = False
    # file = list(sgm_data.__dict__['scans'].keys())
    # sample_name = list(sgm_data.__dict__['scans'][file[0]].__dict__.keys())
    # signals = list(sgm_data.__dict__['scans'][file[0]].__getitem__(sample_name[0]).__getattr__('signals').keys())
    # i = 0
    # while i < len(signals) and not has_sdd:
    #     if "sdd" in signals[i]:
    #         has_sdd = True
    #     else:
    #         i += 1
    # if not has_sdd:
    #     raise ValueError("Scans must have sdd values to be able to predict the number of scans required. One or "
    #                      "more of the scans you have provided do not have sdd values. Please try again using "
    #                      "scans with sdd values. ")
    # sample_type = sgm_data.__dict__['scans'][file[0]].__getitem__(sample_name[0])['sample']
    # for indiv_file in file:
    #     sample_name = list(sgm_data.__dict__['scans'][indiv_file].__dict__.keys())
    #     for scan in sample_name:
    #         if sgm_data.__dict__['scans'][indiv_file].__getitem__(scan)['sample'] != sample_type:
    #             raise ValueError("In order to predict, the scans in the hdf5 file passed in by user must all be from"
    #                              " the same sample. The scans in the hdf5 file passed in by the user are not all from"
    #                              " the same sample. Please "
    #                              "try again with an hdf5 file only containing scans from the"
    #                              " same sample. ")
    # interp_list = sgm_data.interpolate(resolution=0.1)
    # return interp_list


x = file_retrieval('C:/Users/roseh/Desktop/Internship/MyCode/h5Files/*bee2-c*.hdf5')
y = check_sample_fitness(x)
# print(y)

