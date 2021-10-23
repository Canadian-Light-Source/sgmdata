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
    sgm_data.interpolate()
    data = sgm_data.__dict__
    # print(data)
    print(sgm_data._repr_console_())
#
x = file_retrieval('C:/Users/roseh/Desktop/Internship/MyCode/h5Files/*Co-nitrate*.hdf5')
# # x = file_retrieval('C:/Users/roseh/Desktop/Internship/MyCode/h5Files/*bee*.hdf5')
y = check_sample_fitness(x)

# import logging
# import threading
# import time
# import concurrent.futures
#
#
# def thread_function(name):
#     logging.info("Thread %s: starting", name)
#     time.sleep(2)
#     logging.info("Thread %s: finishing", name)
#
#
# if __name__ == "__main__":
#     format="%(asctime)s: %(message)s"
#     logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
#
#     with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
#         executor.map(thread_function, range(3))


### TESTING POOL
# from multiprocessing import Process
# import os
#
#
# def info(title):
#     print(title)
#     print('module name:', __name__)
#     print('parent process:', os.getppid())
#     print('process id:', os.getpid())
#
#
# def f(name):
#     info('function f')
#     print('hello', name)
#
#
# if __name__ == '__main__':
#     info('main line')
#     p = Process(target=f, args=('bob',))
#     p.start()
#     p.join()
#
#
### TESTING MULTIPROCESSING.POOL
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# was not commented out when I started working again
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# from multiprocessing import Pool
# import time
#
#
# def f(x):
#     return x*x
#
#
# if __name__ == '__main__':
#     with Pool(processes=4) as pool:         # start 4 worker processes
#         result = pool.apply_async(f, (10,)) # evaluate "f(10)" asynchronously in a single process
#         print(result.get(timeout=1))        # prints "100" unless your computer is *very* slow
#
#         print(pool.map(f, range(10)))       # prints "[0, 1, 4,..., 81]"
#
#         it = pool.imap(f, range(10))
#         print(next(it))                     # prints "0"
#         print(next(it))                     # prints "1"
#         print(it.next(timeout=1))           # prints "4" unless your computer is *very* slow
#
#         result = pool.apply_async(time.sleep, (10,))
#         # print(result.get(timeout=1))        # raises multiprocessing.TimeoutError

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *



