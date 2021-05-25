# General imports
import glob
from IPython.testing.globalipapp import get_ipython
import asyncio
import h5py
import pandas
from dask.distributed import Client
from distributed import Client
import numpy as np
import sgmdata
from lmfit import Model
from sgmdata.load import SGMData
# Plotting function imports
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import HoverTool, CustomJS, BooleanFilter, LinearColorMapper, LogColorMapper, ColorBar
from bokeh.io import show

ip = get_ipython()
# # ip.run_line_magic('pip', 'install lmfit')


# GETTING DATA FILES FROM DISK.
l = []
# Getting all files (intended to be hdf5 files) that match a specific pattern. Appending them to l.
for filename in glob.iglob('/Users/roseh/Desktop/Internship/MyCode/h5Files/*Bee*.hdf5', recursive=True):
    l.append(filename)


# LOADING DATA
# Reports amount of time required to execute command.
# # ip.run_line_magic('time', 'sgm_data = SGMData(l, sample="Ni-chloride_-_Ni-b9da")') # Runs timer.
# Creates a new SGMData.
sgm_data = SGMData(l, sample='Bee2-C')
print(str(sgm_data) + "\n\n")

# Reports amount of time required to execute command.
# # ip.run_line_magic('time', 'sgm_data')
# Displays data stored in our sgm_data object on the console.
# # # sgm_data

# Reports amount of time required to execute command.
# # ip.run_line_magic('time', 'interp_list = sgm_data.interpolate(resolution=0.1)')
# Interpolates data in our sgm_data object. Puts interpolated data into interp_list.
interp_list = sgm_data.interpolate(resolution=0.1, sample='Bee2-C')
print(interp_list)

# # SCRATCH
# def plot1d(xarr, yarr):
#     """
#     Sets the specifications for a graph, then shows graph. Graph represents the data samples, and the predicted data
#     samples.
#     Variables:
#         xarr: a list of two lists containing the x values for the points we will be plotting.
#         yarr: a list of two lists containing the y values for the points we will be plotting.
#     """
#     # A string listing the tools we will have available for our graph.
#     TOOLS = 'pan, hover, box_zoom, box_select, crosshair, reset, save'
#     # Specifying the appearance of our graph.
#     fig = figure(
#         tools=TOOLS,
#         title="Plot",
#         background_fill_color="white",
#         background_fill_alpha=1,
#         x_axis_label="x",
#         y_axis_label="y",
#     )
#     colors = []
#     # For every group of six in yarr (rounded down) and once more, add " 'purple', 'black', 'yellow', 'firebrick',
#     # 'red', 'orange' " to the 'colors' list.
#     for i in range(np.floor(len(yarr) / 6).astype(int) + 1):
#         colors += ['purple', 'black', 'yellow', 'firebrick', 'red', 'orange']
#     # Making colors variable into an iterator that can iterate through the previous version of colors.
#     colors = iter(colors)
#     if not isinstance(xarr, list):
#         xarr = [xarr]
#     if not len(xarr) == len(yarr):
#         yarr = [yarr]
#     # For the number of items in xarr, plot a new point on our graph.
#     for i, x in enumerate(xarr):
#         fig.circle(x=x, y=yarr[i], color=next(colors), legend_label="Curve" + str(i))
#     fig.legend.location = "top_left"
#     fig.legend.click_policy = "hide"
#     show(fig)
#
#
# # INTERPOLATING DATA
# sdd_list = []
#
# # Checking for items in interp_list containing the characters 'sdd.' those items are appended to sdd_list.
# for df in interp_list:
#     sdd_list.append(df.filter(regex=("sdd.*"), axis=1).to_numpy())
#
# prev_mean = sdd_list[0]
# avg_list = [sdd_list[0]]
# diff_list = []
# indices = []
#
# for i, arr in enumerate(sdd_list[1:]):
# # *Note that arr is the array of arrays representing the each scan's sdd values and i is the count of the number of
# # times the loop has been cycled through, starting at 0.
#     avg_list.append(arr)
#     cur_mean = np.mean(avg_list, axis=0)
#
#     # * Cur_mean is an array with the mean of all values in the same positions in the in arrays within avg_list. Eg, the
#     # average of the item [0] of each of the lists, then the average of the item[1] of each of the lists, etc.
#
#     # Finding the variance between the last scan and the current scan.
#     diff_list.append(np.var(np.subtract(cur_mean, prev_mean)))
#     if len(diff_list) > 2:
#         if diff_list[-2] - diff_list[-1] < -50:
#             avg_list = avg_list[:-1]
#             diff_list = diff_list[:-1]
#             # If the variance between the difference between our current scan's mean and the mean of the scan before it
#             # is 51 or more than the variance of the difference between the previous scan's mean and the mean of the
#             # scan before it, then don't include the variance of the difference between the current scan's mean and the
#             # previous scan's mean in diff list, or the average of this scan in avg_list.
#         else:
#             indices.append(i)
#             prev_mean = cur_mean
#     else:
#         indices.append(i)
#         prev_mean = cur_mean
#
# # Turns diff_list into and array, then converts NaN values to 0s and deals with infinity values.
# diff_list = np.nan_to_num(np.array(diff_list))
#
#
# def noise(idx, amp, shift, ofs):
#     """
#     Purpose: Calculates the level of noise in a scan.
#     Variables:
#         idx: Independent variable. An array of the indices of the values in sdd_list that were previously deemed to be
#         acceptable sdd values.
#         amp: Dependant variable. How much to amplify the noise level by.
#         shift: Dependant variable. Amount to shift the index by.
#         ofs: Dependant variable.
#     """
#     # Fitting of parametrized noise reduction equation
#     return amp * (idx + shift) ** (-3 / 2) + ofs
#
#
# # Using lmfit to prepare the execute noise, initially without parameters.
# noisemodel = Model(noise)
#
# # Creates the parameters for our 'noisemodel' calling of the 'noise' function, but does not assign values to them.
# params = noisemodel.make_params()
# # Setting the values of 'amp', 'shift' and 'ofs' for our 'noisemodel' calling of the 'noise' function.
# params['amp'].set(value=diff_list[0], vary=True, min=0.1 * diff_list[0], max=2 * diff_list[0])
# params['shift'].set(value=0.1, vary=True, min=-2, max=2)
# params['ofs'].set(value=0, vary=True, min=-1, max=0)
#
# # Fits our 'noisemodel' calling of the 'noise' function with data set to diff_list and idx set to an arrayed version of
# # indices and the same parameters we previously set.
# result_noise = noisemodel.fit(diff_list, idx=np.array(indices), params=params)
# values = result_noise.best_values
# # *Calling the plot1d function with yarr equal to a list containing the list of indices, twice. xarr is set to diff_list
# # and the results of the following pseudocode:
# # for every element in indices + 1
# #   element = (element - shift) to the power of -3/2
# #   put element into an array
# # return (array * amp) - ofs
# plot1d([indices, indices], [
#     diff_list,
#     values['amp'] * np.array([(float(ind) - values['shift']) ** (-3 / 2) for ind in np.add(indices, 1)]) - values[
#         'ofs']])
#
# # AVERAGING DATA
# # # get_ipython().run_line_magic('time', 'averaged = sgm_data.mean()')
# # # averaged = sgm_data.mean()
# # # print(averaged)
# # # averaged['ZP_CaBM_4Hr_R1-O'][0]