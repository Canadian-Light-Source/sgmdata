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

# ip = get_ipython()
# # ip.run_line_magic('pip', 'install lmfit')


# GETTING DATA FILES FROM DISK * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


l = []

# desired_file = input("Pleas input the absolute path to your hdf5 file: ")
# for filename in glob.iglob(desired_file, recursive=True):
#     l.append(filename)
# if len(l) < 1:
#     print("There were no files matching your description found in the specified directory.\n")
# else:
#     print("The following files match your input: " + str(l))

# Getting all files (intended to be hdf5 files) that match a specific pattern. Appending them to l.
for filename in glob.iglob('/Users/roseh/Desktop/Internship/MyCode/h5Files/*nitrate*.hdf5', recursive=True):
    l.append(filename)
print(l)


# LOADING DATA * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

sgm_data = sgmdata.load.SGMData(l, sample = "Co-nitrate - N")

print(sgm_data._repr_console_())

interp_list = sgm_data.interpolate(resolution=0.1)


# FUNCTIONS * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

def plot1d(xarr, yarr):
    """
    Sets the specifications for a graph, then shows graph. Graph represents the data samples, and the predicted data
    samples.
    Variables:
        xarr: a list of two lists containing the x values for the points we will be plotting.
        yarr: a list of two lists containing the y values for the points we will be plotting.
    """
    TOOLS = 'pan, hover, box_zoom, box_select, crosshair, reset, save'
    fig = figure(
        tools=TOOLS,
        title="Plot",
        background_fill_color="white",
        background_fill_alpha=1,
        x_axis_label="x",
        y_axis_label="y",
    )
    colors = []
    for i in range(np.floor(len(yarr) / 6).astype(int) + 1):
        colors += ['purple', 'yellow', 'black', 'firebrick', 'red', 'orange']
    colors = iter(colors)
    if not isinstance(xarr, list):
        xarr = [xarr]
    if not len(xarr) == len(yarr):
        yarr = [yarr]
    for i, x in enumerate(xarr):
        fig.circle(x=x, y=yarr[i], color=next(colors), legend_label="Curve" + str(i))
    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"
    show(fig)


def lowest_variance(d_list):
    """
    From a list, calculates the lowest level of variance between 5 of the consecutive values in the list. Returns this
    level of variance and the index of the values that make up this level.
    Variables:
        d_list: the list of variances between each consecutive spot on a graph.
    """
    pos = 0
    recent_differences = []
    differences = []
    recent_vars = []
    for diff in d_list:
        if len(recent_vars) < 4:
            recent_vars.append(diff)
        elif len(recent_vars) == 4:
            recent_differences.clear()
            pos = 4
            recent_vars.append(diff)
            for var in recent_vars:
                recent_differences.append(((var - np.mean(recent_vars)) ** 2))
            differences.append(np.sum(recent_differences) / len(recent_differences))
        else:
            recent_differences.clear()
            pos = pos + 1
            recent_vars.pop(0)
            recent_vars.append(diff)
            for var in recent_vars:
                recent_differences.append(((var - np.mean(recent_vars)) ** 2))
            differences.append(np.sum(recent_differences) / len(recent_differences))
    for diff in differences:
        print(str(differences.index(diff)) + " - " + str(differences.index(diff) + 4) + " is: " + str(diff))
    return "The lowest average variance (of the variance between 2 consecutive variances) between 5 consecutive " \
           "variances is " + str(min(differences)) + ".\nIt is reached with the variances between the " \
            "values within the range: " + str(pos - 4) + " through to position: " + str(pos) + ".\n"


def noise(idx, amp, shift, ofs):
    """
    Purpose: The equation to calculate the predicted level of noise in a scan.
    Variables:
        idx: Independent variable. An array of the indices of the values in sdd_list that were previously deemed to be
        acceptable sdd values.
        amp: Dependant variable. How much to amplify the noise level by.
        shift: Dependant variable. Amount to shift the index by.
        ofs: Dependant variable.
    """
    return amp * (idx + shift) ** (-3 / 2) + ofs


def predict(d_list, cur_indices):
    """
    Takes predicted noise levels from d_list and inputs them to the noise function to predict the level of noise
    in the next scan.
    Variables:
        d_list: a list of the differences between our first ten scans. As our determine_scan_num function continues
        it will add values to the d_list.
        num_indices: a list of the x values of the first 10 scans (1, 2, 3, etc.). As our determine_scan_num function
        progresses it will add more x values.
    Return Value: the values for amp, shift and ofs that fit num_indices and diff_list best.
    """
    noisemodel = Model(noise)
    params = noisemodel.make_params()
    params['amp'].set(value=d_list[0], vary=True, min=0.1 * d_list[0], max=2 * d_list[0])
    params['shift'].set(value=0.1, vary=True, min=-2, max=2)
    params['ofs'].set(value=0, vary=True, min=-1, max=0)
    result_noise = noisemodel.fit(d_list, idx=np.array(cur_indices), params=params)
    chosen_vals = result_noise.best_values
    return (chosen_vals['amp'] * np.array([(float(ind) - chosen_vals['shift']) ** (-3/2) for ind in
                                         np.add(cur_indices, 15)]) - chosen_vals['ofs'])


def determine_num_scans(d_list, indices, desired_difference=0.17961943):
    """
    Takes a string of variations in noise levels and a desired noise level, and returns the number of additional
    variations (ie, scans) required to reach the desired noise level.
    Variables:
        d_list: a list of the differences between our first ten acceptable scans. As our determine_scan_num function
        continues it will add values to the d_list.
        indices: the scan numbers of the first ten acceptable scans
        desired_noise: the variance between 5 consecutive scans the user would like to achieve, ie, we'll need to
        continue scans until this level of variance is reached. Default is the highest variance among sample scans.
    Return value: the number of scans required to reach the desired level of variance.
    """
    copied_indices = indices.copy()
    num_predictions = 9
    keep_predicting = True
    recent_differences = d_list[5:]
    variances = []
    recent_vars = []
    # Checking if the desired level of variance has already been reached in the first 10 scans.
    for element in d_list:
        if len(recent_differences) < 4:
            recent_differences.append(element)
        elif len(recent_differences) == 4:
            recent_differences.append(element)
            for var in recent_differences:
                recent_vars.append(((var - np.mean(recent_differences)) ** 2))
            variances.append(np.sum(recent_vars) / len(recent_vars))
            if variances[-1] <= desired_difference:
                return "Desired level of variance already reached within first 10 scans. No additional scans needed."
        else:
            recent_vars.clear()
            recent_differences.pop(0)
            recent_differences.append(element)
            for var in recent_differences:
                recent_vars.append(((var - np.mean(recent_differences)) ** 2))
            variances.append(np.sum(recent_vars) / len(recent_vars))
            if variances[-1] <= desired_difference:
                return "Desired level of variance already reached within first 10 scans. No additional scans needed."
    # Predicting variances until we predict a variance that's at or below the desired level of variance.
    while keep_predicting:
        # Predicting the next difference.
        predicted_level = predict(d_list, copied_indices)
        copied_indices.append(int(copied_indices[-1]) + 1)
        num_predictions = num_predictions + 1
        # Adding the newly predicted differance to our list of variances.
        d_list = np.append(d_list, predicted_level[-1])
        # Calculating the variance of the newest differences.
        recent_vars.clear()
        recent_differences.pop(0)
        recent_differences.append(predicted_level[-1])
        for var in recent_differences:
            recent_vars.append(((var - np.mean(recent_differences)) ** 2))
        variances.append((np.sum(recent_vars)) / len(recent_vars))
        # Stopping the predicting of indices if the desired level of variance has already been reached.
        if variances[-1] <= desired_difference:
            keep_predicting = False
    ###
    print("d_list:\n" + str(d_list))
    for vari in variances:
        print(str(vari))
    ###
    # Adding plus one because, since we're working with variances, and a variance comes from 2 values,
    # to get any number of variances you'll need that number of variances plus 1 scans
    return num_predictions + 1


# INTERPOLATING DATA * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


def interpolating_data(interp_list_param):
    """Takes the data returned from interpolate and collects the sdd values. Sorts through these sdd values and removes
     unfit values. Keeps a separate list of the indices of the fit values. Deals with nan and infinity values in the
     list of sdd values and returns it to caller as a numpy array."""
    sdd_list = []
    for df in interp_list_param:
        sdd_list.append(df.filter(regex=("sdd.*"), axis=1).to_numpy())
    prev_mean = sdd_list[0]
    avg_list = [sdd_list[0]]
    diff_list = []
    indices = []
    for i, arr in enumerate(sdd_list[1:]):
        avg_list.append(arr)
        cur_mean = np.mean(avg_list, axis=0)
        diff_list.append(np.var(np.subtract(cur_mean, prev_mean)))
        if len(diff_list) > 2:
            if diff_list[-2] - diff_list[-1] < -50:
                avg_list = avg_list[:-1]
                diff_list = diff_list[:-1]
            else:
                indices.append(i)
                prev_mean = cur_mean
        else:
            indices.append(i)
            prev_mean = cur_mean
    diff_list = np.nan_to_num(np.array(diff_list))
    return diff_list, indices


# * * * * * * * * * * * * * * CALLING/ TESTING ZONE * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


returned_data = interpolating_data(interp_list)
returned_indices = returned_data[1]
returned_diff_list = returned_data[0]
print("\ndiff_list: \n" + str(returned_diff_list))
# print("\nindices: \n" + str(returned_indices))


returned_diff_list_listed = []
for item in returned_diff_list:
    returned_diff_list_listed.append(item)
returned_indices_listed = []
for item in returned_indices:
    returned_indices_listed.append(item)


# lowest_variance(returned_diff_list_listed)
print("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ")
print(determine_num_scans(returned_diff_list_listed[:10], returned_indices_listed[:10], 5))

