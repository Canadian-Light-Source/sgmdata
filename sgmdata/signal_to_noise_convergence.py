# General imports
import glob
from IPython.testing.globalipapp import get_ipython
import asyncio
import h5py
import pandas
from dask.distributed import Client
from distributed import Client
import numpy as np
import os
import sys
import inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
import sgmdata
from lmfit import Model
from sgmdata.load import SGMData
# Plotting function imports
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import HoverTool, CustomJS, BooleanFilter, LinearColorMapper, LogColorMapper, ColorBar
from bokeh.io import show

ip = get_ipython()
# # ip.run_line_magic('pip', 'install lmfit')


# GETTING DATA FILES FROM DISK * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


l = []

# desired_file = input("Pleas input the absolute path to your hdf5 file: ")
# for filename in glob.iglob(desired_file, recursive=True):
#     l.append(filename)
# if l.isempty():
#     print("There were no files matching your description found in the specified directory.\n")
# else:
#     print("The following files match your input: " + str(l))

# Getting all files (intended to be hdf5 files) that match a specific pattern. Appending them to l.
for filename in glob.iglob('/Users/roseh/Desktop/Internship/MyCode/h5Files/*nitrate*.hdf5', recursive=True):
    l.append(filename)
print(l)


# LOADING DATA * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


# Reports amount of time required to execute command.
ip.run_line_magic('time', 'sgm_data = SGMData(l, sample="Feb21-32 - C")')
# Creates a new SGMData object.
sgm_data = sgmdata.load.SGMData(l, sample='Co-nitrate - N')

# Reports amount of time required to execute command.
ip.run_line_magic('time', 'sgm_data')
# Displays data stored in our sgm_data object on the console.
# print(sgm_data._repr_console_())

# Reports amount of time required to execute command.
ip.run_line_magic('time', 'interp_list = sgm_data.interpolate(resolution=0.1)')
# Interpolates data in our sgm_data object. Puts interpolated data into interp_list.
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
    # A string listing the tools we will have available for our graph.
    TOOLS = 'pan, hover, box_zoom, box_select, crosshair, reset, save'
    # Specifying the appearance of our graph.
    fig = figure(
        tools=TOOLS,
        title="Plot",
        background_fill_color="white",
        background_fill_alpha=1,
        x_axis_label="x",
        y_axis_label="y",
    )
    colors = []
    # For every group of six in yarr (rounded down) and once more, add " 'purple', 'black', 'yellow', 'firebrick',
    # 'red', 'orange' " to the 'colors' list.
    for i in range(np.floor(len(yarr) / 6).astype(int) + 1):
        colors += ['purple', 'yellow', 'black', 'firebrick', 'red', 'orange']
    # Making colors variable into an iterator that can iterate through the previous version of colors.
    colors = iter(colors)
    if not isinstance(xarr, list):
        xarr = [xarr]
    if not len(xarr) == len(yarr):
        yarr = [yarr]
    # For the number of items in xarr, plot a new point on our graph.
    for i, x in enumerate(xarr):
        fig.circle(x=x, y=yarr[i], color=next(colors), legend_label="Curve" + str(i))
    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"
    show(fig)


def examine_graph_data(xarr, yarr):
    """
    Takes a set of x and y values, and prints out various information about the y values. Used for testing.
    Variables:
        xarr: x values. A list of two lists containing the varience between each consecutive spot on a graph.
        yarr: y values. The y offset of the consecutive spots.
    """
    pred = []
    actu = []
    current_list = actu
    for element in enumerate(xarr):
        for val in element:
            current_list.append(val)
        current_list = pred
    i = 0
    varienceL = 0
    while i < len(actu):
        print("\n\n" + str(i) + ". Predicted:\t\t\t\t Actual:\t\t\t\t Varience:\t\t\t\t Deviation %:\n"
              + str(pred[i]) + "\t\t " + str(actu[i]) + "\t\t " + str(pred[i] - actu[i]) + "\t\t " +
              str(((pred[i] / actu[i]) - 1) * 100))
        if str(pred[i] / actu[i]) != 'nan':
            varienceL += pred[i] / actu[i]
        i += 1
    print(((varienceL / i) - 1) * 100)


def lowest_variance(d_list):
    """
    From a list, calculates the lowest level of variance between 5 of the consecutive values in the list. Returns this
    level of variance and the index of the values that make up this level.
    Variables:
        d_list: the list of variances between each consecutive spot on a graph.
    """
    print("\nd_list: \n" + str(d_list))
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
    # print("difference: \n" + str(differences))
    i = 0
    for boop in differences:
        print(str(i) + "-" + str(i + 4) + " difference is: " + str(boop))
        i += 1
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
    # Fitting of parametrized noise reduction equation
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
    # Using lmfit to prepare the execute noise, initially without parameters.
    noisemodel = Model(noise)
    # Creates the parameters for our 'noisemodel' calling of the 'noise' function, but does not assign values to them.
    params = noisemodel.make_params()
    # Setting the values of 'amp', 'shift' and 'ofs' for our 'noisemodel' calling of the 'noise' function.
    params['amp'].set(value=d_list[0], vary=True, min=0.1 * d_list[0], max=2 * d_list[0])
    params['shift'].set(value=0.1, vary=True, min=-2, max=2)
    params['ofs'].set(value=0, vary=True, min=-1, max=0)
    # Fits our 'noisemodel' calling of the 'noise' function with data set to diff_list and idx set to an arrayed
    # version of indices and the same parameters we previously set.
    result_noise = noisemodel.fit(d_list, idx=np.array(cur_indices), params=params)
    chosen_vals = result_noise.best_values
    # print("\nd_list:\n " + str(d_list) + "\n\nPredicted values: ")
    return (chosen_vals['amp'] * np.array([(float(ind) - chosen_vals['shift']) ** (-3/2) for ind in
                                         np.add(cur_indices, 15)]) - chosen_vals['ofs'])


def determine_num_scans(d_list, desired_difference=0.6400111799776529):
    """
    Takes a string of variations in noise levels and a desired noise level, and returns the number of additional
    variations (ie, scans) required to reach the desired noise level.
    Variables:
        d_list: a list of the differences between our first ten scans. As our determine_scan_num function continues
        it will add values to the d_list.
        desired_noise: the variance between 5 consecutive scans the user would like to achieve, ie, we'll need to
        continue scans until this level of variance is reached. Default is the highest variance among sample scans.
    Return value: the number of scans required to reach the desired level of variance.
    """
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
    # Predicting variances until we predict the desires level of variance.
    while keep_predicting:
        x_vals = []
        # Making a list of indices for the graph.
        # **** Should we import indices list here? Probably ****
        for num in range(num_predictions + 1):
            x_vals.append(num)
        # Predicting the next difference.
        predicted_level = predict(d_list, x_vals)
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
    # Adding plus one because, since we're working with variances, and a variance comes from 2 values,
    # to get any number of variances you'll need that number of variances plus 1 scans
    ###
    print("d_list: " + str(d_list) + "\ndifferences: " + str(variances))
    ###
    return num_predictions + 1


# INTERPOLATING DATA * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
def interpolating_data(interp_list_param):
    """Takes the data returned from interpolate and collects the sdd values. Sorts through these sdd values and removes
     unfit values. Keeps a separate list of the indices of the fit values. Deals with nan and infinity values in the
     list of sdd values and returns it to caller as a numpy array."""
    sdd_list = []
    # Checking for items in interp_list containing the characters 'sdd.' those items are appended to sdd_list.
    for df in interp_list_param:
        sdd_list.append(df.filter(regex=("sdd.*"), axis=1).to_numpy())
    prev_mean = sdd_list[0]
    avg_list = [sdd_list[0]]
    diff_list = []
    indices = []
    for i, arr in enumerate(sdd_list[1:]):
        # *Note that arr is the array of arrays representing the each scan's sdd values and i is the count of the number of
        # times the loop has been cycled through, starting at 0.
        avg_list.append(arr)
        cur_mean = np.mean(avg_list, axis=0)
        # * Cur_mean is an array with the mean of all values in the same positions in the in arrays within avg_list. Eg, the
        # average of the item [0] of each of the lists, then the average of the item[1] of each of the lists, etc.
        # Finding the variance between the last scan and the current scan.
        diff_list.append(np.var(np.subtract(cur_mean, prev_mean)))
        if len(diff_list) > 2:
            if diff_list[-2] - diff_list[-1] < -50:
                avg_list = avg_list[:-1]
                diff_list = diff_list[:-1]
                # If the variance between the difference between our current scan's mean and the mean of the scan before it
                # is 51 or more than the variance of the difference between the previous scan's mean and the mean of the
                # scan before it, then don't include the variance of the difference between the current scan's mean and the
                # previous scan's mean in diff list, or the average of this scan in avg_list.
            else:
                indices.append(i)
                prev_mean = cur_mean
        else:
            indices.append(i)
            prev_mean = cur_mean
    # Turns diff_list into and array, then converts NaN values to 0s and deals with infinity values.
    diff_list = np.nan_to_num(np.array(diff_list))
    return diff_list


# * * * * * * * * * * * * * * CALLING/ TESTING ZONE * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


# Below, diff_list stops at 8 because that'll give us 9 variances, and if we have 9 variances we'll have taken
# 10 scans to get them.

returned_diff_list = interpolating_data(interp_list)
# print(determine_num_scans([:10], 1.0))
# my_d_list = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2]
returned_diff_list_listed = []
for item in returned_diff_list:
    returned_diff_list_listed.append(item)

i = 0
for item in returned_diff_list_listed:
    print(str(i) + ". " + str(item))
    i += 1

print(lowest_variance(returned_diff_list_listed))
# print(determine_num_scans(returned_diff_list_listed[:10], 1.0))
# print(determine_num_scans(my_d_list[:10], 1.0))

# # Using lmfit to prepare the execute noise, initially without parameters.
# noisemodel = Model(noise)
# # Creates the parameters for our 'noisemodel' calling of the 'noise' function, but does not assign values to them.
# params = noisemodel.make_params()
# # Setting the values of 'amp', 'shift' and 'ofs' for our 'noisemodel' calling of the 'noise' function.
# params['amp'].set(value=returned_diff_list[0], vary=True, min=0.1 * returned_diff_list[0], max=2 * returned_diff_list[0])
# params['shift'].set(value=0.1, vary=True, min=-2, max=2)
# params['ofs'].set(value=0, vary=True, min=-1, max=0)
# # Fits our 'noisemodel' calling of the 'noise' function with data set to diff_list and idx set to an arrayed
# # version of indices and the same parameters we previously set.
# result_noise = noisemodel.fit(returned_diff_list, idx=np.array(cur_indices), params=params)
# chosen_vals = result_noise.best_values
#
# plot1d([returned_indices, returned_indices], [returned_diff_list,
#                                               (chosen_vals['amp'] * np.array(
#                                                   [(float(ind) - chosen_vals['shift']) ** (-3 / 2) for ind in
#                                                    np.add(cur_indices, 15)]) - chosen_vals['ofs'])
#                                               ])


# AVERAGING DATA
# # get_ipython().run_line_magic('time', 'averaged = sgm_data.mean()')
# # averaged = sgm_data.mean()
# # print(averaged)
# # averaged['ZP_CaBM_4Hr_R1-O'][0]