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
# from ..signal_to_noise_convergence import sgmdata
import sgmdata
# print(globals())
from sgmdata.load import SGMData
# from
# print(os.dirname())


def file_retrieval():
    """
    Purpose: Gets files that match a specified pattern from a specified directory.
    Returns:
        list_of_files(list): a list of the files that match the given pattern.
    """
    list_of_files = []
    for filename in glob.iglob('/Users/roseh/Desktop/Internship/MyCode/h5Files/*nitrate*.hdf5', recursive=True):
        list_of_files.append(filename)
    return list_of_files


def loading_data(list_of_files):
    """
    Purpose: Create an SGMData object from hdf5 files. Interpolate the data in the SGMData object. Return this interpolated data.
    Parameters:
        list_of_files(list of str): the names of the hdf5 files to create an SGMData object out of.
    Returns:
        interp_list(list): a list of pandas dataframes. Contains the interpolated version of the data in the files
        specified in list_of_files.
    """
    sgm_data = sgmdata.load.SGMData(list_of_files, sample = "Co-nitrate - N")
    # print(sgm_data._repr_console_())
    interp_list = sgm_data.interpolate(resolution=0.1)
    print("interp_list\t" + str(type(interp_list)))
    i = 0
    print("interp_list items:\t" + str(type(interp_list[0])))
    return interp_list


def lowest_variance(d_list):
    """
    Purpose: Takes a list of differences between values. Calculates the lowest variance between 5 of these consecutive
    differences and keeps track of the positions of these 5 values in d_list.
    Parameters:
        d_list(list of float): a list of the differences between each 2 consecutive values.
    Returns:
        (str): the lowest variance value and the index of the list differences that make up this variance value.
    """
    pos = 0
    recent_variances = []
    variances = []
    recent_vars = []
    for diff in d_list:
        if len(recent_vars) < 4:
            recent_vars.append(diff)
        elif len(recent_vars) == 4:
            pos = 4
            recent_vars.append(diff)
            for var in recent_vars:
                recent_variances.append(((var - np.mean(recent_vars)) ** 2))
            variances.append(np.sum(recent_variances) / len(recent_variances))
        else:
            recent_variances.clear()
            pos = pos + 1
            recent_vars.pop(0)
            recent_vars.append(diff)
            for var in recent_vars:
                recent_variances.append(((var - np.mean(recent_vars)) ** 2))
            variances.append(np.sum(recent_variances) / len(recent_variances))
    return "The lowest average variance (of the variance between 2 consecutive variances) between 5 consecutive " \
           "variances is " + str(min(variances)) + ".\nIt is reached with the variances between the " \
            "values within the range: " + str(pos - 4) + " through to position: " + str(pos) + ".\n"


def noise(idx, amp, shift, ofs):
    """
    Purpose: The equation to calculate the predicted level of noise in a scan.
    Parameters:
        idx(np.array): Independent variable. An array of the indices of the values in sdd_list that were previously deemed to be
        acceptable sdd values.
        amp(float): Dependant variable. How much to amplify the noise level by.
        shift(float): Dependant variable. Amount to shift the index by.
        ofs(float): Dependant variable. How much to offset the noise level by.
    Returns:
         (np.array): the values in idx fit to the curve outlined by amp, shift and ofs.
    """
    return amp * (idx + shift) ** (-3 / 2) + ofs


def predict(d_list, cur_indices):
    """
    Purpose: Takes predicted differences from d_list and inputs them to the noise function. This predicts the level of
    noise in the next scan.
    Variables:
        d_list(list of floats): a list of the differences between our first ten scans. As our determine_scan_num function continues
        it will add values to the d_list.
        num_indices(list of ints): a list of the x values of the first 10 scans (1, 2, 3, etc.). As our determine_scan_num function
        progresses it will add more x values.
    Returns:
        (np.array): an array of the differences between consecutive values.
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
# result_noise is the predicted noise level (I think?) Does the equation change it back to a difference?


def determine_num_scans(d_list, indices, desired_difference=0.17961943):
    """
    Purpose: takes a string of variations in noise levels and a desired noise level, and returns the number of additional
    variations (ie, scans) required to reach the desired noise level.
    Variables:
        d_list(list of floats): a list of the differences between our first ten acceptable scans. As our
        determine_scan_num function continues it will add values to the d_list.
        indices(list of ints): the scan numbers of the scans used to generate d_list.
        desired_noise(float): the variance between 5 consecutive scans the user would like to achieve, ie, we'll need to
        continue scans until this level of variance is reached. Default is the highest variance among sample scans.
    Returns:
         num_predictions+1(int): the number of scans required to reach the desired level of variance. Adding one
         because, since we're working differences in d_list, and a difference comes from 2 values, to get any number of
         differences you'll need that number of scans plus 1.
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
    return num_predictions + 1


def interpolating_data(interp_list_param):
    """
    Purpose: Takes the results returned by SGMData's "interpolate" function and collects the sdd values within it. Sorts
    through these sdd values and removes unfit values. Keeps a separate list of the indices of the fit values. Deals
    with nan and infinity values in the list of sdd values. Returns the modified list of sdd values and the list
    of the indices of fit valuer to caller as a numpy arrays.
    Variables:
        interp_list_param(list of pandas dataframes): the list of results returned by SGMData's "interpolate" function.
    Returns:
        diff_list(list of floats): a list of the differences between our first ten acceptable scans. Used to formulate
        the model to predict the differences between future scans.
        indices(list of ints): The numbers of the scans used to formulate diff_list.
        """
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


def run_all():
    """
    Purpose: Puts together function calls to previously created functions so that desired results can be received from
    running only this function. Returns the number of additional scans that should be run on a sample.
    Returns:
        determine_num_scans(returned_diff_list_listed[:10], returned_indices_listed[:10], 5)(int): the number of scans
        required to reach the desired level of variance.
    """
    # Running functions to set up data.
    list_of_files = file_retrieval()
    interp_list = loading_data(list_of_files)
    returned_data = interpolating_data(interp_list)
    # Organizing data returned from functions to set up data.
    returned_indices = returned_data[1]
    returned_diff_list = returned_data[0]
    returned_diff_list_listed = []
    for item in returned_diff_list:
        returned_diff_list_listed.append(item)
    returned_indices_listed = []
    for item in returned_indices:
        returned_indices_listed.append(item)
    # Running functions to manipulate data.
    return determine_num_scans(returned_diff_list_listed[:10], returned_indices_listed[:10], 5)


print(run_all())

