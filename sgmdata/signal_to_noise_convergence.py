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
from sgmdata.load import SGMData
# from sgmdata.utilities.util import plot1d


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
        raise ValueError("There are no files that match the given pattern.")
    return list_of_files


def loading_data(list_of_files):
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
    if len(sgm_data.__dict__['scans']) == 0:
        raise ValueError("hdf5 file must contain scans to be able to predict the number of scans required. The hdf5 "
                         "file you have provided does not contain any scans. PLease try again with an hdf5 file that"
                         " does contain scans.")
    has_sdd = False
    for element in sgm_data.__dict__['scans']:
        i = 0
        signals = list(sgm_data.__dict__['scans'][element].__getitem__('entry1').__getattr__('signals').keys())
        while i < len(signals) and not has_sdd:
            if "sdd" in signals[i]:
                has_sdd = True
            else:
                i += 1
        if not has_sdd:
            raise ValueError("Scans must have sdd values to be able to predict the number of scans required. One or "
                             "more of the scans you have provided do not have sdd values. Please try again using "
                             "scans with sdd values. ")

    # print(sgm_data._repr_console_())
    interp_list = sgm_data.interpolate(resolution=0.1)#, sample='Co-nitrate - N')
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
    if len(d_list) < 5:
        raise ValueError("lowest_variance function can only be used on scan sets with 6 or more scans.")
    recent_variances = []
    variances = []
    recent_vals = []
    for diff in d_list:
        if len(recent_vals) < 4:
            recent_vals.append(diff)
        # elif len(recent_vals) == 4:
        #     recent_vals.append(diff)
        #     for var in recent_vals:
        #         recent_variances.append(((var - np.mean(recent_vals)) ** 2))
        #     variances.append(np.sum(recent_variances) / len(recent_variances))
        # else:
        #     recent_variances.clear()
        #     recent_vals.pop(0)
        #     recent_vals.append(diff)
        #     for var in recent_vals:
        #         recent_variances.append(((var - np.mean(recent_vals)) ** 2))
        #     variances.append(np.sum(recent_variances) / len(recent_variances))
        elif len(recent_vals) == 4:
            recent_vals.append(diff)
            for var in recent_vals:
                recent_variances.append(((var - np.mean(recent_vals)) ** 2))
            variances.append(np.sum(recent_variances) / len(recent_variances))
            print(variances[-1])
            if variances[-1] <= .1796195:
                pos = (variances.index(variances[-1]))
                return ".1796 reached by scans:\t" + str(pos) + " - " + str(pos + 5) + "\nvalue is:\t" + \
                       str(variances[-1])
        else:
            recent_variances.clear()
            recent_vals.pop(0)
            recent_vals.append(diff)
            for var in recent_vals:
                recent_variances.append(((var - np.mean(recent_vals)) ** 2))
            variances.append(np.sum(recent_variances) / len(recent_variances))

            print(variances[-1])
            if variances[-1] <= .1796195:
                pos = (variances.index(variances[-1]))
                return ".1796 reached by scans:\t" + str(pos) + " - " + str(pos + 5) + "\nvalue is:\t" + \
                       str(variances[-1])
    # pos = (variances.index(min(variances)) + 4)
    # return "\tThe lowest average variance (of the variance between 2 consecutive variances) between 5 consecutive " \
    #        "variances is " + str(min(variances)) + ".\nIt is reached with the variances between the " \
    #         "values within the range: " + str(pos - 5) + " through to position: " + str(pos) + ".\n"


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
        d_list(list of floats): a list of the differences between our first ten scans. As our determine_scan_num
        function continues it will add values to the d_list.
        num_indices(list of ints): a list of the x values of the first 10 scans (1, 2, 3, etc.). As our
        determine_scan_num function progresses it will add more x values.
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
    recent_differences = d_list[:5]
    variances = []
    recent_variances = []
    if len(d_list) < 9:
        raise ValueError("Insufficient number of scans. Prediction can only be made on scan sets with 10 or more "
                         "scans.")
    # Checking if the desired level of variance has already been reached in the first 10 scans.
    for element in d_list:
        if len(recent_differences) < 4:
            recent_differences.append(element)
        elif len(recent_differences) == 4:
            recent_differences.append(element)
            for var in recent_differences:
                recent_variances.append(((var - np.mean(recent_differences)) ** 2))
            if (np.sum(recent_variances) / len(recent_variances)) <= desired_difference:
                # return 0
                return d_list, copied_indices
        else:
            recent_variances.clear()
            recent_differences.pop(0)
            recent_differences.append(element)
            for var in recent_differences:
                recent_variances.append(((var - np.mean(recent_differences)) ** 2))
            if (np.sum(recent_variances) / len(recent_variances)) <= desired_difference:
                # return 0
                return d_list, copied_indices
    # Predicting variances until we predict a variance that's at or below the desired level of variance.
    while keep_predicting:
        # Predicting the next difference.
        predicted_level = predict(d_list, copied_indices)
        copied_indices.append(int(copied_indices[-1]) + 1)
        num_predictions = num_predictions + 1
        # Adding the newly predicted differance to our list of variances.
        d_list = np.append(d_list, predicted_level[-1])
        # Calculating the variance of the newest differences.
        recent_variances.clear()
        recent_differences.pop(0)
        recent_differences.append(predicted_level[-1])
        for var in recent_differences:
            recent_variances.append(((var - np.mean(recent_differences)) ** 2))
        variances.append((np.sum(recent_variances)) / len(recent_variances))
        # Stopping the predictions if the desired level of variance has already been reached.
        if variances[-1] <= desired_difference:
            keep_predicting = False
        if num_predictions > 100:
            # If the desired_difference was the default value, not input by user.
            if desired_difference == 0.17961943:
                raise RuntimeError("Sufficiently accurate prediction cannot be made.")
            else:
                raise ValueError("Desired level of variance cannot be reached.")
    # # # for plot1D:
    return d_list, copied_indices
    # # #
    # return num_predictions


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
            # if diff_list[-2] - diff_list[-1] < -50:
            #     avg_list = avg_list[:-1]
            #     diff_list = diff_list[:-1]
            # else:
            indices.append(i)
            prev_mean = cur_mean
        else:
            indices.append(i)
            prev_mean = cur_mean
    diff_list = np.nan_to_num(np.array(diff_list))
    return diff_list, indices


def plot1d(xarr, yarr, title="Plot", labels=[]):
    TOOLS = 'pan, hover,box_zoom,box_select,crosshair,reset,save'
    fig = figure(
            tools=TOOLS,
            title=title,
            background_fill_color="white",
            background_fill_alpha=1,
            x_axis_label = "x",
            y_axis_label = "y",
    )
    colors = []
    for i in range(np.floor(len(yarr)/6).astype(int)+1):
        colors += ['purple', 'orange', 'firebrick', 'red', 'navy', 'black']
    colors = iter(colors)
    if not isinstance(xarr, list):
        xarr = [xarr]
    if not len(xarr) == len(yarr):
        yarr = [yarr]
    for i,x in enumerate(xarr):
        fig.circle(x=x, y=yarr[i], color=next(colors), legend_label=labels[i])
    fig.legend.location = "top_left"
    fig.legend.click_policy="hide"
    show(fig)


def run_all(files):
    """
    Purpose: Puts together function calls to previously created functions so that desired results can be received from
    running only this function. Returns the number of additional scans that should be run on a sample.
    Returns:
        determine_num_scans(returned_diff_list_listed[:10], returned_indices_listed[:10], 5)(int): the number of scans
        required to reach the desired level of variance.
    """
    # Running functions to set up data.
    list_of_files = file_retrieval(files)
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
    # print("\tActual:")
    # print(lowest_variance(returned_diff_list_listed))

    # print(lowest_variance(returned_diff_list_listed[:10]))
    # i = 0
    # while i < len(returned_diff_list_listed):
    #     print(returned_diff_list_listed[i])
    #     i += 1
    # results = determine_num_scans(returned_diff_list_listed[:10], returned_indices_listed[:10], 0.6716435)
    # a = 0
    # print("\tPredicted:")
    # while a < len(results[0]):
    #     print(results[0][a])
    #     a += 1
    # predicted = results[0]
    # indices = results[1]
    ###
    # returned_diff_list_listed.append(350)
    # returned_indices_listed.append(returned_diff_list_listed[-1] + 1)
    # plot1d([returned_indices_listed], [returned_diff_list_listed],
    #        "Sample Fe2O3", ["Actual Values", "Predicted Value"])
    # plot1d([returned_indices_listed[:10], indices[10:]], [returned_diff_list_listed[:10], predicted[10:]],
    #        "Sample Medge", ["Actual Values", "Predicted Value"])
    return len(determine_num_scans(returned_diff_list_listed[:10], returned_indices_listed[:10])[0]) - 1


def testing():
    # lowest_variance()
    func = "lowest_variance()"
    # lowest_variance(): testing on a regular list.
    situ = "regular list"
    expected_result = 'The lowest average variance (of the variance between 2 consecutive variances) between 5 ' \
                      'consecutive variances is 2.0.\nIt is reached with the variances between the values within the ' \
                      'range: 10 through to position: 14.\n'
    test_list = [0, 5, 10, 15, 20, 10, 20, 30, 40, 50, 1, 2, 3, 4, 5]
    result = lowest_variance(test_list)
    if str(result) != expected_result:
        print("Error with " + str(func) + " when using on " + str(situ) + ". Expected " + str(expected_result) +
              " but got " + str(result))
    # lowest_variance(): testing on a list of all 0s.
    situ = "list of all 0s"
    expected_result = 'The lowest average variance (of the variance between 2 consecutive variances) between 5 ' \
                      'consecutive variances is 0.0.\nIt is reached with the variances between the values within the ' \
                      'range: 7 through to position: 11.\n'
    test_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    result = lowest_variance(test_list)
    if str(result) != expected_result:
        print("Error with " + str(func) + " when using on " + str(situ) + ". Expected " + str(expected_result) +
              " but got " + str(result))
    # lowest_variance(): testing on a list containing ints and floats.
    situ = "list containing ints and floats "
    expected_result = 'The lowest average variance (of the variance between 2 consecutive variances) between 5' \
                      ' consecutive variances is 0.01999999999999999.\nIt is reached with the variances between the' \
                      ' values within the range: 10 through to position: 14.\n'
    test_list = [.5, .1, 1.5, 2.0, 2.5, 1, 2, 3, 4, 5, 1.1, 1.2, 1.3, 1.4, 1.5]
    result = lowest_variance(test_list)
    if str(result) != expected_result:
        print("Error with " + str(func) + " when using on " + str(situ) + ". Expected " + str(expected_result) +
              " but got " + str(result))

    # predict()
    func = "predict()"
    # predict(): regular lists passed for both variables
    situ = "regular lists passed for both variables"
    expected_result = np.array([12.28664511, 11.19761088, 10.26045971, 9.44710632, 8.73580685, 8.10949264, 7.55459775,
                                7.06021877, 6.61750288, 6.21919547, 5.85930105, 5.53282617])
    test_list1 = [500, 200, 100, 75, 65, 60, 56, 52, 50, 49, 48, 47]
    test_list2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    result = predict(test_list1, test_list2)
    same = True
    i = 0
    while i < len(expected_result):
        if round(expected_result[i], 5) != round(result[i], 5):
            same = False
        i += 1
    if not same:
        print("Error with " + str(func) + " when using on " + str(situ) + ". Expected\n" + str(expected_result) +
              " but got \n" + str(result))
    # predict(): list of both ints and floats for d_list
    situ = "list of both ints and floats for d_list"
    expected_result = np.array([4.547589, 4.1427363, 3.79458128, 3.49259853, 3.22864805, 2.99634627, 2.79062404, 2.60741038,
                               2.44340228, 2.29589481, 2.16265387, 2.04181978, 1.9318333])
    test_list1 = [140, 120, 100, 90, 80, 75, 70, 67.5, 65, 63.5, 62, 61.5, 60]
    test_list2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    result = predict(test_list1, test_list2)
    same = True
    i = 0
    while i < len(expected_result):
        if round(expected_result[i], 5) != round(result[i], 5):
            same = False
        i += 1
    if not same:
        print("Error with " + str(func) + " when using on " + str(situ) + ". Expected\n" + str(expected_result) +
              " but got \n" + str(result))
    # predict(): random list for indices
    situ = "list of both ints and floats for d_list"
    expected_result = np.array([0.30573543, 0.14051718, 0.01636123, 0.23149906, 0.02215948, 0.27704, 0.19719866, 0.0404422,
                                0.01406676, 0.21320443, 0.10660011, 0.00481277, 0.10660011])
    test_list1 = [140, 120, 100, 90, 80, 75, 70, 67.5, 65, 63.5, 62, 61.5, 60]
    test_list2 = [0, 10, 89, 3, 70, 1, 5, 42, 100, 4, 15, 220, 15]
    result = predict(test_list1, test_list2)
    same = True
    i = 0
    while i < len(expected_result):
        if round(expected_result[i], 5) != round(result[i], 5):
            same = False
        i += 1
    if not same:
        print("Error with " + str(func) + " when using on " + str(situ) + ". Expected\n" + str(expected_result) +
              " but got \n" + str(result))

    # determine_num_scans():
    func = "determine_num_scans()"
    # determine_num_scans(): regular values input for d_list and indices.
    situ = "regular values input for d_list and indices"
    expected_result = 15
    test_list1 = [140, 120, 100, 90, 80, 75, 70, 67.5, 65, 63.5]
    test_list2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    result = determine_num_scans(test_list1, test_list2)
    if result != expected_result:
        print("Error with " + str(func) + " when using on " + str(situ) + ". Expected " + str(expected_result) +
              " but got " + str(result))
    # determine_num_scans(): d_list too small.
    situ = "d_list too small"
    expected_result = 1
    test_list1 = [140, 120, 100]
    test_list2 = [0, 1, 2]
    result = determine_num_scans(test_list1, test_list2)
    if result != expected_result:
        print("Error with " + str(func) + " when using on " + str(situ) + ". Expected " + str(expected_result) +
              " but got " + str(result))
    # determine_num_scans(): d_list where desired level of noise already reached.
    situ = "desired level of noise already reached"
    expected_result = 0
    test_list1 = [100, 200, 0.1, 0.11, 0.111, 0.1111, 0.11111, 300, 500, 20]
    test_list2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    result = determine_num_scans(test_list1, test_list2)
    if result != expected_result:
        print("Error with " + str(func) + " when using on " + str(situ) + ". Expected " + str(expected_result) +
              " but got " + str(result))
    # determine_num_scans(): normal d_list, random ints in indices.
    situ = "normal d_list, random ints in indices"
    expected_result = 15
    test_list1 = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    test_list2 = [0, 10, 4, 88, 90, 4, 67, 12, 100, 43]
    result = determine_num_scans(test_list1, test_list2)
    if result != expected_result:
        print("Error with " + str(func) + " when using on " + str(situ) + ". Expected " + str(expected_result) +
              " but got " + str(result))
    # determine_num_scans(): very low desired noise level.
    situ = "very low desired noise level"
    expected_result = 583
    test_list1 = [140, 120, 100, 90, 80, 75, 70, 67.5, 65, 63.5]
    test_list2 = [0, 10, 4, 88, 90, 4, 67, 12, 100, 43]
    result = determine_num_scans(test_list1, test_list2, 0.00001)
    if result != expected_result:
        print("Error with " + str(func) + " when using on " + str(situ) + ". Expected " + str(expected_result) +
              " but got " + str(result))


# testing()
print("\tNumber of additional scans needed:\t" +
      str(run_all('C:/Users/roseh/Desktop/Internship/MyCode/h5Files/NotPrimary/*S2-7*.hdf5')))
# print("\tNumber of additional scans needed:\t" +
      # str(run_all('C:/Users/roseh/Desktop/Internship/MyCode/h5Files/*Amm*.hdf5')))



