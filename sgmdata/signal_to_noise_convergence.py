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
    file = list(sgm_data.__dict__['scans'].keys())
    sample_name = list(sgm_data.__dict__['scans'][file[0]].__dict__.keys())
    signals = list(sgm_data.__dict__['scans'][file[0]].__getitem__(sample_name[0]).__getattr__('signals').keys())
    i = 0
    while i < len(signals) and not has_sdd:
        if "sdd" in signals[i]:
            has_sdd = True
        else:
            i += 1
    if not has_sdd:
        raise ValueError("Scans must have sdd values to be able to predict the number of scans required. One or "
                         "more of the scans you have provided do not have sdd values. Please try again using "
                         "scans with sdd values. ")
    sample_type = sgm_data.__dict__['scans'][file[0]].__getitem__(sample_name[0])['sample']
    for indiv_file in file:
        sample_name = list(sgm_data.__dict__['scans'][indiv_file].__dict__.keys())
        for scan in sample_name:
            if sgm_data.__dict__['scans'][indiv_file].__getitem__(scan)['sample'] != sample_type:
                raise ValueError("In order to predict, the scans in the hdf5 file passed in by user must all be from"
                                 " the same sample. The scans in the hdf5 file passed in by the user are not all from"
                                 " the same sample. Please try again with an hdf5 file only containing scans from the"
                                 " same sample. ")
    interp_list = sgm_data.interpolate(resolution=0.1)
    return interp_list


# # # Primarily for testing, not in final code.
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
        raise ValueError("lowest_variance function can only be used on hdf5 file containing 6 or more scans. The hdf5"
                         " file passed does not contain 6 or more scans. Please try again with an hdf5 file containing"
                         " 6 or more files")
    recent_variances = []
    variances = []
    recent_vals = []
    for diff in d_list:
        if len(recent_vals) < 4:
            recent_vals.append(diff)
        elif len(recent_vals) == 4:
            recent_vals.append(diff)
            for var in recent_vals:
                recent_variances.append(((var - np.mean(recent_vals)) ** 2))
            variances.append(np.sum(recent_variances) / len(recent_variances))
        else:
            recent_variances.clear()
            recent_vals.pop(0)
            recent_vals.append(diff)
            for var in recent_vals:
                recent_variances.append(((var - np.mean(recent_vals)) ** 2))
            variances.append(np.sum(recent_variances) / len(recent_variances))
    pos = (variances.index(min(variances)) + 4)
    return "\tThe lowest average variance (of the variance between 2 consecutive variances) between 5 consecutive " \
           "variances is " + str(min(variances)) + ".\nIt is reached with the variances between the values within " \
                                                   "the range: " + str(pos - 5) + " through to position: " + str(pos) +\
           ".\n"


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
    return amp * (idx + shift) ** (-3 / 2)


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


def predict_cut_off(d_list, percent_of_log=0.4):
    first_ten_average_variance = sum(d_list) / len(d_list)
    log_of_ftav = math.log(first_ten_average_variance, 10)
    log_cut_off = log_of_ftav * percent_of_log
    # # # Testing purposes only: print avg of first 10, log of avg of first 10, and cut-ff point, based on log of avg of
    # first 10
    print(" *** Messages starting with \" ***\" are messages printing for testing purposes only, will not be in final "
          "code.")
    print(" *** Average of initial 10 values: " + str(first_ten_average_variance))
    print(" *** Log of average of inital 10 values: " + str(log_of_ftav))
    print(" *** Cut off val, based on log of average of initial 10 values: " + str(log_cut_off) + "\n")
    # # #
    return log_cut_off


def find_cut_off(d_list, cut_off_point):
    keep_predicting = True
    i = 9
    while keep_predicting:
        avg_of_ten = (sum(d_list[i - 9:i]) / 9)
        log_of_avg_of_ten = math.log(avg_of_ten, 10)
        # # # Testing purposes only: print what number of scans we're at, the avg of previous nine scan values and log
        # of the average of the previous nine scan values.
        # print(" *** " + str(i-8) + ".      " + str(avg_of_ten) + "\t\t\t" + str(log_of_avg_of_ten))
        # # #
        if log_of_avg_of_ten <= cut_off_point:
            current_point = i + 1
            # # # Testing purposes only: print what number of scans were needed to get to cut-off val or less, and the
            # val that is less than or equal to the cut-off val or less
            print(" *** Cut-off at scan number: " + str(current_point))
            print(" *** Value at scan " + str(current_point) + "(scans at which cut-off point is reached): "
                  + str(log_of_avg_of_ten) + "\n\n")
            # # #
            return current_point
        else:
            j = 0
            indices = []
            while j <= i:
                indices.append(j)
                j += 1
            predicted_level = predict(d_list, indices)
            d_list = np.append(d_list, predicted_level[-1])
            # # # Testing purposes only: print the list of values (predicted and actual) in our list of scan values,
            # after our most recent prediction.
            # print(" *** \tnew d_list: " + str(d_list))
            # # #
            if i > 60:
                raise RuntimeError("Sufficiently accurate prediction cannot be made.")
            i += 1


# Must be modified to fit new equation.
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
        raise ValueError("Prediction can only be made with hdf5 file containing 10 or more scans. Less than 10 scans"
                         " in the hdf5 file passed in. Please try again with an hdf5 file containing 10 or more scans.")
    # Checking if the desired level of variance has already been reached in the first 10 scans.
    for element in d_list:
        if len(recent_differences) < 10:
            recent_differences.append(element)
        elif len(recent_differences) == 10:
            recent_differences.append(element)
            for var in recent_differences:
                recent_variances.append(((var - np.mean(recent_differences)) ** 2))
            if (np.sum(recent_variances) / len(recent_variances)) <= desired_difference:
                return 0
                # # # Testing purposes only: returns indices and values if desired level of variance already reached
                # return d_list, copied_indices
                # # #
        # # # Testing purposes only: code for if we're looking at 5 of initial 10 scans at a time, not all ten initial
        # scans at once. Current and final code should both look at all initial ten scans at once.
        # else:
        #     recent_variances.clear()
        #     recent_differences.pop(0)
        #     recent_differences.append(element)
        #     for var in recent_differences:
        #         recent_variances.append(((var - np.mean(recent_differences)) ** 2))
        #     if (np.sum(recent_variances) / len(recent_variances)) <= desired_difference:
        #         # # # For final code.
        #         # return 0
        #         # # # Temporary, for testing only.
        #         return d_list, copied_indices
        #         # # #
        # # #
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
        if num_predictions > 60:
            # If the desired_difference was the default value, not input by user.
            if desired_difference == 0.17961943:
                raise RuntimeError("Sufficiently accurate prediction cannot be made.")
            else:
                raise ValueError("Desired level of variance cannot be reached.")
    # # # Temporary, for testing only: returns d_list with the predicted values added, and the indices of the actual
    # and predicted d_list values.
    # return d_list, copied_indices
    # # #
    return num_predictions


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
        indices.append(i)
        prev_mean = cur_mean
    diff_list = np.nan_to_num(np.array(diff_list))
    return diff_list, indices


# def run_all(files):
#     """
#     Purpose: Puts together function calls to previously created functions so that desired results can be received from
#     running only this function. Returns the number of additional scans that should be run on a sample.
#     Returns:
#         determine_num_scans(returned_diff_list_listed[:10], returned_indices_listed[:10], 5)(int): the number of scans
#         required to reach the desired level of variance.
#     """
#     # Running functions to set up data.
#     list_of_files = file_retrieval(files)
#     interp_list = loading_data(list_of_files)
#     returned_data = interpolating_data(interp_list)
#     # Organizing data returned from functions to set up data.
#     returned_indices = returned_data[1]
#     returned_diff_list = returned_data[0]
#     returned_diff_list_listed = []
#     for item in returned_diff_list:
#         returned_diff_list_listed.append(item)
#     returned_indices_listed = []
#     for item in returned_indices:
#         returned_indices_listed.append(item)

    # # # Temporary, for testing only.
    # print("\tActual:")
    # print(lowest_variance(returned_diff_list_listed))
    # print(lowest_variance(returned_diff_list_listed[:10]))
    # i = 0
    # while i < len(returned_diff_list_listed):
    #     print(returned_diff_list_listed[i])
    #     i += 1
    # results = determine_num_scans(returned_diff_list_listed[:10], returned_indices_listed[:10], 0.0033708)
    # a = 0
    # print("\tPredicted:")
    # while a < len(results[0]):
    #     print(results[0][a])
    #     a += 1
    # predicted = results[0]
    # indices = results[1]
    # #
    # # Temporary, for testing with graph only/
    # returned_diff_list_listed.append(350)
    # returned_indices_listed.append(returned_diff_list_listed[-1] + 1)
    # plot1d([returned_indices_listed], [returned_diff_list_listed],
    #        "Sample Citric Acid", ["Actual Values", "Predicted Value"])
    # plot1d([returned_indices_listed[:10], indices[10:]], [returned_diff_list_listed[:10], predicted[10:]],
    #        "Sample Medge", ["Actual Values", "Predicted Value"])
    # # #
    # # # Temporary, for testing only.
    # return len(determine_num_scans(returned_diff_list_listed[:10], returned_indices_listed[:10])[0]) - 1
    # #
    # # For final code.
    # return determine_num_scans(returned_diff_list_listed[:10], returned_indices_listed[:10])
    # #


def testing():
    func = "file_retrieval()"
# # # New testing
    # scenario = "running with a correct path"
    # expected_result = ['C:/Users/roseh/Desktop/Internship/MyCode/h5Files\\Feb21-32-C-Bottom8.hdf5',
    #                    'C:/Users/roseh/Desktop/Internship/MyCode/h5Files\\Feb21-32-C-Top22.hdf5']
    # result = file_retrieval('C:/Users/roseh/Desktop/Internship/MyCode/h5Files/*Feb21-32-C*.hdf5')
    # if result != expected_result:
    #     print("Problem encountered with " + str(func) + " when " + str(scenario) + ". Expected to get\n \""
    #           + str(expected_result) + "\" \nfrom function, but instead got:\n \"" + str(result) + ".\"")
    # scenario = "running with pattern matching multiple scan sets"
    # expected_result = ['C:/Users/roseh/Desktop/Internship/MyCode/h5Files\\Fe2O3-7cc1.hdf5',
    #                    'C:/Users/roseh/Desktop/Internship/MyCode/h5Files\\Feb21-32-C-Bottom8.hdf5',
    #                    'C:/Users/roseh/Desktop/Internship/MyCode/h5Files\\Feb21-32-C-Top22.hdf5']
    # result = file_retrieval('C:/Users/roseh/Desktop/Internship/MyCode/h5Files/*fe*.hdf5')
    # if result != expected_result:
    #     print("Problem encountered with " + str(func) + " when " + str(scenario) + ". Expected to get\n \""
    #           + str(expected_result) + "\" \nfrom function, but instead got:\n \"" + str(result) + ".\"")
    # scenario = "running with a path to a non-hdf5 file"
    # expected_result = "ValueError"
    # try:
    #     result = file_retrieval('C:/Users/roseh/Desktop/Internship/TrainingInfo/*Readings*.hdf5')
    #     print("Problem encountered with " + str(func) + " when " + str(scenario) + ". Expected to get a "
    #           + str(expected_result) + "from function, but instead got \"" + str(result) + ".\"")
    # except ValueError:
    #     None
    # except:
    #     print("Problem encountered with " + str(func) + " when " + str(scenario) + ". Expected to get " +
    #           str(expected_result) + " from function, but instead got a different error. ")
    # scenario = "running with a path to a folder not containing a file with the specified pattern"
    # expected_result = "ValueError"
    # try:
    #     result = file_retrieval('C:/Users/roseh/Desktop/Internship/MyCode/h5Files/*ABCDEFG*.hdf5')
    #     print("Problem encountered with " + str(func) + " when " + str(scenario) + ". Expected to get a "
    #           + str(expected_result) + "from function, but instead got \"" + str(result) + ".\"")
    # except ValueError:
    #     None
    # except:
    #     print("Problem encountered with " + str(func) + " when " + str(scenario) + ". Expected to get a "
    #           + str(expected_result) + "from function, but instead got a different error. ")
    # scenario = "running with an int passed as parameter"
    # expected_result = "TypeError"
    # try:
    #     result = file_retrieval(1)
    #     print("Problem encountered with " + str(func) + " when " + str(scenario) + ". Expected to get a "
    #           + str(expected_result) + "from function, but instead got \"" + str(result) + ".\"")
    # except TypeError:
    #     None
    # except:
    #     print("Problem encountered with " + str(func) + " when " + str(scenario) + ". Expected to get a " +
    #           str(expected_result) + " from function, but instead got a different error.")
    #
    # scenario = "file_retrieval() with a float passed as parameter"
    # expected_result = "TypeError"
    # try:
    #     result = file_retrieval(1.1)
    #     print("Problem encountered with " + str(func) + " when " + str(scenario) + ". Expected to get a "
    #           + str(expected_result) + "from function, but instead got \"" + str(result) + ".\"")
    # except TypeError:
    #     None
    # except:
    #     print("Problem encountered with " + str(func) + " when " + str(scenario) + ". Expected to get a " +
    #           str(expected_result) + " from function, but instead got a different error.")
    # scenario = "file_retrieval() with a bool passed as a parameter"
    # expected_result = "TypeError"
    # try:
    #     result = file_retrieval(True)
    #     print("Problem encountered with " + str(func) + " when " + str(scenario) + ". Expected to get a "
    #           + str(expected_result) + "from function, but instead got \"" + str(result) + ".\"")
    # except TypeError:
    #     None
    # except:
    #     print("Problem encountered with " + str(func) + " when " + str(scenario) + ". Expected to get a " +
    #           str(expected_result) + " from function, but instead got a different error.")
    #
    # func = "loading_data()"
    # scenario = "running with a list of normal files, of same sample passed as parameter"
    # expected_result = "a list, with elements of type pandas.core.frame.DataFrame and for first item to have shape " \
    #                   "(500, 1031)"
    # result = loading_data(file_retrieval('C:/Users/roseh/Desktop/Internship/MyCode/h5Files/*Feb21-32-C*.hdf5'))[:1]
    # if type(result) != list or type(result[0]) != pandas.core.frame.DataFrame or result[0].shape != (500, 1031):
    #     print("Problem encountered with " + str(func) + " when " + str(scenario) + ". Expected to get\n \""
    #           + str(expected_result) + "\" \nfrom function, but instead got:\n \"" + str((result)) + ".\"")
    # scenario = "running with a list of normal files, of different samples, passed as parameter"
    # expected_result = "ValueError"
    # try:
    #     result = loading_data(file_retrieval
    #                           ('C:/Users/roseh/Desktop/Internship/MyCode/h5Files/*fe*.hdf5'))
    #     print("Problem encountered with " + str(func) + " when " + str(scenario) + ". Expected to get a "
    #           + str(expected_result) + "from function, but instead got \"" + str(result) + ".\"")
    # except ValueError:
    #     None
    # except:
    #     print("Problem encountered with " + str(func) + " when " + str(scenario) + ". Expected to get a " +
    #           str(expected_result) + " from function, but instead got a different error.")
    # scenario = "file with scans taken without sdds passed as parameter"
    # expected_result = "ValueError"
    # try:
    #     result = loading_data(file_retrieval
    #                           ('C:/Users/roseh/Desktop/Internship/MyCode/h5Files/NotPrimary/*C60*.hdf5'))
    #     print("Problem encountered with " + str(func) + " when " + str(scenario) + ". Expected to get a "
    #           + str(expected_result) + "from function, but instead got \"" + str(result) + ".\"")
    # except ValueError:
    #     None
    # except:
    #     print("Problem encountered with " + str(func) + " when " + str(scenario) + ". Expected to get a " +
    #           str(expected_result) + " from function, but instead got a different error.")
    # scenario = "file with scans with different signal lengths passed as parameter"
    # expected_result = "ValueError"
    # try:
    #     result = loading_data(file_retrieval
    #                           ('C:/Users/roseh/Desktop/Internship/MyCode/h5Files/NotPrimary/*Amm*.hdf5'))
    #     print("Problem encountered with " + str(func) + " when " + str(scenario) + ". Expected to get a "
    #           + str(expected_result) + "from function, but instead got \"" + str(result) + ".\"")
    # except ValueError:
    #     None
    # except:
    #     print("Problem encountered with " + str(func) + " when " + str(scenario) + ". Expected to get a " +
    #           str(expected_result) + " from function, but instead got a different error.")
    #
    # func = "interpolate_data()"
    # scenario = "running normally, with arguments that should not cause any errors"
    # expected_result = [np.array([377.68061431, 195.32269117, 80.04149944, 41.842782, 25.70721239, 18.31504572,
    #                              16.91533411, 22.14749096, 27.35943654, 9.91876352, 5.39883691, 4.81133296, 4.2115095,
    #                              3.56783262, 3.25804028, 4.25442748, 2.28460761, 2.23089784, 2.20958047, 1.80775647,
    #                              1.77157229, 1.70732908, 1.26666473, 1.18252861, 1.19934344, 1.0290663, 1.02153401,
    #                              1.61149456, 1.09338628]), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    #                                                         17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]
    # result = interpolating_data(loading_data(file_retrieval(
    #     'C:/Users/roseh/Desktop/Internship/MyCode/h5Files/*Feb21-32-C*.hdf5')))
    # i = 0
    # no_error = result[1] == expected_result[1]
    # while i < max(len(result[0]), len(expected_result[0])) and no_error:
    #     if result[0][i].round(8) != expected_result[0][i]:
    #         no_error = False
    #     i += 1
    # if not no_error:
    #     print("Problem encountered with " + str(func) + " when " + str(scenario) + ". Expected to get\n \""
    #           + str(expected_result) + "\" \nfrom function, but instead got:\n \"" + str(result) + ".\"")

    # func = "predict()"
    # situ = "regular lists passed for both variables"
    # expected_result = np.array([12.28664511, 11.19761088, 10.26045971, 9.44710632, 8.73580685, 8.10949264, 7.55459775,
    #                             7.06021877, 6.61750288, 6.21919547, 5.85930105, 5.53282617])
    # test_list1 = [500, 200, 100, 75, 65, 60, 56, 52, 50, 49, 48, 47]
    # test_list2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # result = predict(test_list1, test_list2)
    # same = True
    # i = 0
    # while i < len(expected_result):
    #     if round(expected_result[i], 5) != round(result[i], 5):
    #         same = False
    #     i += 1
    # if not same:
    #     print("Error with " + str(func) + " when using on " + str(situ) + ". Expected\n" + str(expected_result) +
    #           " but got \n" + str(result))
    # # predict(): list of both ints and floats for d_list
    # situ = "list of both ints and floats for d_list"
    # expected_result = np.array(
    #     [4.547589, 4.1427363, 3.79458128, 3.49259853, 3.22864805, 2.99634627, 2.79062404, 2.60741038,
    #      2.44340228, 2.29589481, 2.16265387, 2.04181978, 1.9318333])
    # test_list1 = [140, 120, 100, 90, 80, 75, 70, 67.5, 65, 63.5, 62, 61.5, 60]
    # test_list2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    # result = predict(test_list1, test_list2)
    # same = True
    # i = 0
    # while i < len(expected_result):
    #     if round(expected_result[i], 5) != round(result[i], 5):
    #         same = False
    #     i += 1
    # if not same:
    #     print("Error with " + str(func) + " when using on " + str(situ) + ". Expected\n" + str(expected_result) +
    #           " but got \n" + str(result))
    # # predict(): random list for indices
    # situ = "list of both ints and floats for d_list"
    # expected_result = np.array(
    #     [0.30573543, 0.14051718, 0.01636123, 0.23149906, 0.02215948, 0.27704, 0.19719866, 0.0404422,
    #      0.01406676, 0.21320443, 0.10660011, 0.00481277, 0.10660011])
    # test_list1 = [140, 120, 100, 90, 80, 75, 70, 67.5, 65, 63.5, 62, 61.5, 60]
    # test_list2 = [0, 10, 89, 3, 70, 1, 5, 42, 100, 4, 15, 220, 15]
    # result = predict(test_list1, test_list2)
    # same = True
    # i = 0
    # while i < len(expected_result):
    #     if round(expected_result[i], 5) != round(result[i], 5):
    #         same = False
    #     i += 1
    # if not same:
    #     print("Error with " + str(func) + " when using on " + str(situ) + ". Expected\n" + str(expected_result) +
    #           " but got \n" + str(result))
# # # End of new testing

    # # lowest_variance()
    # func = "lowest_variance()"
    # # lowest_variance(): testing on a regular list.
    # situ = "regular list"
    # expected_result = 'The lowest average variance (of the variance between 2 consecutive variances) between 5 ' \
    #                   'consecutive variances is 2.0.\nIt is reached with the variances between the values within the ' \
    #                   'range: 10 through to position: 14.\n'
    # test_list = [0, 5, 10, 15, 20, 10, 20, 30, 40, 50, 1, 2, 3, 4, 5]
    # result = lowest_variance(test_list)
    # print(result)
    # if str(result) != expected_result:
    #     print("Error with " + str(func) + " when using on " + str(situ) + ". Expected " + str(expected_result) +
    #           " but got " + str(result))
    # # lowest_variance(): testing on a list of all 0s.
    # situ = "list of all 0s"
    # expected_result = 'The lowest average variance (of the variance between 2 consecutive variances) between 5 ' \
    #                   'consecutive variances is 0.0.\nIt is reached with the variances between the values within the ' \
    #                   'range: 7 through to position: 11.\n'
    # test_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # result = lowest_variance(test_list)
    # if str(result) != expected_result:
    #     print("Error with " + str(func) + " when using on " + str(situ) + ". Expected " + str(expected_result) +
    #           " but got " + str(result))
    # # lowest_variance(): testing on a list containing ints and floats.
    # situ = "list containing ints and floats "
    # expected_result = 'The lowest average variance (of the variance between 2 consecutive variances) between 5' \
    #                   ' consecutive variances is 0.01999999999999999.\nIt is reached with the variances between the' \
    #                   ' values within the range: 10 through to position: 14.\n'
    # test_list = [.5, .1, 1.5, 2.0, 2.5, 1, 2, 3, 4, 5, 1.1, 1.2, 1.3, 1.4, 1.5]
    # result = lowest_variance(test_list)
    # if str(result) != expected_result:
    #     print("Error with " + str(func) + " when using on " + str(situ) + ". Expected " + str(expected_result) +
    #           " but got " + str(result))
    #

    # func = "determine_num_scans()"
    # situ = "regular values input for d_list and indices"
    # expected_result = 15
    # test_list1 = [140, 120, 100, 90, 80, 75, 70, 67.5, 65, 63.5]
    # test_list2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # result = determine_num_scans(test_list1, test_list2)
    # if result != expected_result:
    #     print("Error with " + str(func) + " when using on " + str(situ) + ". Expected " + str(expected_result) +
    #           " but got " + str(result))
    # # determine_num_scans(): d_list too small.
    # situ = "d_list too small"
    # expected_result = 1
    # test_list1 = [140, 120, 100]
    # test_list2 = [0, 1, 2]
    # result = determine_num_scans(test_list1, test_list2)
    # if result != expected_result:
    #     print("Error with " + str(func) + " when using on " + str(situ) + ". Expected " + str(expected_result) +
    #           " but got " + str(result))
    # # determine_num_scans(): d_list where desired level of noise already reached.
    # situ = "desired level of noise already reached"
    # expected_result = 0
    # test_list1 = [100, 200, 0.1, 0.11, 0.111, 0.1111, 0.11111, 300, 500, 20]
    # test_list2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # result = determine_num_scans(test_list1, test_list2)
    # if result != expected_result:
    #     print("Error with " + str(func) + " when using on " + str(situ) + ". Expected " + str(expected_result) +
    #           " but got " + str(result))
    # # determine_num_scans(): normal d_list, random ints in indices.
    # situ = "normal d_list, random ints in indices"
    # expected_result = 15
    # test_list1 = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    # test_list2 = [0, 10, 4, 88, 90, 4, 67, 12, 100, 43]
    # result = determine_num_scans(test_list1, test_list2)
    # if result != expected_result:
    #     print("Error with " + str(func) + " when using on " + str(situ) + ". Expected " + str(expected_result) +
    #           " but got " + str(result))
    # # determine_num_scans(): very low desired noise level.
    # situ = "very low desired noise level"
    # expected_result = 583
    # test_list1 = [140, 120, 100, 90, 80, 75, 70, 67.5, 65, 63.5]
    # test_list2 = [0, 10, 4, 88, 90, 4, 67, 12, 100, 43]
    # result = determine_num_scans(test_list1, test_list2, 0.00001)
    # if result != expected_result:
    #     print("Error with " + str(func) + " when using on " + str(situ) + ". Expected " + str(expected_result) +
    #           " but got " + str(result))

# # # TEMPORARY


def run_all(files):
    list_of_files = file_retrieval(files)
    interp_list = loading_data(list_of_files)
    returned_data = interpolating_data(interp_list)
    # # Organizing data returned from functions to set up data.
    # returned_indices = returned_data[1]
    # returned_diff_list = returned_data[0]
    # returned_diff_list_listed = []
    # for item in returned_diff_list:
    #     # print(item)
    #     returned_diff_list_listed.append(item)
    # returned_indices_listed = []
    # for item in returned_indices:
    #     returned_indices_listed.append(item)
    # # Printing info from new functions (Aug 6 2021)
    # cut_off_point = predict_cut_off(returned_diff_list_listed[:9])
    # # # # using imported plot1D
    # # boop=["hello"]
    # # plot1d(returned_diff_list, returned_indices, "Si-Phenyl", boop)
    # return find_cut_off(returned_diff_list_listed[:10], cut_off_point) - 10


print("Number of additional scans needed:\t" +
      str(run_all('C:/Users/roseh/Desktop/Internship/MyCode/h5Files/*fe2o3*.hdf5')))


