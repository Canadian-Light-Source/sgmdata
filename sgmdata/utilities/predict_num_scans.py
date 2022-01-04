import glob
import asyncio
import h5py
import pandas
import numpy as np
from lmfit import Model
from collections import OrderedDict
import math
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import HoverTool, CustomJS, BooleanFilter, LinearColorMapper, LogColorMapper, ColorBar
from bokeh.io import show
### FOLLOWING JUST FOR MY COMP
import os
import sys
import inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(
    inspect.currentframe()))))))
### JUST FOR MY COMP DONE
import sgmdata
from sgmdata.load import SGMData
from sgmdata.search import SGMQuery


def file_retrieval(path):
    """
    Purpose: Gets files that match a specified pattern from a specified directory.
    Parameters:
        path(str): path to directory and pattern to look for in directory.
    Returns:
        sgmdata.load.SGMData(list_of_files)(SGMData object): an SGMData object made from the files the user entered the
         path to.
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
    return sgmdata.load.SGMData(list_of_files)


def check_sample_fitness(sgm_data):
    """
        ### Description:
    -----
        Determines whether the data from the SGMData object passed in has already been interpolated. If it has an
        SGMQuery is used to retrieve it. Otherwise, the SGMData object is checked to ensure it is suitable for
        interpolation, and if it is the data in the SGMData object is interpolated.
    ### Args:
    -----
        > **sgm_data** *(type: SGMData object)* -- AN SGMData object containing the date for a sample the user would
            like to know more about.
    ### Returns:
    -----
        > **interp_list** *(type: list of pandas dataframes)* -- the interpolated data for the SGMData object passed
            in by the user.
    """
    interp = False  # this is a temporary variable, delete later
    file = list(sgm_data.__dict__['scans'].keys())
    sample_name = list(sgm_data.__dict__['scans'][file[0]].__dict__.keys())
    sample_type = sgm_data.__dict__['scans'][file[0]].__getitem__(sample_name[0])['sample']
    sgm_data.interpolate(resolution=0.1)
    interp_list = []
    #         if sgm_data.interpolated is True: #this is what we want in the end, but cannot use yet
    if interp is True:  # this is just a temp replacement for the above line. Remove later.

        print("SGMQuery required for sample " + str(sample_type))
        for item in sgm_data.__dict__['scans']:
            for entry in sgm_data.__dict__['scans'][item]:
                interp_list.append(pandas.DataFrame.from_dict(sgm_data.__dict__['scans'][item].__dict__[entry]
                                                              ['binned']['dataframe']))
    else:
        print("interpolation needed")
        if len(sgm_data.__dict__['scans']) == 0:
            raise ValueError(
                "hdf5 file must contain scans to be able to predict the number of scans required. The hdf5 "
                "file you have provided does not contain any scans. Please try again with an hdf5 file that"
                " does contain scans.")
        has_sdd = False
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
        for indiv_file in file:
            sample_name = list(sgm_data.__dict__['scans'][indiv_file].__dict__.keys())
            for scan in sample_name:
                if sgm_data.__dict__['scans'][indiv_file].__getitem__(scan)['sample'] != sample_type:
                    raise ValueError(
                        "In order to predict, the scans in the hdf5 file passed in by user must all be from"
                        " the same sample. The scans in the hdf5 file passed in by the user are not all from"
                        " the same sample. Please "
                        "try again with an hdf5 file only containing scans from the"
                        " same sample. ")
        interp_list = sgm_data.interpolate(resolution=0.1)
    return interp_list


# # # Primarily for testing, not in final code.
def lowest_variance(d_list):
    """
    ### Description:
    -----
        Takes a list of differences between values. Calculates the lowest variance between 5 of these consecutive
        differences and keeps track of the positions of these 5 values in d_list.
    ### Args:
    -----
        > **d_list** *(type: list of floats)* -- A list of the difference in average variances between each 2
            consecutive values.
    ### Returns:
    -----
        > *(type: str)* -- The lowest variance value and the index of the list differences that make up this variance
            value.
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
    ### Description:
    -----
        Predicts the amount of noise in the next scan.
    ### Args:
    -----
        > **idx** *(type: np.array)* -- An array of the indices of the values in sdd_list that were previously deemed
            to be acceptable sdd values.
        > **amp** *(type: float)* -- How much to amplify the noise level by.
        > **shift** *(type: float)* -- Amount to shift the index by.
        > **ofs** *(type: float)* -- How much to offset the noise level by.
    ### Returns:
    -----
        > *(type: np.array)* -- the values in idx fit to the curve outlined by amp, shift and ofs.
    """
    return amp * (idx + shift) ** (-3 / 2)


def predict_next_scan(d_list, cur_indices):
    """
    ### Description:
    -----
        Takes predicted differences from d_list and inputs them to the noise function. This predicts the level of
        noise in the next scan.
    ### Args:
    -----
        > **d_list** *(type: list of floats)* -- The values from which the user would like the end point to be based on.
            Expected, but not required, to be the variance between the noise levels in the first 10 scans of a sample.
        > **num_indices** *(type: list of ints)* -- The indexes of d_list. Will expand as long as d_list is expanding.
    ### Returns:
    -----
        > *(type: np.array)* -- An array of the differences between consecutive averages of variances.
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
    '''
    ### Description:
    -----
        Takes the variance between the values in d_list and finds the log value of their average. Multiplies log value
        by the percent_of_log to get the value at which scanning should stop.
    ### Args:
    -----
        > **d_list** *(type: list of floats)* -- The values from which the user would like the end point to be based on.
            Expected, but not required, to be the variance between the noise levels in the first 10 scans of a sample.
        > **percent_of_log** *(type: float)* -- The value by which the log of the average of the d_list values is
            multiplied to get the value where scanning should stop. Default value is 0.4.
    ### Returns:
    -----
        > **first_ten_average_variance** *(type: float)* -- The average of the values in d_list.
        > **log_of_ftav** *(type: float)* -- The log value of first_ten_average_variance.
        > **log_cut_off** *(type: float)* -- The ideal value of the log of the average of the variance between the
            last 10 scans.
    '''
    first_ten_average_variance = sum(d_list) / len(d_list)
    log_of_ftav = math.log(first_ten_average_variance, 10)
    log_cut_off = log_of_ftav * percent_of_log
    return first_ten_average_variance, log_of_ftav, log_cut_off


def find_cut_off(d_list, cut_off_point):
    '''
    ### Description:
    -----
        Uses a list of variances between scans to predict the values that would be associated with the next scan. Checks
        if the log of the average of the variance between the most recently predicted scan and preceding scan, and
        the variances between the 9 scans before it is less than or equal to cut_off_point. If it isn't then the values
        that would be associated with the next scan are predicted. If it is then predictions stop, and the number of
        predictions it took to get to that point is returned to the user, indicating that that many additional scans
        should be taken of the sample.
    ### Args:
    -----
        > **d_list** *(type: list of floats)* -- The variance between the initial scans of a sample. Initial number of
            scans is expected to be 10, but it can be any number.
        > **cut_off_point** *(type: float)* -- The ideal value of the log of the average of the variance between the
            last 10 scans.
    ### Returns:
    -----
        > **current_point** *(type: int)* -- The number of scans that had to be predicted for the log of the average of
            the variance between last 10 scans to be equal to or smaller than cut_off_point. Indicative of number of
            additional scans that should be taken of the sample.
        > **log_of_avg_of_ten** *(type: float)* -- The log of the average of the variance between the last 10 scans
            when the log of the average of the variance between last 10 scans to be equal to or smaller than
            cut_off_point.
    '''
    keep_predicting = True
    i = len(d_list)
    log_of_avg_of_ten = []
    while len(d_list) < 9:
        j = 0
        indices = []
        while j < i:
            indices.append(j)
            j += 1
        predicted_level = predict_next_scan(d_list, indices)
        d_list = np.append(d_list, predicted_level[-1])
        i += 1
    while keep_predicting:
        avg_of_ten = (sum(d_list[i - 9:i]) / 9)
        log_of_avg_of_ten.append(math.log(avg_of_ten, 10))
        if log_of_avg_of_ten[-1] <= cut_off_point:
            current_point = i + 1
            return current_point, log_of_avg_of_ten, d_list
        else:
            j = 0
            indices = []
            while j < i:
                indices.append(j)
                j += 1
            predicted_level = predict_next_scan(d_list, indices)
            d_list = np.append(d_list, predicted_level[-1])
            if i > 60:
                raise RuntimeError("Sufficiently accurate prediction cannot be made.")
            i += 1


def determine_num_scans(d_list, indices, desired_difference):
    """
    ### Description:
    -----
        takes a list of the the averages of the variances of the initial scans provided by the user. Uses these to
        determine how many additional scans are needed to have an average variance of desired_difference across the
        most recent 10 scans.
    ### Args:
    -----
        > **d_list** *(type: list of floats)* -- A list of the averages of the variances of the initial scans provided
            by the user. As our determine_scan_num function continues it will add values to the d_list.
        > **indices** *(type: list of ints)* -- The indexes of the scans used to formulate d_list.
        > **desired_noise** *(type: float)* -- the variance between 10 consecutive scans the user would like to
        achieve, ie, we'll need to continue scaning until this level of variance is reached.
    ### Returns:
    -----
        > **num_predictions + 1** *(int)*: The number of scans required to reach the desired level of variance. Adding
         one because, since we're working differences in d_list, and a difference comes from 2 values, to get any
         number of differences you'll need that number of scans plus 1.
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
    for element in d_list:
        if len(recent_differences) < 10:
            recent_differences.append(element)
        elif len(recent_differences) == 10:
            recent_differences.append(element)
            for var in recent_differences:
                recent_variances.append(((var - np.mean(recent_differences)) ** 2))
            if (np.sum(recent_variances) / len(recent_variances)) <= desired_difference:
                return 0
    while keep_predicting:
        predicted_level = predict(d_list, copied_indices)
        copied_indices.append(int(copied_indices[-1]) + 1)
        num_predictions = num_predictions + 1
        d_list = np.append(d_list, predicted_level[-1])
        recent_variances.clear()
        recent_differences.pop(0)
        recent_differences.append(predicted_level[-1])
        for var in recent_differences:
            recent_variances.append(((var - np.mean(recent_differences)) ** 2))
        variances.append((np.sum(recent_variances)) / len(recent_variances))
        if variances[-1] <= desired_difference:
            keep_predicting = False
        if num_predictions > 60:
            if desired_difference == 0.17961943:
                raise RuntimeError("Sufficiently accurate prediction cannot be made.")
            else:
                raise ValueError("Desired level of variance cannot be reached.")
    return num_predictions


def extracting_data(interp_list_param):
    """
    ### Description:
    -----
        Takes the results returned by SGMData's "interpolate" function, or from an SGMQuery and collects the sdd values
        within it. Sorts through these sdd values and removes unfit values. Keeps a separate list of the indices of the
        fit values. Deals with nan and infinity values in the list of sdd values. Returns the modified list of sdd
        values and the list of the indices of fit value to caller as a numpy arrays.
    ### Args:
    -----
        > **interp_list_param** *(type: list of pandas dataframes)* -- The list of results returned by SGMData's
            "interpolate" function, or from the ".data" attribute the return value of an SGMQuery.
    ### Returns:
    -----
        > **diff_list** *(type: list of floats)*: The predicted number of additional scans that should be taken of a
            sample.
        > **indices** *(type: list of ints)*: The indexes of the scans used to formulate diff_list.
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


def plot_predicted(scans_x, noise_levels_y, cut_off, interp_list,  sample_type, num_scans=10):
    '''
    ### Description:
    -----
        Takes the information about a sample and plots it onto a graph. Then takes the data predicted for the sample by
        the 'predict_num_scans' function and plots it on the same graph. Allows for easy comparison of actual data to
        predicted data.
    ### Args:
    -----
        > **scans_x** *(type: list of ints)* -- the indexes of all of the scans up to, and including, the index of the
            last scan the predict_num_scans function predicts will be needed.
        > **noise_levels_y** *(type: numpy array of floats)* -- the averages of the noise levels of all the scans
            leading up to a scan. The averages match up with the indexes in scans_x.
        > **cut_off** *(type: float)* -- The predicted value of the log of the average of the variance between 10
            scans that is low enough that no more scans need to be taken.
        > **interp_list** *(type: list of pandas dataframes)* -- the interpolated data for each scan.
        > **sample_type** *(type: str)* -- the name of the sample that the user wants more information on.
        > **num_scans** *(type: optional int)* -- the number of scans that the user initially provided. Default value
            is 10.
    '''
    extracted_data = extracting_data(interp_list)[0]
    log_of_extracted = []
    i = 0
    while i < len(extracted_data):
        if i + 1 >= num_scans:
            avg_of_ten = (sum(extracted_data[i - 9:i]) / 9)
            log_of_extracted.append(math.log(avg_of_ten, 10))
        i += 1
    TOOLS = 'pan, hover,box_zoom,box_select,crosshair,reset,save'
    fig = figure(
        tools=TOOLS,
        title="Predicted Number of Scans Required for Sample " + sample_type,
        background_fill_color="white",
        background_fill_alpha=1,
        x_axis_label="Scans From Which Average is Derived: ",
        y_axis_label="Log of Average of Noise Values: ",
    )
    ind = 0
    while ind < (len(scans_x)):
        fig.circle(x=scans_x[ind], y=noise_levels_y[ind], color="blue", legend_label="Predicted Log of Average of Noise"
                                                                                     " Values of Scans")
        if ind < len(log_of_extracted):
            fig.circle(x=scans_x[ind], y=log_of_extracted[ind], color="red", legend_label="Actual Log of Average of "
                                                                                          "Noise Values of Scans")
        ind += 1

    fig.line(x=scans_x, y=cut_off, color="yellow", legend_label="Log of Average of Noise Values at Which Sample "
                                                                "Scanning Can Stop")
    fig.legend.location = "top_right"
    fig.legend.click_policy = "hide"
    fig.legend.title = "Legend For Predictive Function"
    show(fig)


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


def predict_num_scans(data, verbose=False, percent_of_log=0.4, num_scans=10):
    """
    ### Description:
    -----
        Takes the SGMData object of a sample and uses a combination of other functions to predict how many additional
        scans should be taken of that sample.
    ### Args:
    -----
        > **data** *(type: SGMData object)* -- The SGMData object for the sample on which the user would like more
            information.
        > **verbose** *(type: optional boolean)* -- Default value is False. If set to True, gives user additional data
            on how the additional number of scans needed was calculated.
        > **percent_of_log** *(type: optional float)* -- Default value is 0.4. The average of the noise values of the
            first ten scans is taken, and the log of it is found. Scans continue to be taken, and the average of the
            noise values of the most recent ten scans is taken. The log of this average is taken,and if it's less than
            percent_of_log multiplied by the log of the first ten averages, then scanning stops.
        > **num_scans** *(type: optional int)* -- Default value is 10. The number of scans from the scans provided by
            the user, that the user would like to be used to predict the number of additional scans to take.
    ### Returns:
    -----
        >*(int)*: The predicted number of additional scans that should be taken of a sample.
    """
    interp_list = check_sample_fitness(data)
    file = list(data.__dict__['scans'].keys())
    sample_name = list(data.__dict__['scans'][file[0]].__dict__.keys())
    sample_type = data.__dict__['scans'][file[0]].__getitem__(sample_name[0])['sample']

    if num_scans > (len(interp_list) + 1):
        num_scans = len(interp_list)
    returned_data = extracting_data(interp_list[:num_scans])
    returned_diff_list_listed = []
    returned_indices_listed = []

    for item in returned_data[0]:
        returned_diff_list_listed.append(item)
    for item in returned_data[1]:
        returned_indices_listed.append(item)

    cut_off_point_info = predict_cut_off(returned_diff_list_listed[: num_scans - 1], percent_of_log)
    cut_off_point = cut_off_point_info[2]
    number_of_scans = find_cut_off(returned_diff_list_listed[: num_scans - 1], cut_off_point)

    i = 1
    num_scans_listed = []
    while i <= number_of_scans[0]:
        num_scans_listed.append(i)
        i += 1
    plot_predicted(num_scans_listed[num_scans - 1:], number_of_scans[1], cut_off_point, interp_list, sample_type, num_scans)

    if verbose:
        print(
            " *** Messages starting with \" ***\" are messages containing additional data, other than the number of "
            "additional scans needed." + "\n *** Average of initial 10 values: " + str(cut_off_point_info[0]) +
            "\n *** Log of average of initial 10 values: " + str(cut_off_point_info[1]) +
            "\n *** Cut off val, based on log of average of initial 10 values: " + str(cut_off_point_info[2]) +
            "\n *** Cut-off at scan number: " + str(number_of_scans[0]) +
            "\n *** Value at scan " + str(number_of_scans[0]) + "(scans at which cut-off point is reached): " +
            str(number_of_scans[1][-1]))
    return number_of_scans[0] - 10


sample = file_retrieval('C:/Users/roseh/Desktop/Internship/SignalToNoiseConvergence/h5Files/*Bee*.hdf5')
print(predict_num_scans(sample, True))

