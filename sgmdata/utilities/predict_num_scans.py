import pandas
import numpy as np
from lmfit import Model
import math
from bokeh.plotting import figure
from bokeh.io import show



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
    file = list(sgm_data.__dict__['scans'].keys())
    sample_name = list(sgm_data.__dict__['scans'][file[0]].__dict__.keys())
    sample_type = sgm_data.__dict__['scans'][file[0]].__getitem__(sample_name[0])['sample']
    interp_list = []
    if sgm_data.interpolated is True:
        print("Sample " + str(sample_type) + "'s data already interpolated. Fetching interpolated data...")
        for item in sgm_data.__dict__['scans']:
            for entry in sgm_data.__dict__['scans'][item]:
                interp_list.append(pandas.DataFrame.from_dict(sgm_data.__dict__['scans'][item].__dict__[entry]
                                                              ['binned']['dataframe']))
    else:
        print("Sample " + str(sample_type) + "'s data not yet interpolated. Fetching data to interpolate...")
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
    if num_scans >= (len(interp_list)):
        num_scans = len(interp_list)
    returned_data = extracting_data(interp_list[:num_scans])
    returned_diff_list_listed = [item for item in returned_data[0]]
    cut_off_point_info = predict_cut_off(returned_diff_list_listed[: num_scans - 1], percent_of_log)
    number_of_scans = find_cut_off(returned_diff_list_listed[: num_scans - 1], cut_off_point_info[2])
    i = 1
    num_scans_listed = []
    while i <= number_of_scans[0]:
        num_scans_listed.append(i)
        i += 1
    if verbose:
        print(
            " *** Messages starting with \" ***\" are messages containing additional data, other than the number of "
            "additional scans needed." + "\n *** Average of initial 10 values: " + str(cut_off_point_info[0]) +
            "\n *** Log of average of initial 10 values: " + str(cut_off_point_info[1]) +
            "\n *** Cut off val, based on log of average of initial 10 values: " + str(cut_off_point_info[2]) +
            "\n *** Cut-off at scan number: " + str(number_of_scans[0]) +
            "\n *** Value at scan " + str(number_of_scans[0]) + "(scans at which cut-off point is reached): " +
            str(number_of_scans[1][-1]))
        plot_predicted(num_scans_listed[num_scans - 1:], number_of_scans[1], cut_off_point_info[2], interp_list,
                       sample_type, num_scans)
    return number_of_scans[0] - num_scans

