import h5py
import h5pyd
import numpy as np
from bokeh.io import show
from bokeh.plotting import figure



#Utility for visualizing HDF5 layout.
def printTree(name, node):
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    depth = len(str(name).split('/'))
    mne = str(name).split('/')[-1]
    if isinstance(node, h5py.Group) or isinstance(node, h5pyd.Group):
        sep = ""
        typ = " <Group> attrs:{"
        for key, val in node.attrs.items():
            typ += "%s:%s," % (key, str(val))
        typ += "}"
        for i in range(depth - 1):
            sep += "    "
        sep += "->"
    if isinstance(node, h5py.Dataset) or isinstance(node, h5pyd.Dataset):
        sep = ""
        typ = " <Dataset> type:%s shape:%s attrs:{" % (node.dtype, str(node.shape))
        for key, val in node.attrs.items():
            typ += "%s:%s," % (key, str(val))
        typ += "}"
        for i in range(depth-1):
            sep += "    "
        sep += "|-"
    print('{:5.120}'.format(BOLD + sep + END + UNDERLINE + str(mne) + END + typ))

def h5tree(h5):
    """
    Description:
    -----
    A function to output the data-tree from an hdf5 file object.

    Args:
    -----
        h5 - Any H5File object, from h5py.

    returns None

    """
    h5.visititems(printTree)
    
    
#Scan health functions (from Habib)

def get_moving_average(data, window_size=4):
    """ 
    Description:
    -----
    A function to calculate the moving average of data using numpy's implementation of convolution
    
    Args:
    -----
        data (numpy.ndarray): A 1d numpy array of the data on which to calculate the moving average
        window_size (int): An integer value of the number of samples to consider when averaging 
        
    Returns:
    -----
        m_average (numpy.ndarray): A 1d numpy array of the calculated moving average. The size of
                                   "m_average" is the same as the size of input "data"
    """ 
    
    # Calculate the moving average
    window = np.ones(int(window_size))/float(window_size)
    m_average = np.convolve(data, window, 'same')
            
    # Return the moving average of input data 
    return m_average


def get_moving_slope(dep, indep, window_size=4):
    
    slope = np.convolve(dep, window_size, mode='same') / np.convolve(indep, window_size, mode='same')
    return slope


def test_abrupt_change(detector, sigma=10.0, tolerance=1000.0): 
    """ 
    Description:
    -----
    A function to detect the percentage of abrupt change in a single detector data
    
    Args:
    -----
        detector (tuple): A python tuple in the form (detector_name, data). The detector_name is a
                          string, while data is a numpy array of the data to detect abrupt change
        sigma (float): A float value for standard deviation. This number define how different a specific 
                       count should be from the standard deviation to be considered abrupt change 
        tolerance (float): A float value specifying the absolute tolerance parameter for detecting if
                           two numbers should be considered close to each other
        
    Returns:
    -----
        String: Percentage of the data that is normal count and the percentage the fuction think might be abrupt change
    """
    
    # Get the detector name and the actual data
    name, counts = detector
    
    # If the data is 2d, sum all the counts in the detector to get the overall input count rate (ICR)
    if counts.ndim == 2:
        counts = np.sum(counts, axis=1)
        
    # Get the moving average of the data
    m_average = get_moving_average(counts,window_size=50)
    
    # Calculate the standard deviation of the moving average
    stdev = np.std(m_average)
        
    # A list to store boolean values specifying if a count is normal or considered abrupt change
    bool_list = []
    
    # A list to store the index of the detected abrupt changes in the data array
    index_list = []
    
    # Loop through each count and decide if it's normal or there is abrupt change based the values of sigma and tolerance
    for i in range(len(counts)):
        if (not(np.isclose(m_average[i]-(stdev*sigma), 0, atol=tolerance))) and (np.isclose(counts[i], 0, atol=tolerance)):
            bool_list.append(True)
            index_list.append(i)
        elif (counts[i] > m_average[i]+(stdev*sigma)) | (counts[i] < m_average[i]-(stdev*sigma)):
            bool_list.append(True)
            index_list.append(i)
        else:
            bool_list.append(False)
        
    bool_arr = np.array(bool_list)
    size_bool_arr = np.size(bool_arr)
        
    # Get the percentage of abrupt change
    percent_true = round(((np.count_nonzero(bool_arr)) / size_bool_arr) * 100, 3)

    # Get the percentage of normal counts
    percent_false = round(((size_bool_arr - np.count_nonzero(bool_arr)) / size_bool_arr) * 100, 3)
    
    return percent_false, percent_true    
    
def test_detector_count_rates(detector, scalar_range=(5000, 500000), sdds_range=(1000, 30000)): 
    """ 
    Description:
    -----
    A function to detect if count rates of a specific detector are within a defined range
    
    Args:
    -----
        detector (tuple): A python tuple in the form (detector_name, data). The detector_name is a
                          string, while data is a numpy array of the data to detect the count rates 
        tey_range (tuple): A python tuple defining the normal count range for the tey detector 
                           in the form (min_normal_count, max_normal_count)
        io_range (tuple): A python tuple defining the normal count range for the io detector 
                          in the form (min_normal_count, max_normal_count)
        sdds_range (tuple): A python tuple defining the normal count range for the sdd[1-4] 
                            detectors in the form (min_normal_count, max_normal_count)
                           
    Returns:
    -----
        String: Percentage of the data that is within normal count range and the percentage 
                that is outside the defined normal count range
    """

    # Get the detector name and the actual data
    name, counts = detector
    
    # Get the lower and upper limit of normal count rates for the specified detector
    count_range = []
    
    if "sdd" in name:
        count_range = sdds_range
    else:
        count_range = scalar_range
        
    dimension = counts.ndim
    
    # A list to store boolean values specifying if a count is normal or outside the specified range
    bool_arr = ""

    # Test each individual counts and decide if it is within or outside the specified range
    if dimension == 1:
        bool_arr = ((counts < count_range[0]) | (counts > count_range[1]))
    elif dimension == 2:
        counts = np.sum(counts, axis=1)
        bool_arr = ((counts < count_range[0]) | (counts > count_range[1]))

    size_bool_arr = np.size(bool_arr)
 
    # Get the percentage of counts that are outside the specified range
    percent_true = round(((np.count_nonzero(bool_arr)) / size_bool_arr) * 100, 3)
    
    # Get the percentage of counts that are within the specified range
    percent_false = round(((size_bool_arr - np.count_nonzero(bool_arr)) / size_bool_arr) * 100, 3)
    
    # Get the index of the counts that are outside the specified range within the data array
    index_list = np.where(bool_arr == True)[0]
        
    return percent_false, percent_true

def test_beam_dump(detector, indep):
    """ 
    Description:
    -----
    A function to detect the percentage of beam dump in a single detector data
    
    Args:
    -----
        detector (tuple): A python tuple in the form (detector_name, data). 
                          The detector_name is a string, while data is a numpy 
                          array of the data to detect beam dump
        indep (numpy.ndarray): A numpy array of the independent variable data
        
    Returns:
    -----
        String: Percentage of the data that is normal count and the percentage 
                the fuction think is a beam dump
    """
    
    # A boolean variable used to detect if the input detector data is 1d or 2d
    is_data_2d = False
    
    # Get the detector name and data
    name, counts = detector
    
    # If the data is 2d, sum all the counts in the detector to get the overall 
    # input count rate (ICR)
    if counts.ndim == 2:
        counts = np.sum(counts, axis=1)
        is_data_2d = True
        
    # Get the moving slope of the input data
    m_slope = get_moving_slope(counts, indep)
    
    # Get the moving average of the calculated moving slope
    m_average = get_moving_average(m_slope, window_size=4)
            
    # List to store boolean values specifying normal counts and beam dump
    bool_list = []
    
    # List to store the index of the detected beam dump 
    index_list = []
    
    # How close to 0 the moving slope should be to be considered a beam dump
    tolerance = np.std(m_slope)
           
        
    if is_data_2d:
        
        start = 0
        
        for i in range(len(m_slope)):
            if (m_slope[i]==0.0 and m_average[i]<0.1):
                start = i
                break  
        
        if not start > 0:
            start = np.argmax(m_slope<1)
    
        if start == 0:
            
            bool_list = [False] * len(m_slope)
            bool_arr = np.array(bool_list)
            size_bool_arr = np.size(bool_arr)
 
            percent_true = round(((np.count_nonzero(bool_arr)) / size_bool_arr) * 100, 3)
            percent_false = round(((size_bool_arr - np.count_nonzero(bool_arr)) / size_bool_arr) * 100, 3)
        
            return percent_false, percent_true
        
        for i in range(len(m_slope)):
                        
            if i<start-1:
                bool_list.append(False)
                continue

            if (np.isclose(m_slope[i], 0, atol=tolerance)):
                bool_list.append(True)
                index_list.append(i)
            else:
                bool_list.append(False)
        
    else:
        
        start = 0
        
        for i in range(len(m_slope)):
            if (np.isclose(m_slope[i], 0, atol=tolerance/2)):
                start = i
                break  
                                                            
        if np.all(m_slope==0) or start == 0: 
            
            bool_list = [False] * len(m_slope)
            bool_arr = np.array(bool_list)
            size_bool_arr = np.size(bool_arr)
 
            percent_true = round(((np.count_nonzero(bool_arr)) / size_bool_arr) * 100, 3)
            percent_false = round(((size_bool_arr - np.count_nonzero(bool_arr)) / size_bool_arr) * 100, 3)
        
            return percent_false, percent_true
            
        for i in range(len(m_slope)):
                        
            if i<start-1:
                bool_list.append(False)
                continue

            if (np.isclose(m_slope[i], 0, atol=tolerance)):
                bool_list.append(True)
                index_list.append(i)
            else:
                bool_list.append(False)

    
    bool_arr = np.array(bool_list)
    size_bool_arr = np.size(bool_arr)
 
    # Get the percentage of beam dump 
    percent_true = round(((np.count_nonzero(bool_arr)) / size_bool_arr) * 100, 3)
    
    # Get the percentage of normal counts
    percent_false = round(((size_bool_arr - np.count_nonzero(bool_arr)) / size_bool_arr) * 100, 3)
        
    return percent_false, percent_true



def scan_health(df, verbose=False, sdd_max=105000):
    """
    Description:
    -----
    Function takes in a interpolated scan (a pandas DataFrame), and returns the overall health.

    Args:
    -----
        df :  pandas dataframe from SGMScan.binned.
        verbose: Explain the returned output in plain text.
        sdd_max (int): 105000 (default) - saturation value for total SDD counts/s


    returns (tuple):  (Discontiunty %,  Beam-dump %,  Saturation %)
    """

    EN = df.index.to_numpy()

    IO_R = df.filter(regex=("i0.*"), axis=1).to_numpy()
    TEY = df.filter(regex=("tey.*"), axis=1).to_numpy()
    diode = df.filter(regex=("pd.*"), axis=1).to_numpy()
    
    SDD1 = df.filter(regex=("sdd1.*"), axis=1).to_numpy()
    SDD2 = df.filter(regex=("sdd2.*"), axis=1).to_numpy()
    SDD3 = df.filter(regex=("sdd3.*"), axis=1).to_numpy()
    SDD4 = df.filter(regex=("sdd4.*"), axis=1).to_numpy()
    det = {'i0': IO_R, 'tey': TEY, 'pd': diode, 'sdd1': SDD1, 'sdd2': SDD2, 'sdd3': SDD3, 'sdd4':SDD4}
    
    dump = np.amax([test_beam_dump((k,v), EN)[1] for k,v in det.items()])
    abrupt = np.amax([test_abrupt_change((k,v))[1] for k,v in det.items()])
    rate = np.mean([test_detector_count_rates((k,v), sdds_range=(0, sdd_max))[1] for k,v in det.items() if k != 'i0' and k != 'pd'])

    if verbose:
        print("----------- BEAM DUMP RESULTS ----------")
        print("Likelihood of Beam-dump:", dump, '%')
        print("----------------------------------------\n")


        print("-------- ABRUPT CHANGE RESULTS ---------")
        print("Likelihood of Discontinuity:", abrupt, '%')   
        print("----------------------------------------\n")


        print("----- DETECTOR COUNT RATES RESULTS -----")
        print("Percentage of Saturated Det Points:", rate, '%')      
        print("----------------------------------------\n")
    
    return dump, abrupt, rate

def plot1d(xarr,yarr, title="Plot", labels=[]):
    """
    Description:
    -----
    Convenience function for plotting a bokeh lineplot, assumes Bokeh is already loaded.

    Args:
    -----
        xarr:  Independent array-like object, or list of array-like objects.
        yarr: Dependent array-like object, or list of array-like objects, same shape as xarr
        title: Plot title (str)
        labels:  Legend labels for multiple objects, defaults to Curve0, Curve1, etc.

    returns None
    """
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
        colors += ['purple', 'firebrick', 'red', 'orange', 'black', 'navy']
    colors = iter(colors)
    if not isinstance(xarr, list):
        xarr = [xarr]
    if not len(xarr) == len(yarr):
        yarr = [yarr]
    if not any(labels):
        labels = ["Curve" + str(i) for i,_ in enumerate(xarr)]
    for i,x in enumerate(xarr):
        fig.line(x=x, y=yarr[i], color=next(colors), legend_label=labels[i])
    fig.legend.location = "top_left"
    fig.legend.click_policy="hide"
    show(fig)


