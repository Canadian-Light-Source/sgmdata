import h5py
import h5pyd
import numpy as np
from bokeh.io import show
from bokeh.plotting import figure

import pandas as pd
import os
import warnings

try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
        from tqdm.notebook import tqdm  # Jupyter notebook or qtconsole
    else:
        from tqdm import tqdm  # Other type (?)
except NameError:
    from tqdm import tqdm

try:
    from IPython.display import display, HTML, clear_output
except ImportError:
    pass




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
    ### Description:
    >A function to output the data-tree from an hdf5 file object.

    ### Args:
    >**h5** *(h5py.File)* -- Any H5File object, from h5py.

    ### Returns:
    >**None**

    ### Example Usage:
    ```python
    from sgmdata.utilities import h5tree
    import h5py

    f = h5py.File("Filename.nxs", 'r')
    h5tree(f)
    ```
    """
    h5.visititems(printTree)
    
    
#Scan health functions (from Habib)

def get_moving_average(data, window_size=4):
    """ 
    ### Description:
    A function to calculate the moving average of data using numpy's implementation of convolution
    
    ### Args:
    > **data** *(numpy.ndarray)* -- A 1d numpy array of the data on which to calculate the moving average
        window_size (int): An integer value of the number of samples to consider when averaging 
        
    ## Returns:
    > **m_average** *(numpy.ndarray)* -- A 1d numpy array of the calculated moving average. The size of
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
    ### Description:
    A function to detect the percentage of abrupt change in a single detector data
    
    ### Args:
    >**detector** *(tuple)*: A python tuple in the form (detector_name, data). The detector_name is a
                          string, while data is a numpy array of the data to detect abrupt change

    >**sigma** *(float)*: A float value for standard deviation. This number define how different a specific
                       count should be from the standard deviation to be considered abrupt change 

    >**tolerance** *(float)*: A float value specifying the absolute tolerance parameter for detecting if
                           two numbers should be considered close to each other
        
    ### Returns:
    >**str**: Percentage of the data that is normal count and the percentage the fuction think might be abrupt change
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
    ### Description:
    >A function to detect if count rates of a specific detector are within a defined range
    
    ### Args:
    >**detector** *(tuple)*: A python tuple in the form (detector_name, data). The detector_name is a
                          string, while data is a numpy array of the data to detect the count rates 

    >**tey_range** *(tuple)*: A python tuple defining the normal count range for the tey detector
                           in the form (min_normal_count, max_normal_count)

    >**io_range** *(tuple)* -- A python tuple defining the normal count range for the io detector
                          in the form (min_normal_count, max_normal_count)

    >**sdds_range** *(tuple)* --  A python tuple defining the normal count range for the sdd[1-4]
                            detectors in the form (min_normal_count, max_normal_count)
                           
    ### Returns:
    >**str** -- Percentage of the data that is within normal count range and the percentage
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
    ### Description:
    >A function to detect the percentage of beam dump in a single detector data
    
    ### Args:
    >**detector** *(tuple)* -- A python tuple in the form (detector_name, data).
                          The detector_name is a string, while data is a numpy 
                          array of the data to detect beam dump

    >**indep** *(numpy.ndarray)* -- A numpy array of the independent variable data
        
    ### Returns:
    >*str* -- Percentage of the data that is normal count and the percentage
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



def scan_health(df, verbose=False, sdd_max=105000, length=None):
    """
    ### Description:
    >Function takes in a interpolated scan (a pandas DataFrame), and returns the overall health.

    ### Args:
    >**df** *(DataFrame)* --  pandas dataframe from SGMScan.binned.

    >**verbose** *(bool)* -- Explain the returned output in plain text.

    >**sdd_max** *(int)* -- 105000 (default) - saturation value for total SDD counts/s

    ### Returns:
    >(tuple):  (Discontiunty %,  Beam-dump %,  Saturation %)
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
    if length:
        if abs(len(EN) - length) > 5:
            abrupt = 100
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



def badscans(interp, **kwargs):
    """
    ### Description:
    >Batch calculation of sgmdata.utilities.scan_health for list of interpolated dataframes.

    ### Args:
    >interp (list) :  list of SGMScan binned dataframes.

    ### Returns:
    >List of indexes for bad scans in interp.

    """
    cont = kwargs.get('cont', 55)
    dump = kwargs.get('dump', 30)
    sat = kwargs.get('sat', 60)
    sdd_max = kwargs.get('sdd_max', 50000)
    length = np.bincount([len(i) for i in interp if i is not None]).argmax()
    bad_scans = []
    health = [scan_health(i, sdd_max=sdd_max, length=length) for i in interp]
    pbar = tqdm(health)
    for i,t in enumerate(pbar):
        pbar.set_description("Finding bad scans...")
        if t[0] > cont or t[1] > dump or t[2] > sat:
            print(i, t)
            bad_scans.append(i)
    return bad_scans


def preprocess(sample, **kwargs):
    """
    ### Description:
    >Utility for automating the interpolation and averaging of a sample in the SGMLive website.

    ### Args:
    >**sample** *(str)*:  The name of the sample in your account that you wish to preprocess.

    ### Keywords:
    >All of the below are optional.

    >**user** *(str)* -- name of user account to limit search to (for use by staff).

    >**resolution** *(float)* -- to be passed to interpolation function, this is histogram bin width.

    >**start** *(float)* --  start energy to be passed to interpolation function.

    >**stop** *(float)* -- stop energy to be passed to interpolation function.

    >**sdd_max** *(int)* -- threshold value to determine saturation in SDDs, to determine scan_health (default
                                is 105000).
    >**bscan_thresh** *(tuple)* -- (continuous, dumped, and saturated)  these are the threshold percentages from
                                    scan_health that will label a scan as 'bad'.

    ### Returns:
    >(HTML) hyperlink for preprocessed data stored in SGMLive

    ### Example Usage:
    ```python
    from sgmdata import preprocess

    preprocess(sample="TiO2", user='regiert', resolution=0.1)
    ```
    """
    from sgmdata.search import SGMQuery
    from sgmdata.load import SGMData

    user = kwargs['user'] = kwargs.get('user', False)
    bs_args = kwargs.get('bscan_thresh', dict(cont=55, dump=30, sat=60))
    sdd_max = kwargs.get('sdd_max', 105000)
    clear = kwargs.get('clear', True)
    query_return = kwargs.get('query', False)
    client = kwargs.get('client', False)
    if isinstance(bs_args, tuple):
        bs_args = dict(cont=bs_args[0], dump=bs_args[1], sat=bs_args[2], sdd_max=sdd_max)
    resolution = kwargs.get('resolution', 0.1)
    kwargs.update({'resolution':resolution})
    if user:
        sgmq = SGMQuery(sample=sample, data=False, **kwargs)
    else:
        sgmq = SGMQuery(sample=sample, data=False, **kwargs)
    if len(sgmq.paths):
        print("Found %d scans matching sample: %s, for user: %s" % (len(sgmq.paths), sample, user))
        sgm_data = SGMData(sgmq.paths, **kwargs)
        print("Interpolating...", end=" ")
        interp = sgm_data.interpolate(**kwargs)
        sgmq.write_proc(sgm_data.scans)
        bscans = badscans(interp, **bs_args)
        if len(bscans) != len(sgm_data.scans):
            print("Removed %d bad scan(s) from average. Averaging..." % len(bscans), end=" ")
            if any(bscans):
                sgm_data.mean(bad_scans=bscans)
                _, http = sgmq.write_avg(sgm_data.averaged, bad_scans=bscans)
            else:
                sgm_data.mean()
                _, http = sgmq.write_avg(sgm_data.averaged)

            html = "\n".join([
                                 '<button onclick="window.open(\'%s\',\'processed\',\'width=1000,height=700\'); return false;">Open %s</button>' % (
                                 l, sgmq.sample) for i, l in enumerate(http)])
            if clear:
                clear_output()
            if client:
                client.restart()
            print(f"Averaged {len(sgm_data.scans) - len(bscans)} scans for {sample}")
            del sgm_data
            if query_return:
                return sgmq
            return HTML(html)
        else:
            if clear:
                clear_output()
            warnings.warn(f"There were no scans that passed the health check for {sample}.")


def sumROI(arr, start, stop):
    return np.nansum(arr[:, start:stop], axis=1)

def create_csv(sample, mcas=None, **kwargs):
    """
    ### Description:
    >Make CSV file from sample(s)

    ### Args:
    >**sample** *(str or list(str))*  -- Sample(s) name(s) from SGMLive that you want to process.

    ### Keywords:
    >**mcas** *(list(str))* -- list of detector names for which the ROI summation should take place.

    >**user** *(str)* -- SGMLive account name, defaults to current jupyterhub user.

    >**out** *(os.path / str)* -- System path to output directory for csv file(s)

    >**I0** *(pandas.DataFrame)** -- Dataframe including an incoming flux profile to be joined to the sample
                                    dataframe and included in the each CSV file.

    >**ROI** *(tuple)** --  Set the upper and lower bin number for the Region-of-Interest integration to be used in
                            reducing the dimensionality of energy MCA data.

    ### Returns:
    >**list(pd.DataFrame)** -- list of dataframes created.
    """
    from slugify import slugify
    from sgmdata.search import SGMQuery

    ## Set default detector list for ROI summing.
    if mcas is None:
        mcas = ['sdd1', 'sdd2', 'sdd3', 'sdd4']

    ## Prepare data output directory.
    out = kwargs.get('out', './data_out/')
    if not os.path.exists(out):
        os.makedirs(out)

    ## Load in I0 if exists:
    i0 = kwargs.get('I0', None)

    ## Get ROI bounds:
    roi = kwargs.get('ROI', (0, 255))

    ## Get user account name.
    try:
        admin = os.environ['JHUB_ADMIN']
    except KeyError:
        raise Exception("SGMQuery can only be run inside sgm-hub.lightsource.ca at the moment.")
    admin = int(admin)
    if admin:
        user = kwargs.get('user', os.environ['JUPYTERHUB_USER'])
    else:
        user = os.environ['JUPYTERHUB_USER']

    ## Ensure sample name is list if singular.
    if isinstance(sample, str):
        sample = [sample]

    ## Find and collect data.
    dfs = []
    for s in sample:
        sgmq = SGMQuery(sample=s, user=user, processed=True)
        data = sgmq.data
        ## get or create processed data.
        try:
            averaged = data.averaged[s]
        except AttributeError as a:
            print("Attribute Error: %s" % a)
            data.interpolate(resolution=0.1)
            data.mean()
            averaged = data.averaged[s]
        ## extract SDDs
        df = averaged['data']
        sdd_tot = []
        for det in mcas:
            mca = averaged.get_arr(det)
            temp = sumROI(mca, start=roi[0], stop=roi[1])
            df.drop(columns=list(df.filter(regex=det+".*")), inplace=True)
            df[det] = temp
            sdd_tot.append(temp)
        ## Should this be averaged?
        df['sdd_total'] = np.nansum(sdd_tot, axis=0)
        if isinstance(i0, pd.DataFrame):
            df = df.join(i0)
        elif isinstance(i0, pd.Series):
            df['i0'] = i0
        df.to_csv(out + '/' + slugify(s) + f'_ROI-{roi[0]}_{roi[1]}.csv')
        dfs.append(df)
    return dfs


def plot1d(xarr,yarr, title="Plot", labels=[]):
    """
    ### Description:
    >Convenience function for plotting a bokeh lineplot, assumes Bokeh is already loaded.

    ### Args:
    >**xarr** *(array-like)* --  Independent array-like object, or list of array-like objects.

    >**yarr** *(array-like)* -- Dependent array-like object, or list of array-like objects, same shape as xarr

    >**title** *(str)* -- Plot title

    >**labels** *(list(str))* --  Legend labels for multiple objects, defaults to Curve0, Curve1, etc.

    ### Returns:
    >**None**
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