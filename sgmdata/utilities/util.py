import h5py
import h5pyd
import numpy as np
from bokeh.io import show
from bokeh.plotting import figure
import dask.dataframe as dd

import pandas as pd
import os
import warnings

from sgmdata.utilities.scan_health import badscans
from sgmdata.utilities.analysis_reports import reports
from sgmdata.utilities.magicclass import DisplayDict

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


def preprocess(sample="", data_id=0, **kwargs) -> object:
    """
    ### Description:
    >Utility for automating the interpolation and averaging of a sample in the SGMLive website.

    ### Args:
    >**sample** *(str)*:  The name of the sample in your account that you wish to preprocess.
    or:
    >**data_id** *(int)*: The primary key of the dataset to preprocress.

    ### Keywords:
    >All of the below are optional.
    >**proposal** *(str)* -- name of proposal to limit search to.

    >**user** *(str)* -- name of user account to limit search to (for use by staff).

    >**resolution** *(float)* -- to be passed to interpolation function, this is histogram bin width.

    >**start** *(float)* --  start energy to be passed to interpolation function.

    >**stop** *(float)* -- stop energy to be passed to interpolation function.

    >**sdd_max** *(int)* -- threshold value to determine saturation in SDDs, to determine scan_health (default
                                is 105000).
    >**bscan_thresh** *(tuple)* -- (continuous, dumped, and saturated)  these are the threshold percentages from
                                    scan_health that will label a scan as 'bad'.
    >**report** *(str)* -- Analysis report type, e.g. "XAS Report".
    >**report_id** *(int)* -- primary key of report to be updated.

    ### Returns:
    >SGMQuery object if query_return is True

    ### Example Usage:
    ```python
    from sgmdata import preprocess

    preprocess(sample="TiO2", user='regiert', resolution=0.1)
    ```
    """
    from sgmdata.search import SGMQuery

    user = kwargs['user'] = kwargs.get('user', False)
    bs_args = kwargs.get('bscan_thresh', dict(cont=55, dump=30, sat=60))
    sdd_max = kwargs.get('sdd_max', 105000)
    clear = kwargs.get('clear', False)
    pk = kwargs.get("report_id", None)
    report = kwargs.get('report', 'XAS Report')
    i0 = kwargs.get('i0', 1)
    if isinstance(bs_args, tuple):
        bs_args = dict(cont=bs_args[0], dump=bs_args[1], sat=bs_args[2], sdd_max=sdd_max)
    resolution = kwargs.get('resolution', 0.1)
    kwargs.update({'resolution':resolution})
    if sample:
        sgmq = SGMQuery(sample=sample, **kwargs)
    elif data_id:
        sgmq = SGMQuery(pk=data_id, **kwargs)
        sample = next(iter(sgmq.samples.values()))
    if len(sgmq.paths):
        print("Found %d samples matching name: %s, for user: %s" % (len(sgmq.paths), sample, user))
        for k in sgmq.paths.keys():
            print(f"Processing ID: {k}")
            sgm_data = sgmq.data[k]
            print("Interpolating...", end=" ")
            interp = sgm_data.interpolate(**kwargs)
            if report:
                bs_args.update({'report': [k for k in sgm_data.scans.keys()]})
            print("Writing files.")
            if 'XAS' in report:
                bscans, bad_report = badscans(interp, **bs_args)
                score = 1.0 - len(bscans)/len(sgm_data.scans)
                sgmq.post_report(k, report, [], score, report_id=pk)
                pk = sgmq.report_ids[k]
                sgmq.write_processed(k, 'XAS')
                if len(bscans) == len(sgm_data.scans):
                    warnings.warn(f"There were no scans that passed the health check for {sample}, dataset: {k}.")
                    bscans = []
                print("Removed %d bad scan(s) from average. Averaging..." % len(bscans), end=" ")
                if any(bscans):
                    sgm_data.mean(bad_scans=bscans)
                    sgmq.write_average(k)
                else:
                    sgm_data.mean()
                    sgmq.write_average(k)
                callback = reports.get(report, False)
                if callable(callback):
                    print(f"Making report #{pk}...", end=" ")
                    js_report, fits = callback(sgm_data, sample=sample, i0=i0, bscan_report=bad_report)
                    for f in fits:
                        for i, p in enumerate(f['peaks']):
                            sgmq.create_csvs(k, ROI=(p - f['widths'][i], p + f['widths'][i]))
                    url = sgmq.post_report(k, report, js_report, score, report_id=pk)
                    html = f'<button onclick="window.open(\'{url}\',\'processed\',\'width=1000,height=700\'); ' \
                           f'return false;">Open Report for {k}</button>'
                    display(HTML(html))
                if clear:
                    clear_output()
                print(f"Averaged {len(sgm_data.scans) - len(bscans)} scans for {sample}")

            elif 'MAP' in report.upper():
                score = 1.0
                sgmq.post_report(k, report, [], score, report_id=pk)
                pk = sgmq.report_ids[k]
                sgmq.write_processed(k, 'MAP')
                callback = reports.get(report, False)
                if callable(callback):
                    js_report, fits = callback(sgm_data, sample=sample, i0=i0)
                    url = sgmq.post_report(k, report, js_report, score, report_id=pk)
                    html = f'<button onclick="window.open(\'{url}\',\'processed\',\'width=1000,height=700\'); ' \
                           f'return false;">Open Report for {k}</button>'
                    display(HTML(html))
                if clear:
                    clear_output()
    return sgmq

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

    >**proposal** *(str)* -- SGMLive proposal name, will search all if none.

    >**user** *(str)* -- SGMLive account name, defaults to current jupyterhub user.

    >**out** *(os.path / str)* -- System path to output directory for csv file(s)

    >**I0** *(pandas.DataFrame)** -- Dataframe including an incoming flux profile to be joined to the sample
                                    dataframe and included in the each CSV file.

    >**ROI** *(tuple)** --  Set the upper and lower bin number for the Region-of-Interest integration to be used in
                            reducing the dimensionality of energy MCA data.

    >**step** *(bool)* -- If the scans you're interested in are step scans, then set True to bypass the imposed interpolation.

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

    ## Are the scans step scans?
    step = kwargs.get('step', False)

    ## Get user account name.
    try:
        admin = os.environ['JHUB_ADMIN']
    except KeyError:
        admin = 0
    admin = int(admin)
    if admin:
        user = kwargs.get('user', os.environ['JUPYTERHUB_USER'])
    else:
        user = os.environ.get('JUPYTERHUB_USER', os.environ['USER'])

    ## Ensure sample name is list if singular.
    if isinstance(sample, str):
        sample = [sample]

    ## Find and collect data.
    dfs = []
    for s in sample:
        try:
            sgmq = SGMQuery(sample=s, user=user, processed=True)
        except IndexError:
            sgmq = SGMQuery(sample=s, user=user)
        for data in sgmq.data.values():
            ## get or create processed data.
            try:
                averaged = data.averaged[s]
            except AttributeError as a:
                if step:
                    for k1, d in data.scans.items():
                        for k2, e in d.items():
                            x = [(k,v) for k,v in e['independent'].items()]
                            if len(x) > 1:
                                print("CSV not available for scans with more than 1 independent axis")
                                return
                            en = dd.from_dask_array(x[0][1], columns=[x[0][0]]).groupby('en').mean()
                            df = DisplayDict()
                            for k, v in e['signals'].items():
                                if len(v.shape) == 2:
                                    columns = [k + "-" + str(i) for i in range(v.shape[1])]
                                elif len(v.shape) == 1:
                                    columns = [k]
                                else:
                                    continue
                                df[k] = dd.merge(en, dd.from_dask_array(v, columns=columns))
                            data.scans[k1][k2]['binned'] = df
                else:
                    print("Attribute Error: %s" % a)
                    data.interpolate(resolution=0.1)
                data.mean()
                averaged = data.averaged[s]
            ## extract SDDs
            df = averaged['data']['i0']
            df = df.join(averaged['data']['pd'])
            df = df.join(averaged['data']['tey'])
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
                df['i0_aux'] = i0

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


"""
Methods for extracting a spectral density plot from an SGMData file
"""
def integrate_time(clock: np.ndarray) -> np.ndarray:
    clock = np.array(clock)
    intg_clock = np.zeros(clock.shape)
    for i in range(len(clock)):
        intg_clock[i] = np.sum(clock[:i])
    return intg_clock


def calc_fft_spectrum(int_time: np.ndarray, fp: np.ndarray, acq_rate: float) -> tuple:
    duration = int(np.round_(np.max(int_time)))
    time_bins = np.linspace(0, duration, acq_rate * duration)
    binned_sig = np.interp(time_bins, int_time, fp)
    inv_sig = np.fft.fft(binned_sig)
    n = np.arange(len(inv_sig))
    T = len(inv_sig) // acq_rate
    freq = n / T
    return freq, np.abs(inv_sig) / len(n)


def get_fft_spectra(data: object, channel='i0', acq_rate=20) -> None:
    """
    ### Description:
    Use clock signal in SGM data, with any scalar signal to produce a spectral density graph.

    ### Args:
    >**data** *(SGMData)* --  SGMData object

    ### Keywords:
    >**channel** *(str)* -- Scalar signal name in SGMData object / file.

    >**acq_rate** *(int)* -- Data needs to be evenly spaced, integer value of approximate acquisition rate in Hz will
                            is used to set a bin size.

    ### Returns:
    >**None**
    """
    x = []
    y = []
    names = []
    for f, scans in data.scans.items():
        for e, entry in scans.items():
            clock = entry['signals']['clock']
            if hasattr(clock, 'compute'):
                clock = clock.compute()
            intg_clock = integrate_time(clock)
            chan = entry['signals'][channel]
            if hasattr(chan, 'compute'):
                chan = chan.compute()
            freq, spectrum = calc_fft_spectrum(intg_clock, chan, acq_rate)
            x.append(freq)
            y.append(spectrum)
            names.append(f"{f} - {e}")
    plot1d(x, y, labels=names)