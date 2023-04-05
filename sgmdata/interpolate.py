import numpy as np
from dask import delayed
import dask.array as da
import dask.dataframe as dd
import pandas as pd
import warnings
from dask.distributed import get_client
from .utilities.magicclass import DisplayDict
from sys import getsizeof

def label_bins(bins, bin_edges, independent):
    """
        Program creates an array the length of an independent axis and fills it
        with bin center values that the independent array falls into.
    """
    bin_labels = {}
    val = independent
    for i, key in enumerate(val.keys()):
        bin_labels[key] = np.full(val[key].shape, np.nan)
        indep_value = val[key]
        for j, b in enumerate(bins[i]):
            bin_labels[key][np.where(
                np.logical_and(indep_value >= bin_edges[i][j], indep_value <= bin_edges[i][j + 1]))] = b
    axes = np.squeeze(np.vstack([v for k, v in bin_labels.items()]).T)
    columns = {k: axes if len(axes.shape) == 1 else axes[:, i] for i, k in enumerate(bin_labels.keys())}
    return pd.DataFrame.from_dict(columns)


def make_df(independent, signals, labels, npartitions=1):
    c = [k for k, v in independent.items()]
    df = dd.from_delayed(labels).persist()
    dfs = DisplayDict()
    for k, v in signals.items():
        if len(v.shape) == 2:
            columns = [k + "-" + str(i) for i in range(v.shape[1])]
        elif len(v.shape) == 1:
            columns = [k]
        else:
            continue
        dfs[k] = df.merge(dd.from_dask_array(v, columns=columns)).groupby(c).mean(split_out=npartitions)
    return dfs

def dask_max(value, sig_digits=2):
    res = 10**sig_digits
    if hasattr(value, 'compute'):
        return int(np.unique(np.floor(value.compute() * res)).shape[0] / 1.1)
    else:
        return int(np.unique(np.floor(value * res)).shape[0] / 1.1)

def dask_unique(value):
    if hasattr(value, 'compute'):
        return int(np.unique(value.compute()).shape[0] / 5)
    else:
        return int(np.unique(value).shape[0] / 5)

def compute_df(dfs, df_idx, ldim=1, method = 'nearest'):
    df_dict = DisplayDict()
    if ldim == 1:
        df_dict.update({k: dd.merge(df_idx,
                        df,
                        how='left',
                        left_index=True,
                        right_index=True
                       ).compute()
                        .sort_index(level=0, sort_remaining=True)
                        .interpolate() for k, df in dfs.items()})
    elif ldim == 2:
        df_dict.update({k: dd.merge(df_idx,
                        df,
                        how='left',
                        left_index=True,
                        right_index=True
                       ).compute()
                        .sort_index(level=0, sort_remaining=True)
                        .interpolate(method=method)
                        .fillna(0) for k, df in dfs.items()})
    return df_dict

def shift_cmesh(x, shift=0.5):
    return shift * (x + np.roll(x, -1))

def interpolate(independent, signals, command=None, **kwargs):
    """
    ### Description:
    >Creates the bins required for each independent axes to be histogrammed into for interpolation,
     then uses dask dataframe groupby commands to perform a linear interpolation.

    ### Args:
    >**independent** *(dict)* -- Dictionary of independent axes from SGMScan.entry

    >**signals** *(dict)* -- Dictionary of signals from SGMScan.entry

    ### Keywords:
    >**start** *(list or number)* -- starting position of the new array

    >**stop**  *(list or number)* -- ending position of the new array

    >**bins** *(list of numbers or arrays)* --  this can be an array of bin values for each axes,
                                                  or can be the number of bins desired.

    >**resolution** *(list or number)* -- used instead of bins to define the bin to bin distance.

    >**sig_digits** *(int)* -- used to overide the default uncertainty of the interpolation axis of 2 (e.g. 0.01)
    """
    compute = kwargs.get('compute', True)
    method = kwargs.get('method', 'nearest')
    accuracy = kwargs.get('sig_digits', 2)
    axis = independent
    client = get_client()
    dim = len(axis.keys())
    if 'start' not in kwargs.keys():
        if command:
            if 'scan' in command[0]:
                start = [round(float(command[2]))]
            elif 'mesh' in command[0]:
                xstart = min((float(command[2]), float(command[3])))
                ystart = min((float(command[6]), float(command[7])))
                start = [xstart, ystart]
        else:
            start = [round(v.min()) for k, v in axis.items()]
    else:
        start = kwargs['start']
    if 'stop' not in kwargs.keys():
        if command:
            if 'scan' in command[0]:
                stop = [round(float(command[3]))]
            elif 'mesh' in command[0]:
                xstop = max((float(command[2]), float(command[3])))
                ystop = max((float(command[6]), float(command[7])))
                stop = [xstop, ystop]
        else:
            stop = [round(v.max()) + 1 for k, v in axis.items()]
    else:
        stop = kwargs['stop']
    if not isinstance(start, list):
        start = [start for i, _ in enumerate(axis.keys())]
    if not isinstance(stop, list):
        stop = [stop for i, _ in enumerate(axis.keys())]
    if len(start) != len(stop):
        raise ValueError("Start and Stop coordinates must have same length")
    if 'resolution' in kwargs.keys() and 'bins' in kwargs.keys():
        raise KeyError("You can only use the keyword bins, or resolution not both")
    if 'resolution' not in kwargs.keys() and 'bins' not in kwargs.keys():
        if command and len(axis) > 1:
            xrange = (float(command[2]), float(command[3]))
            yrange = (float(command[6]), float(command[7]))
            dx = abs(xrange[0] - xrange[1]) / (int(command[4]) * 15)
            dy = abs(yrange[0] - yrange[1]) / (int(command[-1]))
            resolution = [dx, dy]
            bin_num = [int(abs(stop[i] - start[i]) / resolution[i]) for i, _ in enumerate(axis.keys())]
            offset = [item / 2 for item in resolution]
            bins = [np.linspace(start[i], stop[i], bin_num[i], endpoint=True) for i in range(len(bin_num))]
        elif command:
            xrange = (float(command[2]), float(command[3]))
            dx = round(abs(xrange[0] - xrange[1]) / (int(command[4]) * 8.33333333), 2)  # About 0.1eV for standard XAFS
            resolution = [dx]
            offset = [item / 2 for item in resolution]
            bin_num = [int(abs(stop[i] - start[i]) / resolution[i]) for i, _ in enumerate(axis.keys())]
            bins = [np.linspace(start[i], stop[i], bin_num[i], endpoint=True) for i in range(len(bin_num))]
        else:
            bin_num = [dask_unique(v) for k, v in axis.items()]
            offset = [abs(stop[i] - start[i]) / (2 * bin_num[i]) for i in range(len(bin_num))]
            bins = [np.linspace(start[i], stop[i], bin_num[i], endpoint=True) for i in range(len(bin_num))]
    elif 'resolution' in kwargs.keys():
        resolution = kwargs['resolution']
        if not isinstance(kwargs['resolution'], list):
            resolution = [resolution for i, _ in enumerate(axis.keys())]
        max_res = [dask_max(v, sig_digits=accuracy) for k, v in axis.items()]
        bin_num = [int(abs(stop[i] - start[i]) / resolution[i]) for i, _ in enumerate(axis.keys())]
        for i, l in enumerate(max_res):
            if l < bin_num[i] and l > 0:
                warnings.warn(
                    "Resolution setting can't be higher than experimental resolution, setting resolution for axis %s to %f" % (
                        i, abs(stop[i] - start[i]) / l), UserWarning)
                bin_num[i] = l
        offset = [item / 2 for item in resolution]
        bins = [np.linspace(start[i], stop[i], bin_num[i], endpoint=True) for i in range(len(bin_num))]
    elif 'bins' in kwargs.keys():
        if isinstance(kwargs['bins'], list):
            if len(kwargs['bins'][0]) == 1:
                bin_num = [np.floor(len(np.unique(axis[k])) / kwargs['bins'][i]) for i, k in
                           enumerate(axis.keys())]
                bins = [np.linspace(start[i], stop[i], bin_num[i], endpoint=True, dtype=np.float32) for i in range(len(bin_num))]
            else:
                start = [item[0] for item in kwargs['bins']]
                stop = [item[1] for item in kwargs['bins']]
                bin_num = []
        elif isinstance(kwargs['bins'], int):
            bin_num = [int(len(axis[k]) / kwargs['bins']) for i, k in enumerate(axis.keys())]
        else:
            raise ValueError("Bins can only be int or list")
        offset = [abs(stop[i] - start[i]) / (2 * bin_num[i]) for i in range(len(bin_num))]
    else:
        raise ValueError("Not enough information to set evenly spaced grid for interpolation.")

    bin_edges = [np.linspace(start[i] - offset[i], stop[i] + offset[i], bin_num[i] + 1, endpoint=True) for i in
                 range(len(bin_num))]
    labels = delayed(label_bins)(bins, bin_edges, independent)
    npartitions = kwargs.get('npartitions', 3 if len(bin_num) > 1 else 1)
    dfs = make_df(independent, signals, labels, npartitions=npartitions)
    nm = [k for k, v in independent.items()]
    if len(nm) == 1:
        df_idx = dd.from_pandas(pd.DataFrame({nm[0]: bins[0]}), npartitions=npartitions).groupby(nm[0]).mean()
    elif len(nm) == 2:
        _y = np.array([bins[1] for b in bins[0]]).flatten()
        _x = np.array([[bins[0][j] for i in range(len(bins[1]))] for j in range(len(bins[0]))]).flatten()
        array = [_x, _y]
        d = {nm[0]: _x, nm[1]: _y}
        df_idx = dd.from_pandas(pd.DataFrame(d), npartitions=npartitions).groupby(nm).mean()
    else:
        raise ValueError("Too many independent axis for interpolation")
    if compute:
        try:
            dfs = compute_df(dfs, df_idx, ldim=len(nm), method=method)
        except Exception as e:
            print("Trouble computing dataframe, error msg: %s" % e)
            return None, None
    return dfs, df_idx
