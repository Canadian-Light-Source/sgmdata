import numpy as np
from dask import delayed
import dask.array as da
import dask.dataframe as dd
import pandas as pd
import warnings
from dask.diagnostics import ProgressBar

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


def make_df(independent, signals, labels):
    c = [k for k, v in independent.items()]
    df = dd.from_delayed(labels)
    for k, v in signals.items():
        if len(v.shape) == 2:
            columns = [k + "-" + str(i) for i in range(v.shape[1])]
        elif len(v.shape) == 1:
            columns = [k]
        else:
            continue
        df = df.merge(dd.from_dask_array(v, columns=columns))
    return df.groupby(c).mean()

def dask_max(value):
    if hasattr(value, 'compute'):
        return int(np.unique(np.floor(value.compute() * 100)).shape[0] / 1.1)
    else:
        return int(np.unique(np.floor(value * 100)).shape[0] / 1.1)

def dask_unique(value):
    if hasattr(value, 'compute'):
        return int(np.unique(value.compute()).shape[0] / 5)
    else:
        return int(np.unique(value).shape[0] / 5)

def compute_df(df, idx, method = 'nearest'):
    if len(idx.shape) == 1:
        return df.compute().reindex(idx).interpolate()
    elif len(idx.shape) == 2:
        return df.compute().unstack().interpolate(method=method).fillna(0).stack().reindex(idx)

def interpolate(independent, signals, command=None, **kwargs):
    """
        Creates the bins required for each independent axes to be histogrammed into for interpolation,
        then uses dask dataframe groupby commands to perform a linear interpolation.
        Optional Keywords:
                   start (type: list or number) -- starting position of the new array
                   stop  (type: list or number) -- ending position of the new array
                   bins (type: list of numbers or arrays) --  this can be an array of bin values for each axes,
                                                              or can be the number of bins desired.
                   resolution (type: list or number) -- used instead of bins to define the bin to bin distance.
    """
    compute = kwargs.get('compute', True)
    method = kwargs.get('method', 'nearest')
    npartitions = kwargs.get('npartitions', 3)
    axis = independent
    dim = len(axis.keys())
    if 'start' not in kwargs.keys():
        if command:
            if 'scan' in command[0]:
                start = [round(float(command[2]))]
            elif 'mesh' in command[0]:
                start = [float(command[2]), float(command[6])]
        else:
            start = [round(v.min()) for k, v in axis.items()]
    else:
        start = kwargs['start']
    if 'stop' not in kwargs.keys():
        if command:
            if 'scan' in command[0]:
                stop = [round(float(command[3]))]
            elif 'mesh' in command[0]:
                stop = [float(command[3]), float(command[7])]
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
        bin_num = [dask_unique(v) for k, v in axis.items()]
        offset = [abs(stop[i] - start[i]) / (2 * bin_num[i]) for i in range(len(bin_num))]
        bins = [np.linspace(start[i], stop[i], bin_num[i], endpoint=True) for i in range(len(bin_num))]
    elif 'resolution' in kwargs.keys():
        resolution = kwargs['resolution']
        if not isinstance(kwargs['resolution'], list):
            resolution = [resolution for i, _ in enumerate(axis.keys())]
        max_res = [dask_max(v)  for k, v in axis.items() ]
        bin_num = [int(abs(stop[i] - start[i]) / resolution[i]) for i, _ in enumerate(axis.keys())]
        for i, l in enumerate(max_res):
            if l < bin_num[i]:
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
                bins = [np.linspace(start[i], stop[i], bin_num[i], endpoint=True) for i in range(len(bin_num))]
            else:
                start = [item[0] for item in kwargs['bins']]
                stop = [item[1] for item in kwargs['bins']]
                bin_num = []
        elif isinstance(kwargs['bins'], int):
            bin_num = [int(len(axis[k]) / kwargs['bins']) for i, k in enumerate(axis.keys())]
    bin_edges = [np.linspace(start[i] - offset[i], stop[i] + offset[i], bin_num[i] + 1, endpoint=True) for i in
                 range(len(bin_num))]
    labels = delayed(label_bins)(bins, bin_edges, independent)
    df = make_df(independent, signals, labels)
    nm = [k for k, v in independent.items()]
    if len(nm) == 1:
        idx = pd.Index(bins[0], name=nm[0])
    elif len(nm) == 2:
        _y = np.array([bins[1] for b in bins[0]]).flatten()
        _x = np.array([[bins[0][j] for i in range(len(bins[1]))] for j in range(len(bins[0]))]).flatten()
        array = [_x, _y]
        idx = pd.MultiIndex.from_tuples(list(zip(*array)), names=nm)
    else:
        raise ValueError("Too many independent axis for interpolation")
    if compute:
        try:
            df = compute_df(df, idx, method=method)
        except Exception as e:
            print("Trouble computing dataframe, error msg: %s" % e)
            return None, None
    return df, idx


