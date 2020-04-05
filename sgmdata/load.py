import os
import h5py
import h5pyd
from dask import compute, delayed
import dask.array as da
import dask.dataframe as dd
from dask.distributed import Client
import pandas as pd
import numpy as np
from multiprocessing.pool import ThreadPool
from functools import partial
from .plots import eemscan
import warnings

try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
        from tqdm.notebook import tqdm  # Jupyter notebook or qtconsole
    else:
        from tqdm import tqdm  # Other type (?)
except NameError:
    from tqdm import tqdm



class SGMScan(object):
    """
        Data class for storing dask arrays for SGM data files that have been grouped into 'NXentry',
        and then divided into signals, independent axes, and other data.  Contains convenience classes
        for interpolation.
    """

    class DataDict(dict):
        def __getattr__(self, name):
            return self[name]

        def __setattr__(self, name, value):
            self[name] = value

        @staticmethod
        def label_bins(bins, bin_edges, independent, npartitions):
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
            axes = np.vstack([v for k, v in bin_labels.items()]).T
            bin_l_dask = da.from_array(axes, chunks='auto')
            columns = [k for k, v in bin_labels.items()]
            return dd.from_dask_array(bin_l_dask, columns=columns).compute()

        def make_df(self, labels=None):
            df = dd.from_delayed(labels)
            c = [k for k, v in self['independent'].items()]
            for k, v in self['signals'].items():
                if len(v.shape) == 2:
                    columns = [k + "-" + str(i) for i in range(v.shape[1])]
                elif len(v.shape) == 1:
                    columns = [k]
                else:
                    continue
                df = df.merge(dd.from_dask_array(v, columns=columns))

            self.__setattr__('dataframe', {"binned": df.groupby(c).mean()})
            return df.groupby(c).mean()

        def interpolate(self, **kwargs):
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
            if 'compute' in kwargs.keys():
                compute = kwargs['compute']
            else:
                compute = True
            axis = self['independent']
            dim = len(axis.keys())
            if 'start' not in kwargs.keys():
                start = [int(v.min()) for k, v in axis.items()]
            else:
                start = kwargs['start']
            if 'stop' not in kwargs.keys():
                stop = [int(v.max()) + 1 for k, v in axis.items()]
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
                bin_num = [int(np.unique(v.compute()).shape[0] / 5) for k, v in axis.items()]
                offset = [abs(stop[i] - start[i]) / (2 * bin_num[i]) for i in range(len(bin_num))]
                bins = [np.linspace(start[i], stop[i], bin_num[i], endpoint=True) for i in range(len(bin_num))]
            elif 'resolution' in kwargs.keys():
                resolution = kwargs['resolution']
                if not isinstance(kwargs['resolution'], list):
                    resolution = [resolution for i, _ in enumerate(axis.keys())]
                max_res = [int(np.unique(np.floor(v.compute() * 100)).shape[0] / 1.1) for k, v in axis.items()]
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
            self.__setattr__("new_axes", {"values": bins, "edges": bin_edges})
            labels = delayed(self.label_bins)(bins, bin_edges, self['independent'], self.npartitions)
            df = self.make_df(labels)
            if compute:
                nm = [k for k, v in self['independent'].items()]
                if len(nm) == 1:
                    idx = pd.Index(bins[0], name=nm[0])
                    df = df.compute().reindex(idx).interpolate()
                    self.__setattr__('binned', {"dataframe": df})
                    return df
                elif len(nm) == 2:
                    _y = np.array([bins[1] for b in bins[0]]).flatten()
                    _x = np.array([[bins[0][j] for i in range(len(bins[1]))] for j in range(len(bins[0]))]).flatten()
                    array = [_x, _y]
                    idx = pd.MultiIndex.from_tuples(list(zip(*array)), names=nm)
                    #This method works for now, but can take a fair amount of time.
                    df = df.compute().unstack().interpolate(method='nearest').fillna(0).stack().reindex(idx)
                    binned = {"dataframe": df}
                    self.__setattr__('binned', binned)
                    return df
                else:
                    raise ValueError("Too many independent axis for interpolation")

            else:
                return df

        def __repr__(self):
            represent = ""
            for key in self.keys():
                represent += f"\t {key}:\n\t\t\t"
                val = self[key]
                if isinstance(val, dict):
                    for k in val.keys():
                        if hasattr(val[k], 'shape') and hasattr(val[k], 'dtype'):
                            represent += f"{k} : array(shape:{val[k].shape}, type:{val[k].dtype}), \n\t\t\t"
                        else:
                            represent += f"{k} : {val[k]},\n\t\t\t"
                    represent += "\n\t"
                else:
                    represent += f"{val} \n\t"
            return represent

        def _repr_html_(self):
            entry = [
                "<td>",
                str(self.sample),
                "</td>",
                "<td>",
                str(self.command),
                "</td>",
                "<td>",
                str(list(self.independent.keys())),
                "</td>",
                "<td>",
                str(list(self.signals.keys())),
                "</td>",
                "<td>",
                str(list(self.other.keys())),
                "</td>",
            ]
            return " ".join(entry)

        def plot(self):
            """
            Determines the appropriate plot based on independent axis number and name
            """

            dim = len(self.independent)
            if dim == 1 and 'en' in self.independent.keys():
                keys = eemscan.required
                if 'dataframe' in self.__dict__.keys():
                    if 'binned' in self['dataframe'].keys():
                        df = self['dataframe']['binned'].compute()
                        data = {k: df.fitler(regex=("%s.*" % k), axis=1).to_numpy().T for k in keys}
                        eemscan.plot(**data)
                else:
                    ds = int(self.independent['en'].shape[0] / 1000) + 1
                    data = {k: self.signals[k][::ds].T.compute() for k in self.signals.keys() if k in keys}
                    data.update(
                        {k: self.independent[k][::ds].T.compute() for k in self.independent.keys() if k in keys})
                    data.update({k: self.other[k].compute().T for k in self.other.keys() if k in keys})
                    if 'image' in keys:
                        data.update({'image': self.signals['sdd1'][::ds].T.compute()})
                    eemscan.plot(**data)

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        for key, value in kwargs.items():
            self.__dict__[key] = SGMScan.DataDict(value)

    def __repr__(self):
        represent = ""
        for key in self.__dict__.keys():
            represent += f"\n Entry: {key},\n\t Type: {self.__dict__[key]}"
        return represent

    def _repr_html_(self):
        table = [
            "<table>",
            "  <thead>",
            "    <tr><td> </td><th>Sample</th><th> Command </th><th> Independent </th><th> Signals </th><th> Other </th></tr>",
            "  </thead>",
            "  <tbody>",
        ]
        for key in self.__dict__.keys():
            table.append(f"<tr><th> {key}</th>" + self.__dict__[key]._repr_html_() + "</tr>")
        table.append("</tbody></table>")

        return "\n".join(table)

    def __getitem__(self, item):
        return self.__dict__[item]


class SGMData(object):
    """
        Class for loading in data from h5py or h5pyd files for raw SGM data.
        To substantiate pass the class pass a single (or list of) system file paths
        (or hsds path).  e.g. data = SGMData('/path/to/my/file.nxs') or SGMData(['1.h5', '2.h5'])

        Optional Keywords:  npartitions (type: integer) -- choose how many divisions (threads)
                                                           to split the file data arrays into.
                            scheduler (type: str) -- use dask cluster for operations, e.g. 'dscheduler:8786'
                            axes (type: list(str)) -- names of the axes to use as independent axis and ignore
                                                    spec command issued
    """

    def __init__(self, files, **kwargs):
        self.__dict__.update(kwargs)
        if not isinstance(files, list):
            files = [files]
        if not hasattr(self, 'npartitions'):
            self.npartitions = 3
        if not hasattr(self, 'threads'):
            self.threads = 4
        files = [os.path.abspath(file) for file in files]
        self.scans = {k.split('/')[-1].split(".")[0] : [] for k in files}
        self.interp_params = {}
        with ThreadPool(self.threads) as pool:
                L = list(tqdm(pool.imap_unordered(self._load_data, files), total=len(files)))
        self.scans.update({k:SGMScan(**v) for d in L for k,v in d.items()})
        self.entries = self.scans.items

    def _find_data(self, node, indep=None, other=False):
        data = {}

        def search(name, node):
            if indep:
                if isinstance(node, h5pyd.Dataset) or isinstance(node, h5py.Dataset):
                    if "S" not in str(node.dtype).upper() and node.shape and node.shape[0] > 1:
                        d_name = name.split('/')[-1]
                        l = [True for axis in indep if axis in d_name]
                        if any(l) and not other:
                            data.update({d_name: node})
                        elif other and not any(l):
                            data.update({d_name: node})

        node.visititems(search)
        return data

    def _load_data(self, file):
        """
            Function loads data in from SGM data file, and using the command value groups the data as
            either independent, signal or other.
        """
        entries = {}
        # Try to open the file locally or from a url provided.
        try:
            h5 = h5pyd.File(file, 'r')
        except Exception as e:
            if os.path.exists(file):
                h5 = h5py.File(file, 'r')
            else:
                raise Exception(e)
                return
        file_root = file.split("/")[-1].split(".")[0]
        # Find the number of scans within the file
        NXentries = [str(x) for x in h5['/'].keys() if 'NXentry' in str(h5[x].attrs.get('NX_class'))]
        # Get the commands used to declare the above scans
        independent = []
        if not hasattr(self, 'axes'):
            try:
                if hasattr(self, 'scan_type'):
                    commands = [
                        str(h5[entry + '/command'][()]).split() if isinstance(h5[entry + '/command'][()], str) else str(
                            h5[entry + '/command'][()], 'utf-8').split()[:-1] for entry in NXentries if
                        str(h5[entry + '/command'][()]).split()[0] == self.scan_type]
                    allowed = [i for i, entry in enumerate(NXentries) if
                               str(h5[entry + '/command'][()]).split()[0] == self.scan_type]
                    NXentries = [entry for i, entry in enumerate(NXentries) if i in allowed]
                else:
                    commands = [
                        str(h5[entry + '/command'][()]).split() if isinstance(h5[entry + '/command'][()], str) else str(
                            h5[entry + '/command'][()], 'utf-8').split()[:-1] for entry in NXentries]
            except:
                raise KeyError(
                    "Scan entry didn't have a 'command' string saved. Command load can be skipped by providing a list of independent axis names")
            for i, command in enumerate(commands):
                if 'mesh' in command[0]:
                    independent.append((command[1], command[5]))
                elif 'scan' in command[0]:
                    independent.append((command[1],))
        else:
            for i, entry in enumerate(NXentries):
                independent.append(tuple(self.axes))
        if not independent:
            return {}
        indep = [self._find_data(h5[entry], independent[i]) for i, entry in enumerate(NXentries)]
        # search for data that is not an array mentioned in the command
        data = [self._find_data(h5[entry], independent[i], other=True) for i, entry in enumerate(NXentries)]
        # filter for data that is the same length as the independent axis
        signals = [{k: da.from_array(v, chunks=tuple(
            [np.int(np.divide(dim, self.npartitions)) for dim in v.shape])).astype('f4') for k, v in d.items() if
                    np.abs(v.shape[0] - list(indep[i].values())[0].shape[0]) < 2} for i, d in enumerate(data)]
        # group all remaining arrays
        other_axis = [
            {k: da.from_array(v, chunks=tuple([np.int(np.divide(dim, 2)) for dim in v.shape])) for k, v in d.items() if
             np.abs(v.shape[0] - list(indep[i].values())[0].shape[0]) > 2} for i, d in enumerate(data)]
        # Reload independent axis data as dataarray
        indep = [{k: da.from_array(v,
                                   chunks=tuple([np.int(np.divide(dim, self.npartitions)) for dim in v.shape])).astype(
            'f4') for k, v in d.items()} for d in indep]

        # Get sample name if it exists and add data to scan dictionary
        for i, entry in enumerate(NXentries):
            try:
                scan = {"command": commands[i]}
            except IndexError:
                scan = {}
            if "sample" in h5[entry].keys():
                if "name" in h5[entry + "/sample"].keys():
                    if isinstance(h5[entry + "/sample/name"][()], str):
                        scan.update({"sample": str(h5[entry + "/sample/name"][()])})
                    if "description" in h5[entry + "/sample"].keys():
                        scan.update({"description": str(h5[entry + "/sample/description"][()])})
                elif "description" in h5[entry + "/sample"].keys():
                    scan.update({"sample": str(h5[entry + "/sample/description"][()], 'utf-8').split('\x00')[0]})
            scan.update({"independent": indep[i], "signals": signals[i], "other": other_axis[i],
                         "npartitions": self.npartitions})
            entries.update({entry: scan})
        return {file_root: entries}

    def interpolate(self, **kwargs):
        _interpolate = partial(self._interpolate, **kwargs)
        entries = []
        for file, val in self.entries():
            for key, entry in val.__dict__.items():
                entries.append(entry)
        with ThreadPool(self.threads) as pool:
            results = list(tqdm(pool.imap_unordered(_interpolate, entries), total=len(entries)))
        return results

    def _interpolate(self, entry, **kwargs):
        return entry.interpolate(**kwargs)

    def average(self, bad_scans = None):
        sample_scans = {}
        i = 1
        for k, file in self.scans.items():
            for entry, scan in file.items():
                i = i + 1
                if 'binned' in scan.keys():
                    key=[]
                    if 'sample' in scan.keys():
                        key.append(scan['sample'])
                    if 'command' in scan.keys():
                        key.append("_".join(scan['command']))
                    key = ":".join(key)
                    if i not in bad_scans:
                        if key in sample_scans.keys():
                            l = sample_scans[key]
                            sample_scans[key] = l.append(scan['binned']['dataframe'])
                        else:
                            sample_scans.update({key:[scan['binned']['dataframe']]})
        average = {}
        for k, v in sample_scans.items():
            average.update({k:pd.concat(v).mean()})
        self.average = average
        return average


    def __str__(self):
        return f"Scans: {self.scans}"

    def __repr__(self):
        return f"Scans: {self.scans}"

    def _repr_html_(self):
        table = [
            "<table>",
            "  <thead>",
            "    <tr><th>File</th><th> Entry </th><th> Sample </th><th> Command </th><th>"
            " Independent </th><th> Signals </th><th> Other </th></tr>",
            "  </thead>",
            "  <tbody>",
        ]
        for key in self.scans.keys():
            for subkey in self.scans[key].__dict__.keys():
                table.append(f"<tr><th>{key}</th><td>{subkey}</td>" + self.scans[key][subkey]._repr_html_() + "</tr>")
        table.append("</tbody></table>")

        return "\n".join(table)
