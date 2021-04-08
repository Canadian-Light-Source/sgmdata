import os
import h5py
from . import config
import h5pyd
from dask import compute, delayed
import dask.array as da
import dask.dataframe as dd
from dask.distributed import Client
import pandas as pd
import numpy as np
from multiprocessing.pool import ThreadPool
from functools import partial
from sgmdata.plots import eemscan, xrfmap
from sgmdata.xrffit import fit_peaks
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
            if 'method' in kwargs.keys():
                method = kwargs.keys()
            else:
                method = "nearest"
            axis = self['independent']
            dim = len(axis.keys())
            if 'start' not in kwargs.keys():
                if hasattr(self, 'command'):
                    command = self.command
                    if 'scan' in command[0]:
                        start = [round(float(command[2]))]
                    elif 'mesh' in command[0]:
                        start = [round(float(command[2])), round(float(command[6]))]
                else:
                    start = [round(v.min()) for k, v in axis.items()]
            else:
                start = kwargs['start']
            if 'stop' not in kwargs.keys():
                if hasattr(self, 'command'):
                    command = self.command
                    if 'scan' in command[0]:
                        stop = [round(float(command[3]))]
                    elif 'mesh' in command[0]:
                        stop = [round(float(command[3])), round(float(command[7]))]
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
                    df = df.compute().unstack().interpolate(method=method).fillna(0).stack().reindex(idx)
                    binned = {"dataframe": df}
                    self.__setattr__('binned', binned)
                    return df
                else:
                    raise ValueError("Too many independent axis for interpolation")

            else:
                return df

        def fit_mcas(self, detectors=[], emission=[]):
            if not len(detectors):
                detectors = [k for k,v in self['signals'].items() if 'sdd' in k or 'ge' in k]
            if not len(emission):
                if 'emission' in self['other']:
                    emission = self['other']['emission'].compute()
                else:
                    sig = self['signals'][detectors[0]]
                    emission = np.linspace(0, sig.shape[1]*10, sig.shape[1])
            if 'binned' in self.keys():
                df = self['binned']['dataframe']
                roi_cols = df.filter(regex="sdd[1-4]_[0-2].*").columns
                df.drop(columns=roi_cols, inplace=True)
                scaler_cols = df.filter(regex="sdd[1-4].*").columns
                scaler_df = df.drop(columns=scaler_cols, inplace=False)
                data = []
                for det in detectors:
                    rgx = "%s.*" % det
                    data.append(df.filter(regex=rgx, axis=1))
                fit_df, pks, wid = fit_peaks(emission, data)
                new_df = pd.concat([scaler_df, fit_df], axis=1, sort=False)
                self['fit'] = {"dataframe": new_df, "emission":emission, "peaks": np.array([emission[p] for p in pks]), "width": wid }
                return new_df


        def __repr__(self):
            represent =  ""
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

        def write(self, filename=None):
            """ Write data to NeXuS formatted data file."""
            if 'sdd3' in self['signals']:
                signal = u'sdd3'
            elif 'ge32' in self['signals']:
                signal = u'ge32'
            elif 'tey' in self['signals']:
                signal = u'tey'
            elif 'mu_abs' in self['signals']:
                signal = u'mu_abs'
            else:
                signal = self.signals[0]
            if not filename:
                filename = self.sample + ".nxs"
            if 'binned' in self.keys():
                if 'dataframe' in self['binned'].keys():
                    df = self['binned']['dataframe']
                    h5 = h5py.File(filename, "w")
                    NXentries = [int(str(x).split("entry")[1]) for x in h5['/'].keys() if 'NXentry' in str(h5[x].attrs.get('NX_class'))]
                    if NXentries:
                        NXentries.sort()
                        entry = 'entry' + str(NXentries[-1]+1)
                    else:
                        entry = 'entry1'
                    axes = [nm for nm in df.index.names]
                    nxent = h5.create_group(entry)
                    nxent.attrs.create(u'NX_class', u'NXentry')
                    nxdata = nxent.create_group('data')
                    nxdata.attrs.create(u'NX_class', u'NXdata')
                    nxdata.attrs.create(u'axes', axes)
                    nxdata.attrs.create(u'signal', signal)
                    if len(axes) == 1:
                        arr = np.array(df.index)
                        nxdata.create_dataset(df.index.name, arr.shape, data=arr, dtype=arr.dtype)
                    elif len(axes) > 1:
                        for i, ax in enumerate(axes):
                            arr = np.array(df.index.levels[i])
                            nxdata.create_dataset(ax, arr.shape, data=arr, dtype=arr.dtype)

                    for sig in self.signals:
                        arr = df.filter(regex="%s.*" % sig.split('_')[0]).to_numpy()
                        if len(df.index.names) > 1:
                            shape = [len(df.index.levels[0]),len(df.index.levels[1])]
                            shape += [s for s in arr.shape[1:]]
                            arr = np.reshape(arr, tuple(shape))
                        nxdata.create_dataset(sig, arr.shape, data=arr)
                    h5.close()
            else:
                raise AttributeError("no interpolated data found to write")

        def plot(self, json_out = False):
            """
            Determines the appropriate plot based on independent axis number and name
            """
            dim = len(self.independent)
            if dim == 1 and 'en' in self.independent.keys():
                keys = eemscan.required
                if 'binned' in self.keys():
                    if 'dataframe' in self['binned'].keys():
                        print("Plotting Interpolated Data")
                        df = self['binned']['dataframe']
                        roi_cols = df.filter(regex="sdd[1-4]_[0-2].*").columns
                        df.drop(columns=roi_cols, inplace=True)
                        data = {k: df.filter(regex=("%s.*" % k), axis=1).to_numpy() for k in keys}
                        data = {k: v for k,v in data.items() if v.size}
                        data.update({df.index.name: np.array(df.index), 'emission': np.linspace(0, 2560, 256)})
                        if 'image' in keys:
                            data.update({'image': data['sdd1'], 'filename': str(self.sample)})
                        data.update({'json': json_out})
                        return eemscan.plot(**data)
                else:
                    print("Plotting Raw Data")
                    ds = int(self.independent['en'].shape[0] / 1000) + 1
                    data = {k: self.signals[s][::ds].compute() for s in self.signals.keys() for k in keys if k in s }
                    data.update(
                        {k: self.independent[s][::ds].compute() for s in self.independent.keys() for k in keys if k in s })
                    data.update({k: self.other[s].compute() for s in self.other.keys() for k in keys if s in k })
                    if 'image' in keys:
                        data.update({'image': self.signals['sdd1'][::ds].compute(), 'filename': str(self.sample)})
                    data.update({'json':json_out})
                    return eemscan.plot(**data)
            elif dim == 2:
                keys = xrfmap.required
                if 'fit' in self.keys():
                    df = self['fit']['dataframe']
                    emission = self['fit']['emission']
                    peaks = self['fit']['peaks']
                    width = self['fit']['width']
                    data = {k: df.filter(regex=("%s.*" % k), axis=1).to_numpy() for k in keys if any(k in mystring for mystring in self.signals.keys())}
                    data.update({k:np.reshape(v, (len(df.index.levels[0]),len(df.index.levels[1]),v.shape[-1])) if len(v.shape) == 2 else np.reshape(v,(len(df.index.levels[0]),len(df.index.levels[1]))) for k,v in data.items()})
                    data.update({n:df.index.levels[i] for i,n in enumerate(list(df.index.names))})
                    data.update({'emission': emission, 'peaks':peaks, 'width': width})
                    if 'image' in keys:
                        data.update({"image": data['sdd1']})
                    data.update({'json': json_out})
                    xrfmap.plot(**data)

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

class DisplayDict(dict):
    def __getattr__(self, name):
        return self[name]


    def __setattr__(self, name, value):
        self[name] = value

    def _repr_html_(self):
        table = [
            "<table>",
            "  <thead>",
            "    <tr><td> </td><th>Key</th><th>Value</th></tr>",
            "  </thead>",
            "  <tbody>",
        ]
        for key, value in self.__dict__.items():
            table.append(f"<tr><th> {key}</th><th>{value}</th></tr>")
        table.append("</tbody></table>")
        return "\n".join(table)

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

    class Processed(DisplayDict):

        def write(self, filename=None):
            if 'sdd3' in self['signals']:
                signal = u'sdd3'
            elif 'ge32' in self['signals']:
                signal = u'ge32'
            elif 'tey' in self['signals']:
                signal = u'tey'
            elif 'mu_abs' in self['signals']:
                signal = u'mu_abs'
            else:
                signal = self.signals[0]
            if not filename:
                filename = self.sample + ".nxs"
            h5 = h5py.File(filename, "a")
            NXentries = [int(str(x).split("entry")[1]) for x in h5['/'].keys() if 'NXentry' in str(h5[x].attrs.get('NX_class'))]
            if NXentries:
                NXentries.sort()
                entry = 'entry' + str(NXentries[-1]+1)
            else:
                entry = 'entry1'
            axes = [nm for nm in self.data.index.names]
            nxent = h5.create_group(entry)
            nxent.attrs.create(u'NX_class', u'NXentry')
            nxdata = nxent.create_group('data')
            nxdata.attrs.create(u'NX_class', u'NXdata')
            nxdata.attrs.create(u'axes', axes)
            nxdata.attrs.create(u'signal', signal)
            if len(axes) == 1:
                arr = np.array(self.data.index)
                nxdata.create_dataset(self.data.index.name, arr.shape, data=arr, dtype=arr.dtype)
            elif len(axes) > 1:
                for i, ax in enumerate(axes):
                    arr = np.array(self.data.index.levels[i])
                    nxdata.create_dataset(ax, arr.shape, data=arr, dtype=arr.dtype)
            for sig in self.signals:
                arr = self.data.filter(regex="%s." % sig.split('_')[0]).to_numpy()
                if len(self.data.index.names) > 1:
                    shape = [len(self.data.index.levels[0]),len(self.data.index.levels[1])]
                    shape += [s for s in arr.shape[1:]]
                    arr = np.reshape(arr, tuple(shape))
                nxdata.create_dataset(sig, arr.shape, data=arr, dtype=arr.dtype)
            h5.close()

        def plot(self, json_out=False):
            if 'type' in self.__dict__.keys():
                pass
            else:
                if 'scan' in self.command[0] and "en" == self.command[1]:
                    keys = eemscan.required
                    df = self.data
                    roi_cols = df.filter(regex="sdd[1-4]_[0-2].*").columns
                    df.drop(columns = roi_cols, inplace=True)
                    data = {k: df.filter(regex=("%s.*" % k), axis=1).to_numpy() for k in keys}
                    data.update({df.index.name: np.array(df.index), 'emission': np.linspace(0,2560,256)})
                    data.update({'image':data['sdd1'], 'json':json_out})
                    return eemscan.plot(**data)
                elif 'mesh' in self.command[0]:
                    pass

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
        err = [l['ERROR'] for l in L if 'ERROR' in l.keys()]
        L = [l for l in L if 'ERROR' not in l.keys()]
        if len(err):
            warnings.warn(f"Some scan files were not loaded: {err}")
            for e in err:
                del self.scans[e]
        self.scans.update({k:SGMScan(**v) for d in L for k,v in d.items()})
        self.entries = self.scans.items

    def _find_data(self, node, indep=None, other=False):
        data = {}

        def search(name, node):
            if indep:
                if isinstance(node, h5pyd.Dataset) or isinstance(node, h5py.Dataset):
                    if "S" not in str(node.dtype).upper() and node.shape and node.shape[0] > 1:
                        d_name = name.split('/')[-1]
                        l = [True for axis in indep if d_name.find(axis) == 0]
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
        file_root = file.split("\\")[-1].split("/")[-1].split(".")[0]
        if os.path.exists(file):
            try:
                h5 = h5py.File(file, 'r')
            except Exception as f:
                warnings.warn(f"Could not open file, h5py raised: {f}")
                return {"ERROR": file_root}
        else:
            try:
                h5 = h5pyd.File(file, "w", config.get("h5endpoint"), username=config.get("h5user"),
                                password=config.get("h5pass"))
            except Exception as f:
                warnings.warn(f"Could not open file, h5pyd raised: {f}")
                return {"ERROR": file_root}
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
                warnings.warn(
                    "Scan entry didn't have a 'command' string saved. Command load can be skipped by providing a list of independent axis names")
                return {"ERROR": file_root}
            for i, command in enumerate(commands):
                if 'mesh' in command[0]:
                    independent.append((command[1], command[5]))
                elif 'scan' in command[0]:
                    independent.append((command[1],))
        else:
            for i, entry in enumerate(NXentries):
                independent.append(tuple(self.axes))
        if not independent:
            return {"ERROR": file_root}
        indep = [self._find_data(h5[entry], independent[i]) for i, entry in enumerate(NXentries)]
        # search for data that is not an array mentioned in the command
        data = [self._find_data(h5[entry], independent[i], other=True) for i, entry in enumerate(NXentries)]
        # filter for data that is the same length as the independent axis
        try:
            signals = [{k: da.from_array(v, chunks=tuple(
                [np.int(np.divide(dim, self.npartitions)) for dim in v.shape])).astype('f4') for k, v in d.items() if
                    np.abs(v.shape[0] - list(indep[i].values())[0].shape[0]) < 2} for i, d in enumerate(data)]
        except IndexError:
            return {"ERROR": file_root}
        # group all remaining arrays
        try:
            other_axis = [
                {k: da.from_array(v, chunks=tuple([np.int(np.divide(dim, 2)) for dim in v.shape])) for k, v in d.items() if
                     np.abs(v.shape[0] - list(indep[i].values())[0].shape[0]) > 2} for i, d in enumerate(data)]
        except:
            return {"ERROR": file_root}
        # Reload independent axis data as dataarray
        try:
            indep = [{k: da.from_array(v,
                                   chunks=tuple([np.int(np.divide(dim, self.npartitions)) for dim in v.shape])).astype(
                                    'f4') for k, v in d.items()} for d in indep]
        except:
            return {"ERROR": file_root}

        # Get sample name if it exists and add data to scan dictionary
        for i, entry in enumerate(NXentries):
            try:
                scan = {"command": commands[i]}
            except IndexError:
                scan = {}
            if "sample" in h5[entry].keys():
                if "name" in h5[entry + "/sample"].keys():
                    sample_string = h5[entry + "/sample/name"][()]
                    if isinstance(sample_string, str) :
                        scan.update({"sample": str(sample_string)})
                    elif isinstance(sample_string, bytes):
                        scan.update({"sample": sample_string.decode('utf-8')})
                    if "description" in h5[entry + "/sample"].keys():
                        scan.update({"description": str(h5[entry + "/sample/description"][()])})
                elif "description" in h5[entry + "/sample"].keys():
                    scan.update({"sample": str(h5[entry + "/sample/description"][()], 'utf-8').split('\x00')[0]})
            scan.update({
                        "independent": indep[i],
                        "signals": signals[i],
                        "other": other_axis[i],
                        "npartitions": self.npartitions
            })
            if 'sample' in self.__dict__.keys():
                if 'sample' in scan.keys():
                    if self.sample in scan['sample']:
                        entries.update({entry: scan})
            else:
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

    def mean(self, bad_scans=None):
        if bad_scans is None:
            bad_scans = []
        sample_scans = {}
        i = 1
        for k, file in self.scans.items():
            for entry, scan in file.__dict__.items():
                i = i + 1
                signals = [k for k, v in scan['signals'].items()]
                if 'binned' in scan.keys():
                    key = []
                    if 'sample' in scan.keys():
                        key.append(scan['sample'])
                    else:
                        key.append('Unknown')
                    if 'command' in scan.keys():
                        key.append("_".join(scan['command']))
                    else:
                        key.append("None")
                    key = ":".join(key)
                    if i not in bad_scans:
                        if key in sample_scans.keys():
                            l = sample_scans[key]['data'] + [scan['binned']['dataframe']]
                            d = {'data': l, 'signals': signals}
                            sample_scans.update({key: d})
                        else:
                            sample_scans.update({key: {'data': [scan['binned']['dataframe']], 'signals': signals}})
        average = DisplayDict()
        for k, v in sample_scans.items():
            if len(v['data']) > 1:
                df_concat = pd.concat(v['data'])
                key = k.split(":")[0]
                command = k.split(":")[-1]
                df = df_concat.groupby(df_concat.index).mean()
                if key in average.keys():
                    l = average[key] + [
                        SGMData.Processed(command=command.split('_'), data=df, signals=v['signals'], sample=key)]
                    average.update({key: l})
                else:
                    average.update({key: [
                        SGMData.Processed(command=command.split('_'), data=df, signals=v['signals'], sample=key)]})
        self.averaged = average
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
