import os
import h5py

from . import config
import h5pyd
from dask import delayed
import dask.array as da
import dask.dataframe as dd
import pandas as pd
import numpy as np
from multiprocessing.pool import ThreadPool
from functools import partial
from sgmdata.plots import eemscan, xrfmap
from sgmdata.xrffit import fit_peaks
from sgmdata.interpolate import interpolate, compute_df
from dask.diagnostics import ProgressBar

import warnings

try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
        from tqdm.notebook import tqdm  # Jupyter notebook or qtconsole
    else:
        from tqdm import tqdm  # Other type (?)
except NameError:
    from tqdm import tqdm



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

class SGMScan(object):
    """
        Data class for storing dask arrays for SGM data files that have been grouped into 'NXentry',
        and then divided into signals, independent axes, and other data.  Contains convenience classes
        for interpolation.
    """

    class DataDict(DisplayDict):

        def interpolate(self, **kwargs):
            independent = self['independent']
            signals = self['signals']
            kwargs['npartitions'] = self.npartitions
            if hasattr(self, 'command'):
                command = self.command
            else:
                command = None
            df, idx = interpolate(independent, signals, command=command, **kwargs)
            if isinstance(df, dd.DataFrame) or isinstance(df, pd.DataFrame):
                self.__setattr__('binned', {"dataframe": df, "index": idx})
            return df

        def compute(self, **kwargs):
            method = kwargs.get('method', True)
            if hasattr(self, 'binned'):
                if 'df' in self['binned'].keys():
                    idx = self['binned'].get('idx', None)
                    if isinstance(idx, pd.MultiIndex):
                        df = self["binned"]['dataframe'].compute().interpolate(method=method).fillna(0).stack().reindex(
                            idx)
                    elif isinstance(idx, pd.Index):
                        df = self["binned"]['dataframe'].compute().reindex(idx).interpolate()
                    self['binned']['dataframe'] = df
                    return df
            print("Nothing to compute.")

        def fit_mcas(self, detectors=[], emission=[]):
            if not len(detectors):
                detectors = [k for k, v in self['signals'].items() if 'sdd' in k or 'ge' in k]
            if not len(emission):
                if 'emission' in self['other']:
                    emission = self['other']['emission'].compute()
                else:
                    sig = self['signals'][detectors[0]]
                    emission = np.linspace(0, sig.shape[1] * 10, sig.shape[1])
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
                self['fit'] = {"dataframe": new_df, "emission": emission, "peaks": np.array([emission[p] for p in pks]),
                               "width": wid}
                return new_df

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
                    NXentries = [int(str(x).split("entry")[1]) for x in h5['/'].keys() if
                                 'NXentry' in str(h5[x].attrs.get('NX_class'))]
                    if NXentries:
                        NXentries.sort()
                        entry = 'entry' + str(NXentries[-1] + 1)
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
                            shape = [len(df.index.levels[0]), len(df.index.levels[1])]
                            shape += [s for s in arr.shape[1:]]
                            arr = np.reshape(arr, tuple(shape))
                        nxdata.create_dataset(sig, arr.shape, data=arr)
                    h5.close()
            else:
                raise AttributeError("no interpolated data found to write")

        def plot(self, **kwargs):
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
                        data = {k: v for k, v in data.items() if v.size}
                        data.update({df.index.name: np.array(df.index), 'emission': np.linspace(0, 2560, 256)})
                        if 'image' in keys:
                            data.update({'image': data['sdd1'], 'filename': str(self.sample)})
                        return eemscan.plot(**data)
                else:
                    print("Plotting Raw Data")
                    ds = int(self.independent['en'].shape[0] / 1000) + 1
                    data = {k: self.signals[s][::ds].compute() for s in self.signals.keys() for k in keys if k in s}
                    data.update(
                        {k: self.independent[s][::ds].compute() for s in self.independent.keys() for k in keys if
                         k in s})
                    data.update({k: self.other[s].compute() for s in self.other.keys() for k in keys if s in k})
                    if 'image' in keys:
                        data.update({'image': self.signals['sdd1'][::ds].compute(), 'filename': str(self.sample)})
                    kwargs.update(data)
                    return eemscan.plot(kwargs)
            elif dim == 2:
                keys = xrfmap.required
                if 'fit' in self.keys():
                    df = self['fit']['dataframe']
                    emission = self['fit']['emission']
                    peaks = self['fit']['peaks']
                    width = self['fit']['width']
                    data = {k: df.filter(regex=("%s.*" % k), axis=1).to_numpy() for k in keys if
                            any(k in mystring for mystring in self.signals.keys())}
                    data.update({k: np.reshape(v,
                                               (len(df.index.levels[0]), len(df.index.levels[1]), v.shape[-1])) if len(
                        v.shape) == 2 else np.reshape(v, (len(df.index.levels[0]), len(df.index.levels[1]))) for k, v in
                                 data.items()})
                    data.update({n: df.index.levels[i] for i, n in enumerate(list(df.index.names))})
                    data.update({'emission': emission, 'peaks': peaks, 'width': width})
                    if 'image' in keys:
                        data.update({"image": data['sdd1']})
                    kwargs.update(data)
                    xrfmap.plot(**kwargs)
                else:
                    print("Plotting Raw Data")
                    ds = int(self.independent['xp'].shape[0] / 10000) + 1
                    data = {k: self.signals[s][::ds].compute() for s in self.signals.keys() for k in keys if k in s}
                    data.update({'command': self.command})
                    data.update(
                        {k: self.independent[s][::ds].compute() for s in self.independent.keys() for k in keys if
                         k in s})
                    data.update({k: self.other[s].compute() for s in self.other.keys() for k in keys if s in k})
                    kwargs.update(data)
                    xrfmap.plot_xyz(**kwargs)

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
            NXentries = [int(str(x).split("entry")[1]) for x in h5['/'].keys() if
                         'NXentry' in str(h5[x].attrs.get('NX_class'))]
            if NXentries:
                NXentries.sort()
                entry = 'entry' + str(NXentries[-1] + 1)
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
                    shape = [len(self.data.index.levels[0]), len(self.data.index.levels[1])]
                    shape += [s for s in arr.shape[1:]]
                    arr = np.reshape(arr, tuple(shape))
                nxdata.create_dataset(sig, arr.shape, data=arr, dtype=arr.dtype)
            h5.close()

        def plot(self, **kwargs):
            if 'type' in self.__dict__.keys():
                pass
            else:
                if 'scan' in self.command[0] and "en" == self.command[1]:
                    keys = eemscan.required
                    df = self.data
                    roi_cols = df.filter(regex="sdd[1-4]_[0-2].*").columns
                    df.drop(columns=roi_cols, inplace=True)
                    data = {k: df.filter(regex=("%s.*" % k), axis=1).to_numpy() for k in keys}
                    data.update({df.index.name: np.array(df.index), 'emission': np.linspace(0, 2560, 256)})
                    data.update({'image': data['sdd1']})
                    kwargs.update(data)
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
        self.scans = {k.split('/')[-1].split(".")[0]: [] for k in files}
        self.interp_params = {}
        with ThreadPool(self.threads) as pool:
            L = list(tqdm(pool.imap_unordered(self._load_data, files), total=len(files)))
        err = [l['ERROR'] for l in L if 'ERROR' in l.keys()]
        L = [l for l in L if 'ERROR' not in l.keys()]
        if len(err):
            warnings.warn(f"Some scan files were not loaded: {err}")
            for e in err:
                del self.scans[e]
        self.scans.update({k: SGMScan(**v) for d in L for k, v in d.items()})
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
                            h5[entry + '/command'][()], 'utf-8').split() for entry in NXentries]
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
            warnings.warn(
                "No independent axis was identified, you can supply your own axes name using the keyword \"axes\" (list)"
            )
            return {"ERROR": file_root}
        indep = [self._find_data(h5[entry], independent[i]) for i, entry in enumerate(NXentries)]
        # search for data that is not an array mentioned in the command
        data = [self._find_data(h5[entry], independent[i], other=True) for i, entry in enumerate(NXentries)]
        # filter for data that is the same length as the independent axis
        try:
            signals = [{k: da.from_array(v, chunks=tuple(
                [np.int(np.divide(dim, self.npartitions)) for dim in v.shape])).astype('f4') for k, v in d.items() if
                        np.abs(v.shape[0] - list(indep[i].values())[0].shape[0]) < 30} for i, d in enumerate(data)]
        except IndexError:
            return {"ERROR": file_root}
        # group all remaining arrays
        try:
            other_axis = [
                {k: da.from_array(v, chunks=tuple([np.int(np.divide(dim, 2)) for dim in v.shape])) for k, v in d.items()
                 if
                 np.abs(v.shape[0] - list(indep[i].values())[0].shape[0]) > 2} for i, d in enumerate(data)]
        except:
            return {"ERROR": file_root}
        # Reload independent axis data as dataarray
        try:
            indep = [{k: da.from_array(v,
                                       chunks=tuple(
                                           [np.int(np.divide(dim, self.npartitions)) for dim in v.shape])).astype(
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
                    if isinstance(sample_string, str):
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
        compute = kwargs.get('compute', True)
        if compute:
            return entry.interpolate(**kwargs)
        else:
            independent = entry['independent']
            signals = entry['signals']
            kwargs['npartitions'] = self.npartitions
            if hasattr(entry, 'command'):
                command = entry.command
            else:
                command = None
            return interpolate(independent, signals, command=command, **kwargs)

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
