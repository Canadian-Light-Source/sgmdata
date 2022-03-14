import os
import sys
import h5py
import datetime
from . import config
import h5pyd
import dask.array as da
import dask.dataframe as dd
import pandas as pd
import numpy as np
from multiprocessing.pool import ThreadPool
from functools import partial
from sgmdata.plots import eemscan, xrfmap
from sgmdata.xrffit import fit_peaks
from sgmdata.interpolate import interpolate, shift_cmesh
from .utilities.magicclass import OneList, DisplayDict

import warnings

from collections import OrderedDict
import os.path, time

try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
        from tqdm.notebook import tqdm  # Jupyter notebook or qtconsole
    else:
        from tqdm import tqdm  # Other type (?)
except NameError:
    from tqdm import tqdm

sys_has_tab = False
if 'tabulate' in sys.modules:
    sys_has_tab = True


class SGMScan(DisplayDict):
    """
    ### Description:
    >Data class for storing dask arrays for SGM data files that have been grouped into 'NXentry',
     and then divided into signals, independent axes, and other data.  Contains convenience classes
     for interpolation.

    ### Functions:
    >**interpolate()** -- for each scan entry in self.items() there is a SGMScan.entry.interpolate() function,
    see interpolate() documentation.

    >**plot()** -- for each scan entry in self.items() there exists a SGMScan.entry.plot() method for displaying the 
    contained data with bokeh.

    >**fit_mcas()** -- for each scan entry in self.items() there exists a SGMScan.entry.fit_mcas() method for gaussian
    peak fitting of the interpolated mca data. Returns resulting dataframe.

    >**get_arr()** -- for each scan entry in self.items() there exists a SGMScan.entry.get_arr() which will return a numpy array
    from an stored interpolated dataframe by using a keyword filter:
    ```python
        from sgmdata import SGMData

        data = SGMData('file.nxs')
        data.interpolate()
        sdd1 = data.get_arr('sdd1')
        sdd1.shape # (1290, 256)
    ```
    """

    class DataDict(DisplayDict):

        def get_arr(self, detector):
            """
            ### Description:
            >Function to return a numpy array from the internal pandas dataframe, for a given detector.
            ### Args:
            >**detector** *(str)* -- Name of detector.
            ### Returns:
            >**detector** *(ndarray)*
            """
            if isinstance(detector, str):
                try:
                    return self['binned']['dataframe'].filter(regex=f'{detector}.*').to_numpy()
                except (AttributeError, KeyError):
                    warnings.warn(f"No dataframe loaded in scan dictionary. Have you run interpolate yet?")

        def interpolate(self, **kwargs):
            """
            ### Description:
                Creates the bins required for each independent axes to be histogrammed into for interpolation,
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

        def read(self, filename=None):
            """
            ### Description
            >Function to load in already processed data from file.
            ### Keywords
            >**filename** *(str)* -- path to file on disk.
            """
            if not filename:
                return []
            if os.path.exists(filename):
                try:
                    h5 = h5py.File(filename, 'r')
                except Exception as f:
                    warnings.warn(f"Could not open file, h5py raised: {f}")
                    return
            elif os.path.exists('./data/'):
                try:
                    h5 = h5py.File(filename.replace('/home/jovyan/', './'), 'r')
                except Exception as f:
                    warnings.warn(f"Could not open file, h5py raised: {f}")
                    return
            else:
                try:
                    h5 = h5pyd.File(filename, "r", config.get("h5endpoint"), username=config.get("h5user"),
                                    password=config.get("h5pass"))
                except Exception as f:
                    warnings.warn(f"Could not open file, h5pyd raised: {f}")
                    return
            NXentries = [str(x) for x in h5['/'].keys()
                         if 'NXentry' in str(h5[x].attrs.get('NX_class')) and str(x) in self['name']]
            NXdata = [entry + "/" + str(x) for entry in NXentries for x in h5['/' + entry].keys()
                      if 'NXdata' in str(h5[entry + "/" + x].attrs.get('NX_class'))]
            axes = [[str(nm) for nm in h5[nxdata].keys() for s in h5[nxdata].attrs.get('axes') if str(s) in str(nm) or
                     str(nm) in str(s)] for nxdata in NXdata]
            indep_shape = [v.shape for i, d in enumerate(NXdata) for k, v in h5[d].items() if k in axes[i][0]]
            data = [{k: np.squeeze(v) for k, v in h5[d].items() if v.shape[0] == indep_shape[i][0]} for i, d in
                    enumerate(NXdata)]
            df_sdds = [pd.DataFrame(
                {k + f"-{j}": v[:, j] for k, v in data[i].items() if len(v.shape) == 2 for j in range(0, v.shape[1])})
                for i, _ in enumerate(NXdata)]
            df_scas = [pd.DataFrame.from_dict(
                {k: v for k, v, in data[i].items() if len(v.shape) < 2}).join(df_sdds[i]).groupby(axes[i]).mean()
                       for i, _ in enumerate(NXdata)]
            if len(df_scas) == 1:
                self.__setattr__('binned', {"dataframe": df_scas[0], "index": df_scas[0].index})
            return df_scas

        def write(self, filename=None):
            """
            ### Description:
            >Write data to NeXuS formatted data file.
            ### Keyword:
            >**filename** *(str / os.path)* -- path/name of file for output.
            """
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
            ### Description
            >Determines the appropriate plot based on independent axis number and name.
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
                    return eemscan.plot(**kwargs)
            elif dim == 2:
                keys = xrfmap.required
                if 'fit' in self.keys():
                    df = self['fit']['dataframe']
                    df.fillna(0, inplace=True )
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
                    return xrfmap.plot(**kwargs)
                elif 'binned' in self.keys():
                    print("Plotting Interpolated Data")
                    df = self['binned']['dataframe']
                    roi_cols = df.filter(regex="sdd[1-4]_[0-2].*").columns
                    df.drop(columns=roi_cols, inplace=True)
                    df.fillna(0, inplace=True )
                    data = {k: df.filter(regex=("%s.*" % k), axis=1).to_numpy() for k in keys}
                    data = {k: v for k, v in data.items() if v.size}
                    data.update({n: df.index.levels[i] for i, n in enumerate(list(df.index.names))})
                    data.update({'emission': np.linspace(0, 2560, 256)})
                    kwargs.update(data)
                    return xrfmap.plot_interp(**kwargs)
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
                    return xrfmap.plot_xyz(**kwargs)

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

        def _repr_console_(self):
            final_data = 'sample:\t' + str(self.sample) + '\t\t|\t\t'
            final_data = final_data + 'command:\t' + str(self.command) + '\t\t|\t\t'
            final_data = final_data + 'independent:\t' + str(self.independent.keys()) + '\t\t|\t\t'
            final_data = final_data + 'signals:\t' + str(self.signals.keys()) + '\t\t|\t\t'
            final_data = final_data + 'other:\t' + str(self.other.keys()) + '\t\t|\t\t'
            return final_data

    def __init__(self, *args, **kwargs):
        kw_sorted = []
        longest = max([len(entry) for entry in kwargs.keys()])
        shortest = min([len(entry) for entry in kwargs.keys()])
        cur_len = shortest
        while cur_len <= longest:
            temp = sorted([k for k in kwargs.keys() if len(k) == cur_len])
            for entry in temp:
                kw_sorted.append(entry)
            cur_len += 1
        kwargs_sorted = OrderedDict({k: kwargs[k] for k in kw_sorted})
        super(SGMScan, self).__init__(*args, **kwargs_sorted)

        self.__dict__.update(kwargs)
        for key, value in kwargs.items():
            value.update({'name': key})
            self.__dict__[key] = SGMScan.DataDict(value)

    def __repr__(self):
        represent = ""
        for key in self.keys():
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
        for key in self.keys():
            table.append(f"<tr><th> {key}</th>" + self.__dict__[key]._repr_html_() + "</tr>")
        table.append("</tbody></table>")

        return "\n".join(table)

    def _repr_console_(self):
        needed_info = ['entry', 'sample', 'command', 'independent', 'signals', 'other']
        if sys_has_tab:
            temp_data = []
            final_data = []
            for key in self.keys():
                temp_data.append(key)
                for title in self[key].keys():

                    if title in needed_info:
                        temp_data.append(self.__dict__[key][title])
                final_data.append(temp_data.copy())
                temp_data.clear()
            return tabulate(final_data, headers=needed_info)
        else:
            temp_data = ''
            final_data = ''
            for key in self.keys():
                temp_data = temp_data + 'Entry:\t'
                temp_data = temp_data + str(key)
                for title in self[key].keys():

                    if title in needed_info:
                        temp_data = temp_data + '\t\t|\t\t'
                        temp_data = temp_data + (str(title) + ":\t" + str(self.__dict__[key][title]))
                final_data = final_data + ('\n' + str(temp_data))
                temp_data = ''
            return final_data

    def __getitem__(self, item):
        return self.__dict__[item]


class SGMData(object):
    """
    ### Description:
        Class for loading in data from h5py or h5pyd files for raw SGM data.
        To substantiate pass the class pass a single (or list of) system file paths
        (or hsds path).  e.g. data = SGMData('/path/to/my/file.nxs') or SGMData(['1.h5', '2.h5']).
        The data is auto grouped into three classifications: "independent", "signals", and "other".
        You can view the data dictionary representation in a Jupyter cell by just invoking the SGMData() object.

    ### Args:
        >**file_paths** *(str or list)* List of file names to be loaded in by the data module.

    ### Keywords:
        >**npartitions** *(type: integer)* -- choose how many divisions (threads)
                                       to split the file data arrays into.

        >**scheduler** *(type: str)* -- use specific dask cluster for operations, e.g. 'dscheduler:8786'

        >**axes** *(type: list(str))* -- names of the axes to use as independent axis and ignore
                                  spec command issued

        >**threads** *(type: int)* -- set the number of threads in threadpool used to load in data.

        >**scan_type** *(type: str)* -- used to filter the type of scan loaded, e.g. 'cmesh', '

        >**shift** *(type: float)*  -- default 0.5.  Shifting 'x' axis data on consecutive passes of stage
                                for cmesh scans.

    ### Functions:
        >**interpolate()** -- botch operation on all scans in SGMData, takes in same parameters as interpolate(),
        see interpolate() documentation.

        >**mean()** -- averages all interpolated data together (organized by sample, scan type & range), returns list, saves data
                  under a dictionary in SGMData().averaged


    ### Attributes
        >**scans** *(SGMScan)* By default the query will create an SGMData object containing your data, this can be turned off with the data keyword.

        >**averaged** *(list)*. Contains the averaged data from all interpolated datasets contained in the scan.
    """

    class Processed(DisplayDict):

        def get_arr(self, detector):
            f"""{SGMScan.DataDict.get_arr.__doc__}"""
            if isinstance(detector, str):
                try:
                    return self.data.filter(regex=f'{detector}.*').to_numpy()
                except AttributeError:
                    warnings.warn(f"No dataframe loaded in processed dictionary.")

        def read(self, filename=None):
            f"""{SGMScan.DataDict.read.__doc__}"""
            if not filename:
                try:
                    filename = self.filename
                except AttributeError:
                    try:
                        filename = self.sample + ".nxs"
                    except AttributeError:
                        return []
            if os.path.exists(filename):
                try:
                    h5 = h5py.File(filename, 'r')
                except Exception as f:
                    warnings.warn(f"Could not open file, h5py raised: {f}")
                    return
            elif os.path.exists('./data/'):
                try:
                    h5 = h5py.File(filename.replace('/home/jovyan/', './'), 'r')
                except Exception as f:
                    warnings.warn(f"Could not open file, h5py raised: {f}")
                    return
            else:
                try:
                    h5 = h5pyd.File(filename, "r", config.get("h5endpoint"), username=config.get("h5user"),
                                    password=config.get("h5pass"))
                except Exception as f:
                    warnings.warn(f"Could not open file, h5pyd raised: {f}")
                    return
            NXentries = [str(x) for x in h5['/'].keys()
                         if 'NXentry' in str(h5[x].attrs.get('NX_class'))]
            NXdata = [entry + "/" + str(x) for entry in NXentries for x in h5['/' + entry].keys()
                      if 'NXdata' in str(h5[entry + "/" + x].attrs.get('NX_class'))]
            axes = [[str(nm) for nm in h5[nxdata].keys() for s in h5[nxdata].attrs.get('axes') if str(s) in str(nm) or
                     str(nm) in str(s)] for nxdata in NXdata]
            indep_shape = [v.shape for i, d in enumerate(NXdata) for k, v in h5[d].items() if k in axes[i][0]]

            data = [{k.replace('_processed', ''): np.squeeze(v) for k, v in h5[d].items() if
                     v.shape[0] == indep_shape[i][0]} for i, d in
                    enumerate(NXdata)]
            df_sdds = [pd.DataFrame(
                {k.replace('_processed', '') + f"-{j}": v[:, j] for k, v in data[i].items() if len(v.shape) == 2 for j
                 in range(0, v.shape[1])})
                for i, _ in enumerate(NXdata)]
            processed_axes = [[ax.replace('_processed', '') for ax in x] for x in axes]
            df_scas = [pd.DataFrame.from_dict(
                {k.replace('_processed', ''): v for k, v, in data[i].items() if len(v.shape) < 2}).join(
                df_sdds[i]).groupby(processed_axes[i]).mean()
                       for i, _ in enumerate(NXdata)]
            if len(df_scas) == 1:
                self.data = df_scas[0]
            return df_scas

        def write(self, filename=None):
            """
            ### Description:
            -----
                Write data to NeXuS formatted data file.
            ### Keyword:
            -----
                >**filename** *(str / os.path)* -- path/name of file for output.
            """
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
            f"""{SGMScan.DataDict.plot.__doc__}"""
            if 'type' in self.__dict__.keys():
                scantype = self['type']
            else:
                try:
                    if 'scan' in self.command[0] and "en" == self.command[1]:
                        scantype = 'EEMS'
                    elif 'mesh' in self.command[0]:
                        scantype = 'XRF'
                except AttributeError:
                    try:
                        if len(self.data.index.names) == 1:
                            scantype = 'EEMS'
                        elif len(self.data.index.names) == 2:
                            scantype = 'XRF'
                    except AttributeError:
                        return
            if scantype == 'EEMS':
                keys = eemscan.required
                df = self.data
                roi_cols = df.filter(regex="sdd[1-4]_[0-2].*").columns
                df.drop(columns=roi_cols, inplace=True)
                data = {k: df.filter(regex=("%s.*" % k), axis=1).to_numpy() for k in keys}
                data.update({df.index.name: np.array(df.index), 'emission': np.linspace(0, 2560, 256)})
                data.update({'image': data['sdd1']})
                kwargs.update(data)
                return eemscan.plot(**kwargs)
            elif scantype == 'XRF':
                keys = xrfmap.required
                df = self.data
                roi_cols = df.filter(regex="sdd[1-4]_[0-2].*").columns
                df.drop(columns=roi_cols, inplace=True)
                data = {k: df.filter(regex=("%s.*" % k), axis=1).to_numpy() for k in keys}
                data = {k: v for k, v in data.items() if v.size}
                data.update({n: df.index.levels[i] for i, n in enumerate(list(df.index.names))})
                data.update({'emission': np.linspace(0, 2560, 256)})
                kwargs.update(data)
                xrfmap.plot_interp(**kwargs)

    def __init__(self, files, **kwargs):
        self.__dict__.update(kwargs)
        if not isinstance(files, list):
            files = [files]
        if not hasattr(self, 'npartitions'):
            self.npartitions = 3
        if not hasattr(self, 'threads'):
            self.threads = 4
        self.user = kwargs.get('user', os.environ.get('JUPYTERHUB_USER'))
        self.shift = kwargs.get('shift', 0.5)
        files = [os.path.abspath(file) for file in files]
        # Not sure if this is important/works, but trying to make sure that dask workers have the right path for non-admin users.
        if not any([os.path.exists(f) for f in files]) and os.path.exists(f'/home/jovyan/data/{files[0]}'):
            files = [file.replace(f'/home/jovyan/data/{self.user}/', '/home/jovyan/data/') for file in files]
        if not any([os.path.exists(f) for f in files]) and os.path.exists(f'./data/'):
            files = [file.replace(f'/home/jovyan/', './') for file in files]
        try:
            files_sorted = sorted(files, key=(lambda x: datetime.datetime.strptime(time.ctime(os.path.getctime(x)),
                                                                                   '%a %b %d %H:%M:%S %Y')))
        except ValueError:
            # Following line modified so that self.scans will have the same contents regardless of OS.
            files_sorted = sorted([(os.path.normpath(k)).split('\\')[-1].split('/')[-1].split(".")[0] for k in files])
        self.scans = {(os.path.normpath(k)).split('\\')[-1].split('/')[-1].split(".")[0]: {} for k in files_sorted}

        self.interp_params = {}
        with ThreadPool(self.threads) as pool:
            L = list(tqdm(pool.imap_unordered(self._load_data, files), total=len(files), leave=False))
        err = [l['ERROR'] for l in L if 'ERROR' in l.keys()]
        L = [l for l in L if 'ERROR' not in l.keys()]
        if len(err):
            warnings.warn(f"Some scan files were not loaded: {err}")
            for e in err:
                del self.scans[e]
        self.scans.update({k: SGMScan(**v) for d in L for k, v in d.items()})
        self.entries = self.scans.items
        self.interpolated = False

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
        ### Description:
        -----
            Function loads data in from SGM data file, and using the command value groups the data as
            either independent, signal or other.
        """
        entries = {}
        # Try to open the file locally or from a url provided.
        file_root = (os.path.normpath(file)).split('\\')[-1].split('/')[-1].split(".")[0]
        if os.path.exists(file):
            try:
                h5 = h5py.File(file, 'r')
            except Exception as f:
                warnings.warn(f"Could not open file, h5py raised: {f}")
                return {"ERROR": file_root}
        else:
            try:
                h5 = h5pyd.File(file, "r", config.get("h5endpoint"), username=config.get("h5user"),
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
                if self.shift and 'cmesh' in scan['command'][0]:
                    indep[i][scan['command'][1]] = indep[i][scan['command'][1]].map_overlap(shift_cmesh,
                                                                                            depth=1,
                                                                                            boundary='reflect')
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
        """
        ### Description:
        Batch interpolation of underlying scans.  Creates the bins required for each independent axes to be histogrammed
        into for interpolation, then uses dask dataframe groupby commands to perform a linear interpolation.

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
        _interpolate = partial(self._interpolate, **kwargs)
        entries = []
        for file, val in self.entries():
            for key, entry in val.__dict__.items():
                entries.append(entry)
        with ThreadPool(self.threads) as pool:
            results = list(tqdm(pool.imap_unordered(_interpolate, entries), total=len(entries)))
        self.interpolated = True
        return results

    def _interpolate(self, entry, **kwargs):
        compute = kwargs.get('compute', True)
        if compute:
            try:
                return entry.interpolate(**kwargs)
            except Exception as e:
                print(f"Exception raise while interpolating {entry}: {e}")
                return None
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
                    l = average[key] + OneList([
                        SGMData.Processed(command=command.split('_'), data=df, signals=v['signals'], sample=key)])
                    average.update({key: l})
                else:
                    average.update({key: OneList([
                        SGMData.Processed(command=command.split('_'), data=df, signals=v['signals'], sample=key)])})
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
            for subkey in self.scans[key].keys():
                table.append(f"<tr><th>{key}</th><td>{subkey}</td>" + self.scans[key][subkey]._repr_html_() + "</tr>")
        table.append("</tbody></table>")

        return "\n".join(table)

    def _repr_console_(self):
        if sys_has_tab:
            table = []
            temp_list = []
            for key in self.scans.keys():
                for subkey in self.scans[key]:
                    temp_list.append(key)
                    temp_list.append(subkey)
                    temp_list.append(self.scans[key].__dict__[subkey].sample)
                    temp_list.append(self.scans[key].__dict__[subkey].command)
                    temp_list.append(self.scans[key].__dict__[subkey].independent)
                    temp_list.append(self.scans[key].__dict__[subkey].signals)
                    temp_list.append(self.scans[key].__dict__[subkey].other)
                    table.append(temp_list)
                    temp_list = []
            return tabulate(table, headers=["File", "Entry", "Sample", "Command", "Independent", "Signals", "Other"])
        else:
            final_str = ""
            for key in self.scans.keys():
                for subkey in self.scans[key]:
                    temp_str = ("Entry:\t" + str(subkey))
                    temp_str = (temp_str + "\t\t\tFile: " + str(key))
                    temp_str = (temp_str + "\t\t\tSample: " + str(self.scans[key].__dict__[subkey].sample))
                    temp_str = (temp_str + "\t\t\tCommand: " + str(self.scans[key].__dict__[subkey].command))
                    temp_str = (temp_str + "\t\t\tIndependent: " + str(self.scans[key].__dict__[subkey].independent))
                    temp_str = (temp_str + "\t\t\tSignals: " + str(self.scans[key].__dict__[subkey].signals))
                    temp_str = (temp_str + "\t\t\tOther: " + str(self.scans[key].__dict__[subkey].other) + "\n")
                    final_str = str(final_str) + str(temp_str)
            return final_str
