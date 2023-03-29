import os
import inspect
from . import config
from slugify import slugify
import hashlib
import h5pyd
import numpy as np
from collections import Counter
from .load import SGMData
from .utilities.magicclass import OneList, DisplayDict
import warnings
import datetime
from .sign import get_or_make_key, get_proposals, find_samples, find_data, find_report, SGMLIVE_URL

try:
    import psycopg2
except ImportError:
    warnings.warn("Using sgm-data without database support.  SGMQuery will fail.")

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


# Get file path list from SGMLive database
class SGMQuery(object):
    """
    ### Description:
    >You can find your data in the SGMLive database by using the SGMQuery module (when using the [SGM JupyterHub](
     https://sgm-hub.lightsource.ca) ). The following documentation details the keywords that you can use to customize your
     search.

    ### Keywords:
    >**sample** *(str:required)* -- At minimum you'll need to provide the keyword "sample", corresponding the sample
                                    name in the database as a default this will grab all the data under that sample
                                    name.

    >**proposal** *(str: optional) -- proposal that the sample was measured under.

    >**kind** *(str: optional) -- Dataset type, this is an acronym from SGMLive, e.g. XAS, EEMS, uXRF, and etc.

    >**daterange** *(tuple:optional)* -- This can be used to sort through sample data by the day that it was
                                        acquired. This is designed to take a tuple of the form ("start-date",
                                        "end-date") where the strings are of the form "YYYY-MM-DD". You can also
                                        just use a single string of the same form, instead of a tuple, this will
                                        make the assumption that "end-date" == now().

    >**data** *(bool:optional)* -- As a default (True) the SGMQuery object will try to load the the data from disk,
                                    if this is not the desired behaviour set data=False.

    >**user** *(str:optional:staffonly)* -- Can be used to select the username in SGMLive from which the sample query is
                                            performed. Not available to non-staff.

    >**processed** *(bool:optional)* -- Can be used to return the paths for the processed data (already interpolated) instead
                                        of the raw. You would generally set data = False for this option.

    ### Attributes:
    >**data** *(object)* --  By default the query will create an SGMData object containing your data, this can be turned off
                                 with the data keyword.

    >**paths** *(list)* -- Contains the local paths to your data (or processed_data if processed=True).

    ### Example Usage:
    ```python
    from sgmdata import SGMQuery

    sgmq = SGMQuery(sample="TiO2 - C")
    data = sgmq.data
    data.averaged['TiO2 - C'].plot()
    ```
    """
    kind_dict = {"XRF Map": 8, "XAS Dataset": 5, "EEMs": 10, "Other": 11, "TEY Map": 9, "XRF Dataset": 6}

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.user = kwargs.get('user', os.environ.get('USER', os.environ.get('JUPYTERHUB_USER', None)))
        if not self.user:
            raise Exception("Unable to determine user account from environment, try specifying the user keyword")
        self.signer = get_or_make_key(self.user)
        self.proposals = kwargs.get('proposal', get_proposals(self.user, self.signer))
        if not isinstance(self.proposals, list):
            self.proposals = [self.proposals]
        self.type = kwargs.get('kind', None)
        data = kwargs.get('data', True)
        self.processed = kwargs.get('processed', False)
        self.daterange = kwargs.get('daterange', ())
        if isinstance(self.daterange, tuple) and len(self.daterange) == 2:
            if not isinstance(self.daterange[0], datetime.date) or not isinstance(self.daterange[1], datetime.date):
                try:
                    firstdate = datetime.datetime.strptime(self.daterange[0], '%Y-%m-%d')
                    enddate = datetime.datetime.strptime(self.daterange[1], '%Y-%m-%d')
                except ValueError:
                    raise ValueError("Incorrect data format, should be (YYYY-MM-DD. YYYY-MM-DD), or YYYY-MM-DD")
                self.daterange = (firstdate, enddate)
        elif isinstance(self.daterange, str):
            try:
                firstdate = datetime.datetime.strptime(self.daterange, '%Y-%m-%d')
            except ValueError:
                raise ValueError("Incorrect data format, should be (YYYY-MM-DD. YYYY-MM-DD), or YYYY-MM-DD")
            self.daterange = (firstdate, datetime.datetime.utcnow())
        self.datasets = {}
        self.data = DisplayDict()
        self.paths = DisplayDict()
        self.sessions = DisplayDict()
        self.interp_paths = DisplayDict()
        self.avg_paths = DisplayDict()
        self.samples = {}
        self.data_hash = ""
        self.get_datasets(data)

    def get_datasets(self, data):
        prepend = "/beamlinedata/SGM/projects"
        if os.path.exists("/SpecData/SGM/projects"):
            prepend = "/beamlinedata/SGM/projects"

        for p in self.proposals:
            samples = find_samples(self.user, self.signer, p, name=self.sample)
            self.datasets[p] = DisplayDict()
            self.samples[p] = OneList([])
            for s in samples:
                name = s['name'].strip()
                self.samples[p].append(name)
                sgmdata = OneList([])
                for d in find_data(self.user, self.signer, p, sample=s['id'], kind=self.type):
                    key = f"{d['id']}"
                    self.paths[key] = [f"{prepend}{d['directory']}raw/{f}.nxs" for f in d['files']]
                    self.sessions[key] = d['session']
                    d.update(
                        {'paths': self.paths[key]})
                    if data:
                        d.update({'data': SGMData(d['paths'])})
                    if self.processed:
                        reports = []
                        if self.type:
                            kind = self.type + " Report"
                        else:
                            kind = None
                        for r in find_report(self.user, self.signer, p, data=d['id'], kind=kind):
                            if 'binned' in r['files'].keys():
                                self.interp_paths[key] = [f"{prepend}/{r['url']}/binned/{f}.nxs" for f in r['files']['binned']]
                                r.update(
                                    {'paths': self.interp_paths[key]}
                                )
                                if data:
                                    for i, sgmscan in enumerate(d['data'].scans.values()):
                                        for entry in list(sgmscan.__dict__.values()):
                                            entry.read(filename=r['paths'][i])
                            if 'average' in r['files'].keys():
                                self.avg_paths[key] = f"{prepend}/{r['url']}/{r['files']['average']}.nxs"
                                r.update(
                                    {'avg': self.avg_paths[key]}
                                )
                                if data:
                                    processed = SGMData.Processed(sample=s['name'])
                                    processed.read(filename=r['avg'])
                                    d['data'].averaged = {processed['sample']: OneList([processed])}

                            reports.append(r)
                        d.update({"reports": reports})
                    if data:
                        self.data[key] = d['data']
                    sgmdata.append(DisplayDict(d))
                if name in self.datasets[p].keys():
                    self.datasets[p][name] = OneList([*self.datasets[p][name], *sgmdata])
                else:
                    self.datasets[p][name] = OneList(sgmdata)
        self.datasets = DisplayDict({k: v for k, v in self.datasets.items() if v})

    def write_proc(self, pk: str):
        paths = self.paths[pk]
        session = self.sessions[pk]
        self.interp_paths[pk] = [p.split('raw')[0] + f"preprocessed/{session}/" for p in paths]

    def _repr_html_(self):
        kind = [v for k, v in self.kind_dict.items() if self.type in k]
        table = [
            "<table>",
            "  <thead>",
            "    <tr><th>ID </th><th> Proposal </th><th> Group </th><th> Sample </th><th>"
            " DataType </th><th> # Scans </th>",
            "  </thead>",
            "  <tbody>",
        ]
        for key in self.datasets.keys():
            for subkey in self.datasets[key].keys():
                for d in self.datasets[key][subkey]:
                    if kind:
                        table.append(f"<tr><th><a href='{SGMLIVE_URL}/data/?proposal__name={key}&kind__id__exact={kind[0]}&search={d['id']}'>{d['id']}</a></th><td>{key}</td><td>{d['group']}</td>"
                                 f"<td>{d['sample']}</td><td>{d['kind']}</td><td>{d['num_scans']}</td></tr>")
                    else:
                        table.append(f"<tr><th><a href='{SGMLIVE_URL}/data/?proposal__name={key}&search={d['id']}'>{d['id']}</a></th><td>{key}</td><td>{d['group']}</td>"
                                 f"<td>{d['sample']}</td><td>{d['kind']}</td><td>{d['num_scans']}</td></tr>")
        table.append("</tbody></table>")

        return "\n".join(table)