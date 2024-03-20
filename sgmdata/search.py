import os
from . import config
from slugify import slugify

from .load import SGMData
from .utilities.util import sumROI
from .utilities.magicclass import OneList, DisplayDict
import numpy as np
import pandas as pd
import warnings
import datetime
from .sign import get_or_make_key, get_proposals, find_samples, find_data, find_report, add_report, SGMLIVE_URL


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
    display = repr
    HTML = print
    clear_output = list


# Get file path list from SGMLive database
class SGMQuery(object):
    """
    ### Description:
    >You can find your data in the SGMLive database by using the SGMQuery module (when using the CLS HPC & OPIs ). The following documentation details the keywords that you can use to customize your
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

    >**data_id** *(int:optional)* -- Primary key of the specific dataset requested.

    ### Attributes:
    >**data** *(object)* --  By default the query will create an SGMData object containing your data, this can be turned off
                                 with the data keyword.

    >**paths** *(list)* -- Contains the local paths to your data (or processed_data if processed=True).

    ### Example Usage:
    ```python
    from sgmdata import SGMQuery

    sgmq = SGMQuery(user="username", proposal="38GXXXXX", sample="TiO2 - C")
    data = sgmq.data['11111']
    data.averaged['TiO2 - C'].plot()
    ```
    """
    kind_dict = {"XRF Map": 8, "XAS Dataset": 5, "EEMs": 10, "Other": 11, "TEY Map": 9, "XRF Dataset": 6}
    prepend = config.get("prepend", "/beamlinedata/SGM/projects")

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.user = kwargs.get('user', os.environ.get('USER', os.environ.get('JUPYTERHUB_USER', None)))
        if not self.user:
            raise Exception("Unable to determine user account from environment, try specifying the user keyword")
        self.signer = get_or_make_key(self.user)
        self.proposal_list = kwargs.get('proposal', get_proposals(self.user, self.signer))
        if not isinstance(self.proposal_list, list):
            self.proposal_list = [self.proposal_list]
        self.sample = kwargs.get('sample', '')
        self.pk = kwargs.get('data_id', '')
        if not self.pk:
            self.pk = kwargs.get('pk', '')
        self.type = kwargs.get('kind', '')
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
        self.data = DisplayDict()
        self.paths = DisplayDict()
        self.sessions = DisplayDict()
        self.proposals = DisplayDict()
        self.names = DisplayDict()
        self.endstations = DisplayDict()
        self.energies = DisplayDict()
        self.stretches = DisplayDict()
        self.reports = DisplayDict()
        self.interp_paths = DisplayDict()
        self.avg_paths = DisplayDict()
        self.report_ids = DisplayDict()
        self.samples = DisplayDict()
        self.groups = DisplayDict()
        self.num_scans = DisplayDict()
        self.kind = DisplayDict()
        self.data_hash = ""
        if self.pk:
            self.get_data(data)
        else:
            self.get_datasets(data)


    def get_data(self, data):
        for p in tqdm(self.proposal_list, desc="Searching Proposals"):
            fdata = find_data(self.user, self.signer, p, kind=self.type, data=self.pk)
            for d in fdata:
                key = f"{d['id']}"
                self.paths[key] = [f"{self.prepend}{d['directory']}raw/{f}.nxs" for f in d['files']]
                self.sessions[key] = d['session']
                self.proposals[key] = p
                self.energies[key] = d['energy']
                self.stretches[key] = (d['start'], d['end'])
                self.samples[key] = d['sample']
                self.groups[key] = d['group']
                self.names[key] = d['name']
                self.num_scans[key] = d['num_scans']
                self.kind[key] = d['kind']
                d.update(
                    {'paths': self.paths[key]})
                if data:
                    d.update({'data': SGMData(d['paths'], progress=False)})
                if self.processed:
                    reports = []
                    if self.type:
                        kind = self.type + " Report"
                    else:
                        kind = None
                    for r in find_report(self.user, self.signer, p, data=d['id'], kind=kind):
                        if 'binned' in r['files'].keys():
                            self.interp_paths[key] = [f"{self.prepend}/{f}" for f in
                                                      r['files']['binned']]
                            r.update(
                                {'paths': self.interp_paths[key]}
                            )
                            if data:
                                for i, sgmscan in enumerate(d['data'].scans.values()):
                                    for entry in list(sgmscan.__dict__.values()):
                                        entry.read(filename=r['paths'][i])
                                d['data'].interpolated = True
                        if 'average' in r['files'].keys():
                            self.avg_paths[key] = [f"{f}"
                                                   for f in r['files']['average']]
                            r.update(
                                {'avg': self.avg_paths[key]}
                            )
                            if data and len(r['avg']):
                                processed = SGMData.Processed(sample=d['name'])
                                processed.read(filename=r['avg'][0])
                                d['data'].averaged = DisplayDict({processed['sample']: OneList([processed])})

                        reports.append(r)
                    self.reports[key] = reports
                    d.update({"reports": reports})
                if data:
                    self.data[key] = d['data']

    def get_datasets(self, data):
        for p in tqdm(self.proposal_list, desc="Searching Proposals"):
            samples = find_samples(self.user, self.signer, p, name=self.sample)
            for s in samples:
                name = s['name'].strip()
                fdata = find_data(self.user, self.signer, p, sample=s['id'], kind=self.type, data=self.pk)
                if self.daterange:
                    fdata = [f for f in fdata
                             if datetime.datetime.strptime(f['start'], "%Y-%m-%dT%H:%M:%SZ") > self.daterange[0]
                             and datetime.datetime.strptime(f['end'][:19], "%Y-%m-%dT%H:%M:%S") < self.daterange[1]]
                for d in fdata:
                    key = f"{d['id']}"
                    self.paths[key] = [f"{self.prepend}{d['directory']}raw/{f}.nxs" for f in d['files']]
                    self.sessions[key] = d['session']
                    self.proposals[key] = p
                    self.energies[key] = d['energy']
                    self.stretches[key] = (d['start'], d['end'])
                    self.samples[key] = name
                    self.groups[key] = d['group']
                    self.names[key] = d['name']
                    self.num_scans[key] = d['num_scans']
                    self.kind[key] = d['kind']
                    d.update(
                        {'paths': self.paths[key]})
                    if data:
                        d.update({'data': SGMData(d['paths'], progress=False)})
                    if self.processed:
                        reports = []
                        if self.type:
                            kind = self.type + " Report"
                        else:
                            kind = None
                        for r in find_report(self.user, self.signer, p, data=d['id'], kind=kind):
                            if 'binned' in r['files'].keys():
                                self.interp_paths[key] = [f"{self.prepend}/{f}" for f in r['files']['binned']]
                                r.update(
                                    {'paths': self.interp_paths[key]}
                                )
                                if data:
                                    for i, sgmscan in enumerate(d['data'].scans.values()):
                                        for entry in list(sgmscan.__dict__.values()):
                                            entry.read(filename=r['paths'][i])
                                    d['data'].interpolated = True
                            if 'average' in r['files'].keys():
                                self.avg_paths[key] = [f"{f}"
                                                       for f in r['files']['average']]
                                r.update(
                                    {'avg': self.avg_paths[key]}
                                )
                                if data:
                                    processed = SGMData.Processed(sample=s['name'])
                                    processed.read(filename=r['avg'][0])
                                    d['data'].averaged = DisplayDict({processed['sample']: OneList([processed])})

                            reports.append(r)
                        self.reports[key] = reports
                        d.update({"reports": reports})
                    if data:
                        self.data[key] = d['data']

    def write_processed(self, pk: str, type: str):
        paths = self.paths[pk]
        session = self.sessions[pk]
        proposal = self.proposals[pk]
        title = f"{slugify(self.names[pk])}-report-{self.report_ids[pk]}"
        data = self.data[pk]
        self.interp_paths[pk] = OneList([])

        if 'XAS' in type.upper():
            interp = {p.split('/')[-1].split('.')[0] : f"/prj{proposal}/preprocessed/{session}/{title}/binned/{p.split('/')[-1].split('.')[0]}.h5" for p in paths}
        elif 'MAP' in type.upper() or 'STACK' in type.upper():
            interp = {p.split('/')[-1].split('.')[0] : f"/prj{proposal}/preprocessed/{session}/{title}/{p.split('/')[-1].split('.')[0]}.h5" for p in paths}
        folder = self.prepend + "/" + "/".join(next(iter(interp.values())).split('/')[:-1])
        if not os.path.exists(folder):
            os.makedirs(folder)
        for k, v in interp.items():
            if k in data.scans.keys():
                for entry in data.scans[k].__dict__.values():
                    entry.write(self.prepend + v)
                    self.interp_paths[pk].append(v)

    def write_average(self, pk: str, kind='XAS'):
        data = self.data[pk]
        session = self.sessions[pk]
        proposal = self.proposals[pk]
        title = f"{slugify(self.names[pk])}-report-{self.report_ids[pk]}"
        self.avg_paths[pk] = OneList([])
        for k, v in data.averaged.items():
            average = v
            if not isinstance(average, list):
                average = [v]
            for a in average:
                if 'XAS' in kind.upper():
                    associated = [f"./binned/{p[0]}.h5" for p in a.associated]
                else:
                    associated = [f"./{p[0]}.h5" for p in a.associated]
                path = f"{self.prepend}/prj{proposal}/preprocessed/{session}/{title}/{slugify(k)}.h5"
                a.write(path, associated=associated)
                self.avg_paths[pk].append(path)

    def create_csvs(self, pk, mcas=None, **kwargs):
        data = self.data[pk]
        proposal = self.proposals[pk]
        session = self.sessions[pk]
        title = f"{slugify(self.names[pk])}-report-{self.report_ids[pk]}"
        name = slugify(self.names[pk])

        ## Set default detector list for ROI summing.
        if mcas is None:
            mcas = ['sdd1', 'sdd2', 'sdd3', 'sdd4']

        ## Prepare data output directory.
        out = kwargs.get('out', f'{self.prepend}/prj{proposal}/preprocessed/{session}/{title}')
        if not os.path.exists(out):
            os.makedirs(out)

        ## Load in I0 if exists:
        i0 = kwargs.get('I0', None)

        ## Get ROI bounds:
        roi = kwargs.get('ROI', (0, 255))

        ## Are the scans step scans?
        step = kwargs.get('step', False)

        ## Find and collect data.
        dfs = []

        if len(data.averaged):
            ## get or create processed data.
            s = [k for k in data.averaged.keys()][0]
            averaged = data.averaged[s]

            ## extract SDDs
            df = averaged['data']['i0']
            sdd_tot = []
            for k, v in averaged['data'].items():
                if 'i0' not in k:
                    if len(v.columns) == 1:
                        df[k] = v
                    elif k in mcas and len(v.columns) == 256:
                        mca = averaged.get_arr(k)
                        temp = sumROI(mca, start=int(roi[0]/10), stop=int(roi[1]/10))
                        df[k] = temp
                        sdd_tot.append(temp)
            ## Should this be averaged?
            df['sdd_total'] = np.nansum(sdd_tot, axis=0)
            if isinstance(i0, pd.DataFrame):
                df = df.join(i0)
            elif isinstance(i0, pd.Series):
                df['i0_aux'] = i0
            df.to_csv(out + '/' + name + f'_ROI-{int(roi[0])}_{int(roi[1])}.csv')
            dfs.append(df)
        return dfs

    def post_report(self, pk: str, type: str, report: list, score: float, report_id=None):
        session = self.sessions[pk]
        proposal = self.proposals[pk]
        title = slugify(self.names[pk])
        files = {}
        if 'XAS' in type.upper() and pk in self.interp_paths.keys() and pk in self.avg_paths.keys():
            files = {"binned": list(self.interp_paths[pk]), "average": list(self.avg_paths[pk])}
        if 'MAP' in type.upper() and pk in self.interp_paths.keys():
            files = {"binned": list(self.interp_paths[pk])}
        data_dict = {
            "proposal": proposal,
            "score": score,
            "kind": type,
            "details": report,
            "title": title,
            "files": files,
            "directory": f"prj{proposal}/preprocessed/{session}/{title}-report-{report_id}",
            "data_id": [pk],
        }
        if report_id:
            data_dict.update({'id': report_id})
        resp = add_report(self.user, self.signer, data_dict)
        if not resp:
            raise ValueError("No report created.")
        self.report_ids[pk] = resp['id']
        return f"{SGMLIVE_URL}/reports/{resp['id']}"

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
        for pk in self.proposals.keys():
            p = self.proposals[pk]
            g = self.groups[pk]
            s = self.samples[pk]
            n = self.num_scans[pk]
            k = self.kind[pk]
            if kind:
                table.append(f"<tr><th><a href='{SGMLIVE_URL}/data/?proposal__name={p}&kind__id__exact={kind[0]}&search={pk}'>{pk}</a></th><td>{p}</td><td>{g}</td>"
                         f"<td>{s}</td><td>{k}</td><td>{n}</td></tr>")
            else:
                table.append(f"<tr><th><a href='{SGMLIVE_URL}/data/?proposal__name={p}&search={pk}'>{pk}</a></th><td>{p}</td><td>{g}</td>"
                         f"<td>{s}</td><td>{k}</td><td>{n}</td></tr>")
        table.append("</tbody></table>")

        return "\n".join(table)