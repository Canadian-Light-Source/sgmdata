import os
from . import config
from slugify import slugify
import hashlib
import psycopg2
import sgmdata
import h5pyd
import numpy as np
from collections import Counter
from dask.distributed import Client
from .utilities import h5tree, scan_health
from .load import SGMData
import datetime
import warnings
from tqdm.notebook import tqdm

try:
    from IPython.display import display, HTML, clear_output
except ImportError:
    pass

# Get file path list from SGMLive database
class SGMQuery(object):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        try:
            self.admin = os.environ['JHUB_ADMIN']
        except KeyError:
            raise Exception("SGMQuery can only be run inside sgm-hub.lightsource.ca at the moment.")
        self.admin = int(self.admin)
        if self.admin:
            self.user = kwargs.get('user', os.environ['JUPYTERHUB_USER'])
        else:
            self.user = os.environ['JUPYTERHUB_USER']
        if 'data' not in kwargs.keys():
            self.data = True
        if 'processed' not in kwargs.keys():
            self.processed = False
        if 'daterange' not in kwargs.keys():
            self.daterange = ()
        elif isinstance(self.daterange,tuple) and len(self.daterange) == 2:
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
        self.connection = psycopg2.connect(database=config.get('db_env_db'), user=config.get('db_env_postgres_user'), password=config.get('db_env_secret'),
                                           host=config.get('db_port_5432_tcp_addr'), port=config.get('db_port_5432_tcp_port'))
        self.cursor = self.connection.cursor()
        self.project_id = None
        self.sample_id = None
        self.paths = []
        self.scan_ids = {}
        self.processed_ids = []
        self.domains = []
        self.avg_id = []
        self.date_hash = ""
        self.get_paths()
        if self.paths:
            if os.path.exists("/".join(self.paths[0].split('/')[:-1]).replace('/home/jovyan', "./")):
                local_paths = [p.replace('/home/jovyan', '.') for p in self.paths]
                self.paths = local_paths
            elif os.path.exists("/".join(self.paths[0].split('/')[:-1]).replace('/home/jovyan', "/SpecData")):
                local_paths = [p.replace('/home/jovyan', '/SpecData') for p in self.paths]
                self.paths = local_paths
        if self.data:
            self.data = SGMData(self.paths, **kwargs)

    def get_paths(self):
        self.cursor.execute("SELECT id, name from lims_project WHERE name IN ('%s');" % self.user)
        row = self.cursor.fetchone()
        if row:
            self.project_id = row[0]
        else:
            print(f"No account '{self.user}' found.")
            return []

        SQL = "SELECT id from lims_xassample WHERE project_id = %d AND name = '%s'" % (self.project_id, self.sample)
        self.cursor.execute(SQL)
        row = self.cursor.fetchone()
        if row:
            self.sample_id = row[0]
        else:
            print(f"No sample, {self.sample}, in account {self.user}.")
            return []
        if self.daterange:
            SQL = "SELECT id, domain, \"group\", start_time from lims_xasscan WHERE project_id = %d AND sample_id = %d " \
                  "AND (start_time BETWEEN '%s' AND '%s');" % \
                  (
                      self.project_id, self.sample_id, self.daterange[0], self.daterange[1]
                  )
        else:
            SQL = "SELECT id, domain, \"group\", start_time from lims_xasscan WHERE project_id = %d AND sample_id = %d;" % \
                  (
                      self.project_id, self.sample_id
                  )
        self.cursor.execute(SQL)
        domains = self.cursor.fetchall()
        if len(domains):
            self.date_hash = hashlib.md5(f"{str(domains[0][3])}-{str(domains[-1][3])}".encode("utf-8"))
        if self.processed:
            if not len(domains):
                return []
            SQL = "SELECT average_id, domain, id FROM lims_xasprocessedscan WHERE xasscan_id IN ("
            for ID in domains:
                SQL += "'%d', " % ID[0]
            SQL = SQL[:-2] + ");"

            self.cursor.execute(SQL)
            average_ids = self.cursor.fetchmany(500)
            if not average_ids:
                print(f"No processed scans found for, {self.sample}, in account {self.user}.")
                return []
            avg_ids = [i[0] for i in average_ids]

            procdomains = [i[1] for i in average_ids]
            self.processed_ids = [i[2] for i in average_ids]

            # Get most common average scan id.
            f = Counter(avg_ids)
            most_common = [i for i in f.most_common()]
            if most_common:
                if most_common[0][0] == None and len(most_common) > 1:
                    mc = most_common[1][0]
                else:
                    mc = most_common[0][0]
            else:
                return []
            self.avg_id = [mc]
            if not self.avg_id[0]:
                print(f"No average scan found for, {self.sample}, in account {self.user}.")
                return []

            SQL = "SELECT domain from lims_xasscanaverage WHERE project_id = %d AND id = %d;" % \
                  (
                      self.project_id, self.avg_id[0]
                  )

            self.cursor.execute(SQL)
            self.avg_domain = self.cursor.fetchone()
            if self.avg_domain:
                self.avg_domain = [self.avg_domain[0]]
            else:
                print(f"Average scan for {self.sample}, is in a different account.")
                return []

            if self.admin:
                self.paths = ["/home/jovyan/data/" + d.split('.')[1] + "/" + d.split('.')[0] + '.nxs' for d in
                              procdomains]
            else:
                self.paths = ["/home/jovyan/data/" + d.split('.')[0] + '.nxs' for d in procdomains]

        else:
            if self.admin:
                self.paths = ["/home/jovyan/data/" + d[1].split('.')[1] + "/" + d[1].split('.')[0] + '.nxs' for d in domains]
            else:
                self.paths = ["/home/jovyan/data/" + d[1].split('.')[0] + '.nxs' for d in domains]
        file_dict = {d[1].split('.')[0]: {} for d in domains}
        for d in domains:
            for k in file_dict.keys():
                if k in d[1]:
                    file_dict[k].update({d[2]: d[0]})
        self.scan_ids = file_dict

    def updateStatus(self, table="lims_xasscan", item_id=None):
        if not item_id:
            if self.scan_id:
                item_id = [self.scan_id]
            else:
                self.__exit__()
                return
        if not isinstance(item_id, list):
            item_id = [item_id]
        now = datetime.datetime.utcnow()
        for item in item_id:
            SQL = "SELECT status FROM %s WHERE id = %d" % (table, item)
            self.cursor.execute(SQL)
            row = self.cursor.fetchone()
            if row:
                status = row[0]
            else:
                status = 0
            if status == 0:
                SQL = "UPDATE %s SET status = '%d', modified = '%s' WHERE id = '%d';" % (table, 5, now, item)
                self.cursor.execute(SQL)
            elif status == 5:
                SQL = "UPDATE %s SET status = '%d', modified = '%s' WHERE id = '%d';" % (table, 6, now, item)
                self.cursor.execute(SQL)
            elif status == 6:
                SQL = "UPDATE %s SET status = '%d', modified = '%s' WHERE id = '%d';" % (table, 7, now, item)
                self.cursor.execute(SQL)
        self.connection.commit()
        return True

    def addProcessedScantoDatabase(self, data, **kwargs):
        project = kwargs.get('project', self.project_id)
        sample = kwargs.get('name', self.sample)
        if not hasattr(self, 'connection'):
            self.connection = psycopg2.connect(database=self.db, user=self.posuser, password=self.secret,
                                               host=self.host, port='5080')
            self.cursor = self.connection.cursor()
        SQL = "SELECT id FROM lims_xasprocessedscan WHERE domain = '%s' AND project_id = %d;" % \
              (
                  data['domain'], project
              )
        now = datetime.datetime.utcnow()
        self.cursor.execute(SQL)
        row = self.cursor.fetchone()
        if row:
            SQL = "UPDATE lims_xasprocessedscan SET modified = '%s', resolution = %.2f WHERE id = '%d';" % (
            now, data['resolution'], row[0])
            self.processed_ids.append(row[0])
            self.cursor.execute(SQL)
            self.connection.commit()
        else:
            indep = [x + "_processed" for x in data['indep']]
            data.update(dict(project_id=project, name=sample, created=now, modified=now,
                             status=5, download=True, indep=indep))
            SQL = (
                f"INSERT INTO lims_xasprocessedscan (project_id, name, created, modified, xasscan_id, download, domain, \"group\","
                f"resolution, range, independent, status) VALUES ({data['project_id']}, '{data['name']}', '{data['created']}',"
                f"'{data['modified']}', {data['xasscan_id']},'t','{data['domain']}','{data['entry']}', {data['resolution']},"
                f"'{data['range']}', '{data['indep'][0]}', {data['status']}) RETURNING id;"
            )
            self.cursor.execute(SQL)
            proc_id = self.cursor.fetchone()
            self.processed_ids.append(proc_id[0])

            self.connection.commit()
            self.updateStatus(table='lims_xasscan', item_id=data['xasscan_id'])

            # Register new processed scan

    def addAverageScantoDatabase(self, data, **kwargs):
        project = kwargs.get('project', self.project_id)
        sample = kwargs.get('name', self.sample)
        processed = kwargs.get('processed', self.processed_ids)
        if not hasattr(self, 'connection'):
            self.connection = psycopg2.connect(database=self.db, user=self.posuser, password=self.secret,
                                               host=self.host, port='5080')
            self.cursor = self.connection.cursor()
        now = datetime.datetime.utcnow()
        SQL = "SELECT id FROM lims_xasscanaverage WHERE domain = '%s' AND project_id = %d;" % \
              (
                  data['domain'], project
              )
        self.cursor.execute(SQL)
        row = self.cursor.fetchone()
        if row:
            self.cursor.execute("""UPDATE lims_xasprocessedscan SET average_id = %s, modified = %s WHERE id IN %s ;""",
                                (row[0], now, tuple(processed))
                                )
            self.cursor.execute("""SELECT id FROM lims_xasprocessedscan WHERE average_id = %s AND id NOT IN %s ;""",
                                (row[0], tuple(processed)))
            t = self.cursor.fetchall()
            if t:
                self.cursor.execute("""UPDATE lims_xasprocessedscan SET average_id = %s, modified = %s WHERE id in %s ;""",
                                    (None, now, tuple([e[0] for e in t]))
                                    )
            self.cursor.execute("""UPDATE lims_xasscanaverage SET modified = %s WHERE id = %s;""",
                                (now, row[0])
                                )
            avg_id = row[0]
            self.connection.commit()
        else:
            indep = [x + "_processed" for x in data['indep']]
            data.update(
                dict(project_id=project, name=sample, created=now, modified=now,
                     status=5, entry='entry1/'), indep=indep)
            SQL = (f"INSERT INTO lims_xasscanaverage (project_id, name, created, modified, download, domain, \"group\","
                   f"status) VALUES ({data['project_id']}, '{data['name']}', '{data['created']}',"
                   f"'{data['modified']}', 't','{data['domain']}','{data['entry']}', {data['status']}) RETURNING id;"
                   )
            self.cursor.execute(SQL)
            avg_id = self.cursor.fetchone()
            self.cursor.execute("""UPDATE lims_xasprocessedscan SET average_id = %s, modified = %s WHERE id IN %s ;""",
                                (avg_id[0], now, tuple(processed))
                                )
            self.connection.commit()
        SQL = "SELECT id, name from lims_xassample WHERE name IN ('%s') AND project_id IN (%d);" % \
              (
                  sample, project
              )
        self.cursor.execute(SQL)
        row = self.cursor.fetchone()
        if row:
            self.updateStatus(table='lims_xassample', item_id=row[0])
        return avg_id

    def write_proc(self, file, **kwargs):
        domain_list = []
        pbar1 = tqdm(file.keys())
        for k in pbar1:
            pbar1.set_description("Saving")
            for entry in file[k].__dict__.keys():
                if 'binned' in file[k][entry].keys():
                    data = file[k][entry]['binned']['dataframe']
                else:
                    data = file[k][entry].interpolate()
                domain = ".".join(["processed_" + k, self.user, "vsrv-sgm-hdf5-01.clsi.ca"])
                try:
                    self.write(data, domain)
                    domain_list.append(domain)
                except Exception as e:
                    print("Error: %s \n %s" % (e, domain))
                    continue
                resolution = data.index[1] - data.index[0]
                rng = f"{data.index[0]} {data.index[-1]}"
                xasscan = self.scan_ids[k][entry]
                indep = [entry + '/data/' + ax for ax in file[k][entry].independent.keys()]
                sqldata = {'entry': entry, 'domain': domain, 'xasscan_id': xasscan,
                           'resolution': resolution, 'range': rng, 'indep': indep}
                self.addProcessedScantoDatabase(sqldata)
        return domain_list

    def write_avg(self, average, **kwargs):
        domain_list = []
        sgmlive_list = []
        if 'bad_scans' in kwargs.keys():
            processed = [d for i, d in enumerate(self.processed_ids) if i not in kwargs['bad_scans']]
        else:
            processed = self.processed_ids
        if processed:
            pbar1 = tqdm(average.keys())
            for k in pbar1:
                pbar1.set_description("Saving")
                for i, r in enumerate(average[k]):
                    data = r['data']
                    sample = slugify(self.sample)
                    domain = ".".join([sample + f"-{self.date_hash.hexdigest()}" + f"-{i}", self.user, "vsrv-sgm-hdf5-01.clsi.ca"])
                    try:
                        self.write(data, domain)
                        domain_list.append(domain)
                    except Exception as e:
                        print("Error: %s" % e)
                    indep = str(['entry1/data/' + ax for ax in data.index.names])
                    sqldata = {'domain': domain, 'indep': indep, 'name': self.sample}
                    avg = self.addAverageScantoDatabase(sqldata, processed=processed)
                    sgmlive_list.append("https://sgmdata.lightsource.ca/users/xasexperiment/useravg/%d" % avg)
                    self.avg_id.append(avg)
            self.avg_domain = domain_list
        return domain_list, sgmlive_list

    def write(self, data, domain, **kwargs):
        if 'signal' in kwargs.keys():
            signal = kwargs['signal']
        else:
            signal = 'sdd3_processed'
        if 'detectors' in kwargs.keys():
            detectors = kwargs['detectors']
        else:
            detectors = list(set([d.split('-')[0] for d in data.columns]))
        try:
            h5 = h5pyd.File(domain, "w", config.get("h5endpoint"), username=config.get("h5user"), password=config.get("h5pass"))
        except:
            h5 = h5pyd.File(domain, "a", config.get("h5endpoint"), username=config.get("h5user"), password=config.get("h5pass"))

        NXentries = [int(str(x).split("entry")[1]) for x in h5['/'].keys() if
                     'NXentry' in str(h5[x].attrs.get('NX_class'))]
        if NXentries:
            NXentries.sort()
            entry = 'entry' + str(NXentries[-1] + 1)
        else:
            entry = 'entry1'
        axes = [nm for nm in data.index.names]
        nxent = h5.create_group(entry)
        nxent.attrs.create(u'NX_class', u'NXentry')
        nxdata = nxent.create_group('data')
        nxdata.attrs.create(u'NX_class', u'NXdata')
        nxdata.attrs.create(u'axes', axes)
        nxdata.attrs.create(u'signal', signal)
        if len(axes) == 1:
            arr = np.array(data.index)
            nxdata.create_dataset(data.index.name + "_processed", arr.shape, data=arr, dtype=arr.dtype)
        elif len(axes) > 1:
            for i, ax in enumerate(axes):
                arr = np.array(data.index.levels[i])
                nxdata.create_dataset(ax + "_processed", arr.shape, data=arr, dtype=arr.dtype)
        for sig in detectors:
            arr = data.filter(regex="%s.*" % sig).to_numpy()
            if len(data.index.names) > 1:
                shape = [len(data.index.levels[i]) for i in range(len(data.index.levels))]
                shape += [s for s in arr.shape[1:]]
                shape = tuple([s for s in shape if s > 0])
                arr = np.reshape(arr, shape)
            nxdata.create_dataset(sig + "_processed", data=arr, dtype=arr.dtype)
        h5.close()


def badscans(interp, **kwargs):
    """
    Description:
    ----
    Batch calculation of sgmdata.utilities.scan_health for list of interpolated dataframes.

    Args:
    ____
        interp (list) :  list of SGMScan binned dataframes.

    Returns:
    ____
        List of indexes for bad scans in interp.

    """
    cont = kwargs.get('cont', 55)
    dump = kwargs.get('dump', 30)
    sat = kwargs.get('sat', 60)
    sdd_max = kwargs.get('sdd_max', 50000)
    bad_scans = []
    health = [scan_health(i, sdd_max=sdd_max) for i in interp]
    pbar = tqdm(health)
    for i,t in enumerate(pbar):
        pbar.set_description("Finding bad scans...")
        if t[0] > cont or t[1] > dump or t[2] > sat:
            print(i, t)
            bad_scans.append(i)
    return bad_scans

def preprocess(sample, **kwargs):
    """
    Description:
    ----
    Utility for automating the interpolation and averaging of a sample in the SGMLive website.

    Args:
    ----
        sample (str):  The name of the sample in your account that you wish to preprocess.
        kwargs: user (str) - name of user account to limit search to (for use by staff).
                resolution - to be passed to interpolation function, this is histogram bin width.
                start (float) -  start energy to be passed to interpolation function.
                stop (float) - stop energy to be passed to interpolation function.
                sdd_max (int) - threshold value to determine saturation in SDDs, to determine scan_health (default
                                is 105000).
                bscan_thresh (tuple) - (continuous, dumped, and saturated)  these are the threshold percentages from
                                    scan_health that will label a scan as 'bad'.

    Returns:
    ----
         (HTML) hyperlink for preprocessed data stored in SGMLive
    """
    user = kwargs['user'] = kwargs.get('user', False)
    bs_args = kwargs.get('bscan_thresh', dict(cont=55, dump=30, sat=60))
    sdd_max = kwargs.get('sdd_max', 105000)
    clear = kwargs.get('clear', True)
    query_return = kwargs.get('query', False)
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
        sgm_data = sgmdata.SGMData(sgmq.paths, **kwargs)
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
            print(f"Averaged {len(sgm_data.scans) - len(bscans)} scans for {sample}")
            del sgm_data
            if query_return:
                return sgmq
            return HTML(html)
        else:
            if clear:
                clear_output()
            warnings.warn(f"There were no scans that passed the health check for {sample}.")
