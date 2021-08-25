import json
import requests
import getpass
from bs4 import BeautifulSoup
import re
from .load import SGMData
from .search import preprocess, SGMQuery
import os
import matplotlib.pyplot as plt
import numpy as np
import h5py
import datetime
import itertools
from shutil import copyfile
from dask.distributed import Client



class ReportBuilder(object):

    def __init__(self, proposal, principal, cycle, session, shifts, **kwargs):
        self.__dict__.update(kwargs)
        if not isinstance(proposal, str) or not len(proposal) > 5:
            raise Exception("Need to provide a valid proposal string; e.g. 33G10000")
        self.proposal = proposal
        if not isinstance(principal, str):
            raise Exception("Prinicpal invesitigator needs to be string")
        self.principal = principal
        if not isinstance(cycle, int):
            raise Exception("Cycle needs to be of type int")
        if not cycle < 100:
            raise Exception("Cycle needs to be an integer < 50")
        self.cycle = cycle
        if not isinstance(session, int):
            raise Exception("Session needs to be of type int")
        self.session = session
        if not isinstance(shifts, int):
            raise Exception("Shifts needs to be of type int")
        self.shifts = shifts
        self.LOGIN_URL = "https://confluence.lightsource.ca/rest/api/content/152768104/child/page"
        if "login" in kwargs.keys():
            self.LOGIN_URL = kwargs['login']
        if "user" in kwargs.keys():
            self.username = kwargs['user']
        else:
            self.username = input("Enter username:")

        if "client" not in kwargs.keys():
            self.client = Client()
        self.exp_info = {}
        self.sample_lists = {}
        self.holders = {}
        self.holder_time_init = {}
        self.holder_notes = {}
        self.holder_edges = {}
        self.holder_scans = {}
        self.eems_log = {}
        self.scans_log = {}
        self.log = self.get_confluence_log()['body']
        if "account" in kwargs.keys():
            self.account = kwargs['account']
            self.find_samples_and_edges()
        else:
            acc = re.findall(r'\bRequesting data be acquired in account <span.*>([A-Za-z0-9]+)</span>, with', self.log)
            if acc:
                self.account = acc[0]
                self.find_samples_and_edges()
            else:
                print("Coulnd't find account information for project. Exiting.")

    def get_confluence_log(self):
        password = getpass.getpass("Enter password:")
        basicauth = (self.username, password)
        with requests.session() as s:
            resp = s.get(self.LOGIN_URL, auth=basicauth)
            del basicauth, password
            if resp.status_code != 200:
                print("Failed to fetch data from " + self.LOGIN_URL + ", with HTTP %d" % resp.status_code)
                return {"title": "", "body": ""}
            pages = json.loads(resp.text)

            for r in pages['results']:
                if self.proposal in r['title']:
                    DATA_URL = r['_links']['self']
                    break
            resp = s.get(DATA_URL + "?expand=body.storage")
            if resp.status_code != 200:
                print("Failed to fetch data from " + DATA_URL + ", with HTTP %d" % resp.status_code)
                return {"title": "", "body": ""}
            page = json.loads(resp.text)
            return {"title": page['title'], "body": page['body']['storage']['value']}

    def find_samples_and_edges(self):
        html = self.log
        soup = BeautifulSoup(html, 'html.parser')
        boilerplate = soup.find_all('p')
        header = []
        for bp in boilerplate[:10]:
            hd = re.sub(r'<(.*?>)', "", repr(bp))
            if len(hd) > 10:
                header.append(hd)
        rows = soup.find_all('tr')
        cols = rows[0].find_all('strong')
        columns = {}
        samples = []
        edges = []
        notes = []
        positions = []
        annotations = []
        holder = []
        for i, c in enumerate(cols):
            if "id" in repr(c.string).lower():
                columns.update({"name": i})
            elif "edge" in repr(c.string).lower():
                columns.update({"edges": i})
            elif "note" in repr(c.string).lower():
                columns.update({"notes": i})
            elif "holder" in repr(c.string).lower():
                columns.update({"holder": i})
        vals = [v for k, v in columns.items()]
        index = []
        for i, r in enumerate(rows[1:]):
            items = r.find_all('td')
            if len(items) >= max(vals) and len(items) > 1:
                index.append(i)
                sample = re.sub(r'<(.*?>)', "", repr(items[columns['name']]))
                samples.append(sample)
                edges.append(items[columns['edges']])
                notes.append(items[columns['notes']])
                positions.append(repr(items[columns['holder']]))
            else:
                annotations.append((i, items[0]))
        for note in annotations:
            annotes = []
            for p in note[1].find_all('p'):
                if 'note' in repr(p.string).lower():
                    annotes.append(repr(p.string).replace('\\xa0', ' '))
            for h in note[1].find_all('h3'):
                if 'holder' in repr(h.string).lower():
                    holder.append((note[0], h.string.strip(), annotes))
        holders = list()
        for i, p in enumerate(positions):
            try:
                alpha = re.findall(r'^<td.*>.*([A-Za-z])-[0-9]+.*</td>', p)[0]
            except:
                print("No position in", p)
                del edges[i]
                del samples[i]
                del notes[i]
                del index[i]
                continue
            find = [(h[1], h[2]) for h in holder if alpha in re.findall(r'^\bHolder\s([A-Za-z])\s*-', h[1])]
            if find:
                holders.append(find[0][0])
        positions = [p for j, p in enumerate(positions) if j in index]
        edges = [re.sub(r'<(.*?>)', "", str(e)) for e in edges]
        notes = [re.sub(r'<(.*?>)', "", str(n)) for n in notes]
        edges = [tuple([j.strip() for j in e.split('\xa0')[0].strip().replace(';', ',').split(',')]) for e in edges]
        edges = [tuple([t for t in e if t]) for e in edges]
        boilerplate = []
        self.exp_info.update({
            "bp": header,
            "samples": samples,
            "edges": edges,
            "notes": notes,
            "locations": positions,
            "holders": holders
        })
        holders = set(self.exp_info['holders'])
        try:
            self.holder_annote = {k[1]: k[2] for k in holder}
            self.holders = {k: [v for i, v in enumerate(self.exp_info['samples']) if self.exp_info['holders'][i] == k]
                            for k in holders}
            self.holder_scans = {k: [v + " - " + e for i, v in enumerate(self.exp_info['samples'])
                                     for e in edges[i] if self.exp_info['holders'][i] == k] for k in holders}
            self.holder_edges = {k: [e for i, e in enumerate(edges) if self.exp_info['holders'][i] == k] for k in
                                 holders}
            self.holder_notes = {
                k: ['%s' % (n if n else " ") for i, n in enumerate(notes) if self.exp_info['holders'][i] == k] for k in
                holders}
        except Exception as e:
            print("Couldn't assign sample scans to appropriate holder, check confluence formatting. Exception: %s" % e)

    def make_data(self, df):
        keys = ['image', 'sdd1', 'sdd2', 'sdd3', 'sdd4', 'tey', 'xp', 'yp', 'emission']
        data = {k: df.filter(regex=("%s.*" % k), axis=1).to_numpy() for k in keys}
        data.update({k: np.reshape(v, (len(df.index.levels[0]), len(df.index.levels[1]), v.shape[-1])) if len(
            v.shape) == 2 else np.reshape(v, (len(df.index.levels[0]), len(df.index.levels[1]))) for k, v in
                     data.items()})
        data.update({n: df.index.levels[i] for i, n in enumerate(list(df.index.names))})
        levels = [max(df.index.levels[0]), min(df.index.levels[0]), max(df.index.levels[1]), min(df.index.levels[1])]
        avg = np.mean([data['sdd1'], data['sdd2'], data['sdd3'], data['sdd4']], axis=0)
        avg = np.sum(avg[:, :, 45:55], axis=2)
        data.update({"extent": levels, "image": np.flip(avg.T, axis=1)})
        return data

    def get_sample_positions(self, paths, entries):
        positions = []
        if len(entries) != len(paths):
            print("Not enough entries (%d) for sample positions (%d) given." % (len(entries), len(paths)))
        for i, s in enumerate(paths):
            try:
                entry = entries[i]
            except:
                entry = 'entry1'
            with h5py.File(s, 'r') as f:
                try:
                    t = (float(f[entry + '/sample/positioner/xp'][()]),
                         float(f[entry + '/sample/positioner/yp'][()]))
                except:
                    continue
                positions.append(t)
        return positions

    def make_plot(self, img, positions, title, labels):
        shapes = itertools.cycle(
            ['*', '^', 'D', 'x', '8', 'p', '.', '4', 'd', '+', 'v', 'o', '1', '>', 's', 'X', '2', '<', '3'])
        fig = plt.figure(figsize=(8, 5))
        ax = plt.subplot(111)
        plt.title(title, fontsize=16)
        for p in positions:
            ax.plot(p[0], p[1], marker=next(shapes), linewidth=2, label=next(labels))
        ax.imshow(img['image'], extent=img['extent'])
        ax.set_xlabel("x(mm)")
        ax.set_ylabel("y(mm)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.05)
        path = "report/%s/" % self.account
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + title.replace(" ", "") + ".png", bbox_inches='tight', pad_inches=0)

    def make_header_tex(self):
        pubnumber = self.proposal + "- Cycle " + str(self.cycle)
        SESH = "https://sgmdata.lightsource.ca/users/xasexperiment/" + str(self.session)
        monthyear = datetime.datetime.now().strftime('%h %Y')
        reporttitle = self.proposal + " - " + self.principal
        numsamples = len(self.exp_info['samples'])
        numscans = np.sum([len(v) for k, v in self.sample_lists.items()])
        intro = " ".join([s.strip().replace('\'', '') for s in self.exp_info['bp']])[4:]
        edges = [", ".join(t) for t in self.exp_info['edges']]
        message = ""
        tex = """
\\newcommand{\\pubnumber}{%s}
\\newcommand{\\SESH}{%s}
\\newcommand{\\monthyear}{%s}
\\newcommand{\\reporttitle}{%s}
\\newcommand{\\numsamples}{%d }
\\newcommand{\\numscans}{%d }
\\newcommand{\\beamtime}{%d }
\\newcommand{\\projectsummary}{
\\normalsize Project \\reporttitle, initiated by %s

\\begin{center}
    \\begin{tabular}{cc}
        \\textbf{Total Samples:}  &  \\numsamples\\\\
        \\textbf{Total Measurements:}  &  \\numscans \\\\
        \\textbf{Shifts:} & \\beamtime  \\\\
    \\end{tabular}
\\end{center} \\\\

In total, there were %d holders measured,  each section that follows represents one of these holders,
all subsections correspond to the samples on these holders.  Representations and links to the data corresponding
to those samples will be contained therein. 
}
        """ % (pubnumber, SESH, monthyear, reporttitle, numsamples, numscans, self.shifts, intro, len(self.holders))
        path = "report/%s/" % self.account
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path + "header.tex", "w") as f:
            f.write(tex)

    def make_scan_subsection(self, key):
        """
            LaTeX creation of sample scan subsection.
        """
        ssec = ""
        for i, sample in enumerate(self.holders[key]):
            ssec += "\n\\FloatBarrier\\subsection{%s}\\label{ssec:%s_%d} \n" % (
            self.texcrub(sample), key.replace(" ", ""), i)
            edges = {}
            if key in self.scans_log.keys():
                edges.update({scan: sub for scan, sub in self.scans_log[key].items() if sample in scan})
            if key in self.eems_log.keys():
                if sample in self.eems_log[key].keys():
                    eems = self.eems_log[key][sample]
                    sample_tex = self.texcrub(sample)
                    ssec += "The excitation emission matrices for sample %s is displayed in Figure \\ref{fig:%s_%d}, " \
                            "this (un-normalized) data is acquired by sweeping the incident energy through " \
                            "the entire energy range of the beamline.  The signal from the 4 energy resolving " \
                            "silicon drift detectors are displayed in the left of the pane.  The right-most pane displays the accumulated " \
                            "XRF projection from this EEMs sweep. A more interactive view can be found by clicking " \
                            "\\href{%s}{here}. \n" % (
                                sample_tex,
                                sample.replace(" ", ""),
                                i,
                                eems['url']
                            )
                    ssec += "\\begin{figure} \n" \
                            "\\centering \n" \
                            "\\includegraphics[width=0.8\\linewidth]{%s.png} \n" \
                            "\\caption{Excitation Emission Matrix (EEMs) of sample %s (four left) " \
                            "taken with incident energy spanning 250 - 2000eV, included is the " \
                            "projected fluorescence signal (right). This data can be accessed at " \
                            "\\url{%s}. \\\\ \n} \n" \
                            "\\label{fig:%s_%d} \n" \
                            "\\end{figure} \\FloatBarrier\n" % (
                                eems['image'],
                                sample_tex,
                                eems['url'],
                                sample.replace(" ", ""),
                                i
                            )

            if len(edges):
                e_list = [e for e in self.holder_edges[key][i]]
                e_str = ",".join(e_list[:-1]) + " and " + e_list[-1]
                res_list = [str(info['res']) for edge, info in edges.items()]
                res_str = ",".join(res_list[:-1]) + " and " + res_list[-1]
                ssec += "The following %d subsections include measurement of %s for the edges: %s, and the figures " \
                        "contained therein show data that has been binned at a resolutions of %s, respectively. The " \
                        "averaged data is from multiple scans rastered across the sample surface, in order to limit " \
                        "radiative damage to the sample. The averaged spectral signals below are shown from each " \
                        "detector source unnormalized. For the energy resolved detectors (SDDs) a programmatic \'best guess\' " \
                        "region of interest (ROI) has been taken in order to display a 1-D trace, this representation " \
                        "is not final, i.e. the full sdd spectra are still available to select from online. \n" % (
                            len(edges),
                            self.texcrub(sample),
                            e_str,
                            res_str
                        )
            for edge, info in edges.items():
                ssec += "\\FloatBarrier\\subsubsection{%s}\\label{sssec:%s_%d}" % (
                self.texcrub(edge), edge.replace(" ", ""), i)
                ssec += "Figure \\ref{fig:%s_%d} contains the raw spectral signals for %s, taken " \
                        "from the average of %d scans (from the %d measured).  Any scans not included " \
                        "were removed for failing a \'scan health\' merit test based on detector " \
                        "saturation thresholds, overall signal strength and detection of discontinuities. \n" % (
                            edge.replace(" ", ""),
                            i,
                            self.texcrub(edge),
                            info['num_processed'],
                            info['num_raw'],
                        )
                ssec += "\\begin{figure} \n" \
                        "\\centering \n" \
                        "\\includegraphics[width=0.8\\linewidth]{%s.png}\n" \
                        "\\caption{X-ray absorption scan detector signals of sample %s (top)" \
                        "averaged from multiple scans (un-normalized), also included are the " \
                        "incoming intensity scalar signals (bottom). This data can be accessed in full at " \
                        "\\url{%s}. \\\\ \n} \n" \
                        "\\label{fig:%s_%d} \n" \
                        "\\end{figure} \\FloatBarrier\n" % (
                            info['image'],
                            self.texcrub(sample),
                            info['url'],
                            edge.replace(" ", ""),
                            i
                        )

        return ssec

    def texcrub(self, string):
        string = string.replace('\\', "$\\backslash$")
        string = string.replace("&", "\\&")
        string = string.replace("#", "\\#")
        string = string.replace("$", "\\$")
        string = string.replace("%", "\\%")
        string = string.replace("~", "\\~")
        string = string.replace("_", "\\_")
        string = string.replace("^", "\\^")
        string = string.replace("{", "\\{")
        string = string.replace("}", "\\}")
        return string

    def make_holder_table(self):
        """
            LaTeX creation of table for samples on holder
        """
        tex = ""
        holders = sorted([h for h in self.holders.keys()])
        for h in holders:
            table = "\\\\\n\t".join(
                ["%s & %s & %s & \\ref{ssec:%s_%d}" % (self.texcrub(self.holders[h][j]), ",".join(s),
                                                       self.texcrub(self.holder_notes[h][j]), h.replace(" ", ""), j) for
                 j, s in enumerate(self.holder_edges[h])])
            subsection = self.make_scan_subsection(h)
            notes = ".".join([n.strip().replace("\'", "").replace("\"", "") for n in self.holder_annote[h]])
            if not notes:
                notes = "None"
            tex += """ 
\\section{%s}
\\label{sec:%s}
There were %d samples on this holder (Figure \\ref{fig:%s}), upon which a total of %d measurements (edges) 
were performed.  \\
Any additional experimental information is listed as follows: 
\\textbf{%s}

\\begin{figure}
    \\centering
    \\includegraphics[width=0.8\\linewidth]{%s}
    \\caption{XRF Map of sample %s, with initial measurement position(s) annotated.  
    Legend lists sample names and edges where appropriate. }
    \\label{fig:%s}
\\end{figure}
\\begin{center}
   \\begin{tabular}{p{0.3\\textwidth}p{0.1\\textwidth}p{0.5\\textwidth}p{0.1\\textwidth}}\\hline
   Sample Name & Measured Edges & Notes & Section \\\\ \\hline
%s
\\end{tabular}
\\captionof{table}{Breakdown of samples loaded onto %s, along with the measured edges and link 
to the relevant subsection of the report.} 
\\end{center}

%s
\\clearpage
""" % \
                   (h, h.replace(" ", ""), len(self.holders[h]), h.replace(" ", ""), len(self.sample_lists[h]),
                    "\\\\" + notes, h.replace(" ", "") + ".png", h, h.replace(" ", ""), table, h, subsection)
        path = "report/%s/" % self.account
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path + "holders.tex", 'w') as f:
            f.write(tex)

    def make_eem_png(self, holder, title, eems):
        """
            Creates EEMs and XRF projection plots from fluorescence data in eems (dict)
        """
        fig = plt.figure(constrained_layout=True, figsize=(8, 5))
        gs = fig.add_gridspec(2, 3, width_ratios=[2, 2, 1])
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title('EEMs (sdd1)')
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title('EEMs (sdd2)')
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_title('EEMs (sdd3)')
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_title('EEMs (sdd4)')
        ax5 = fig.add_subplot(gs[:, -1])
        ax5.set_title('XRF Projection')

        ax5.plot(eems['xrf'], eems['emission'], linewidth=1, color='blue')
        ax5.set_xlabel("Fluoresence Intensity (a.u.)")
        ax5.set(ylabel='Emission Energy (eV)')
        ax1.imshow(eems['sdd1'],
                   extent=eems['extent'],
                   interpolation='nearest',
                   aspect=0.68,
                   )  #
        ax1.set_xlabel("Incident Energy(eV)")
        ax1.set_ylabel("Emission Energy(eV)")
        ax2.imshow(eems['sdd2'],
                   extent=eems['extent'],
                   interpolation='nearest',
                   aspect=0.68,
                   )  #
        ax2.set_xlabel("Incident Energy(eV)")
        ax2.set_ylabel("")
        ax2.set_yticks([])
        ax3.imshow(eems['sdd3'],
                   extent=eems['extent'],
                   interpolation='nearest',
                   aspect=0.68,
                   )  #
        ax3.set_xlabel("Incident Energy(eV)")
        ax3.set_ylabel("Emission Energy(eV)")
        ax4.imshow(eems['sdd4'],
                   extent=eems['extent'],
                   interpolation='nearest',
                   aspect=0.68,
                   )  #
        ax4.set_xlabel("Incident Energy(eV)")
        ax4.set_ylabel("")
        ax4.set_yticks([])

        path = "report/%s/%s/" % (self.account, holder.replace(" ", ""))
        if not os.path.exists(path):
            os.makedirs(path)
        path = path + title.replace(" ", "") + ".png"
        plt.savefig(path, pad_inches=0)
        fig.clear()
        plt.close(fig)
        return "/".join(path.split('/')[-2:])[:-4]

    def make_scan_png(self, holder, title, processed):
        """
            Creates plot of detector intensities from processed data (dict)
        """
        tPlot, axes = plt.subplots(
            nrows=2, ncols=1,
        )
        tPlot.suptitle(title + ": Detector Intensities", fontsize=16)
        ax = axes[0]
        ax2 = axes[1]
        ax2.plot(processed['en'], processed['pd'], linewidth=1, label="photodiode")
        ax2.plot(processed['en'], processed['clock'], linewidth=1, label="clock")
        ax2.plot(processed['en'], processed['i0'], linewidth=1, label="i0")
        ax2.legend(loc="lower right")
        ax.plot(processed['en'], processed['tey'], linewidth=1, label="tey", linestyle=':')
        ax.plot(processed['en'], processed['sdd1'], linewidth=1, label="sdd 1", marker='^', markersize=3, markevery=20,
                alpha=0.5)
        ax.plot(processed['en'], processed['sdd2'], linewidth=1, label="sdd 2", marker='D', markersize=3, markevery=20,
                alpha=0.5)
        ax.plot(processed['en'], processed['sdd3'], linewidth=1, label="sdd 3", )
        ax.plot(processed['en'], processed['sdd4'], linewidth=1, label="sdd 4", marker='s', markersize=3, markevery=20,
                alpha=0.5)
        ax.legend(loc="lower right")
        ax.set(xlabel=None)
        ax.set(xticklabels=[])
        ax2.set_xlabel("Energy (eV)")
        ax2.set_ylabel("Detector Signal (a.u.)")
        ax.set_ylabel("Detector Signal (a.u.)")
        path = "report/%s/%s/" % (self.account, holder.replace(" ", ""))
        if not os.path.exists(path):
            os.makedirs(path)
        path = path + title.replace(" ", "") + ".png"
        plt.savefig(path)
        tPlot.clear()
        plt.close(tPlot)
        return "/".join(path.split('/')[-2:])[:-4]

    def make_scan_figures(self, eems, processed, plots=True, key=None):
        """ Function to open EEMs and Averaged files and if they exist,
        call plotting routines.  Populates self.eems_log (dict), with structure:
                eems_log = {'Holder A - uuid' :
                                {"SampleName": {
                                                "image" : './path/to/plot/eems'
                                                "url": 'https://sgmdata/lightsource.ca/users/xasexperiment/userscan/#'
                                                }
                                ...}
                            ...}
        and populates self.scans_log (dict):
                scans_log = {'Holder A - uuid' :
                                {"SampleName": {
                                                "image" : './path/to/plot/scan'
                                                "url": 'https://sgmdata/lightsource.ca/users/xasexperiment/useravg/#'
                                                }
                                ...}
                            ...}
        """
        sgmdata_avg_url = "https://sgmdata.lightsource.ca/users/xasexperiment/useravg/"
        sgmdata_scan_url = "https://sgmdata.lightsource.ca/users/xasexperiment/userscan/"
        if key:
            if isinstance(key, str):
                holders = {key: self.holders[key]}
            elif isinstance(key, list):
                holders = {k: v for k, v in self.holders.items() if any(l in k for l in key)}
        else:
            holders = self.holders
        for k, v in holders.items():
            eems_list = {}
            for sample in v:
                try:
                    path = eems[k][sample].paths[-1]
                except Exception as e:
                    print("Trouble making EEMs plot for %s, error: %s" % (sample, e))
                    continue
                try:
                    eem_id = [i for v in eems[k][sample].scan_ids.values() for i in v.values()][-1]
                except Exception as e:
                    print("Can't find scan id for %s, error: %s" % (sample, e))
                    continue
                if not plots:
                    eems_list.update({sample:
                                          {"image": k.replace(" ", "") + "/" + sample.replace(" ", ""),
                                           "url": sgmdata_scan_url + str(eem_id)}})

                    continue
                with h5py.File(eems[k][sample].paths[-1], 'r') as h5:

                    NXentries = [str(x) for x in h5['/'].keys() if 'NXentry' in str(h5[x].attrs.get('NX_class'))]
                    try:
                        sdd1 = h5[NXentries[-1] + '/instrument/fluorescence/sdd1'][()]
                        sdd2 = h5[NXentries[-1] + '/instrument/fluorescence/sdd2'][()]
                        sdd3 = h5[NXentries[-1] + '/instrument/fluorescence/sdd3'][()]
                        sdd4 = h5[NXentries[-1] + '/instrument/fluorescence/sdd4'][()]
                    except KeyError:
                        print("Problem loading data from %s, skipping" % sample)
                        continue
                    eem_avg = np.mean([sdd1, sdd2, sdd3, sdd4], axis=0)
                    fluorescence = np.sum(eem_avg, axis=0)
                    fluorescence = fluorescence / np.max(fluorescence)
                    try:
                        emission = h5[NXentries[-1] + '/data/emission'][()]
                        en = h5[NXentries[-1] + '/data/en'][()]
                        extent = [min(en), max(en), min(emission), max(emission)]
                    except Exception as e:
                        print("Problem extracting extent for EEMs graph: %s, setting default." % e)
                        extent = [250, 2000, 10, 2560]

                    img = {'image': np.flip(eem_avg.T, axis=0),
                           'sdd1': np.flip(sdd1.T, axis=0),
                           'sdd2': np.flip(sdd2.T, axis=0),
                           'sdd3': np.flip(sdd3.T, axis=0),
                           'sdd4': np.flip(sdd4.T, axis=0),
                           'extent': extent,
                           'xrf': fluorescence,
                           'emission': emission}
                    path = self.make_eem_png(k, sample, img)
                eems_list.update({sample: {"image": path, "url": sgmdata_scan_url + str(eem_id)}})
            self.eems_log.update({k: eems_list})
            scans_list = {}
            for sample, v in processed[k].items():
                try:
                    d = v.avg_domain[0]
                except AttributeError:
                    continue
                avg_path = "/home/jovyan/data/" + d.split('.')[1] + "/" + d.split('.')[0] + '.nxs'
                if not plots:
                    scans_list.update({sample: {'image': k.replace(' ', '') + "/" + sample.replace(" ", ""),
                                                'url': sgmdata_avg_url + str(v.avg_id[0]),
                                                'num_processed': len(v.processed_ids),
                                                'num_raw': len(v.scan_ids),
                                                'res': 0.1
                                                }})
                    continue
                if os.path.exists(avg_path):
                    fpath = avg_path
                elif os.path.exists(avg_path.replace('/home/jovyan/', './')):
                    fpath = avg_path.replace('/home/jovyan/', './')
                else:
                    fpath = avg_path.replace('/home/jovyan/', './')
                    raise OSError(1, "No such file for sample average %s" % fpath)

                with h5py.File(fpath, 'r') as h5:
                    NXentries = [str(x) for x in h5['/'].keys() if 'NXentry' in str(h5[x].attrs.get('NX_class'))]
                    if not NXentries:
                        NXentries = ['entry1']
                    if NXentries[-1] + '/data/en' in h5:
                        en = h5[NXentries[-1] + '/data/en'][()]
                        sdd1 = h5[NXentries[-1] + '/data/sdd1'][()]
                        sdd2 = h5[NXentries[-1] + '/data/sdd2'][()]
                        sdd3 = h5[NXentries[-1] + '/data/sdd3'][()]
                        sdd4 = h5[NXentries[-1] + '/data/sdd4'][()]
                        tey = h5[NXentries[-1] + '/data/tey'][()]
                        i0 = h5[NXentries[-1] + '/data/i0'][()]
                        pd = h5[NXentries[-1] + '/data/pd'][()]
                        clk = h5[NXentries[-1] + '/data/clock'][()]
                    elif NXentries[-1] + '/data/sdd1_processsed' in h5:
                        try:
                            en = h5[NXentries[-1] + '/data/en_processed'][()]
                            sdd1 = h5[NXentries[-1] + '/data/sdd1_processsed'][()]
                            sdd2 = h5[NXentries[-1] + '/data/sdd2_processsed'][()]
                            sdd3 = h5[NXentries[-1] + '/data/sdd3_processsed'][()]
                            sdd4 = h5[NXentries[-1] + '/data/sdd4_processsed'][()]
                            tey = h5[NXentries[-1] + '/data/tey_processsed'][()]
                            i0 = h5[NXentries[-1] + '/data/i0_processsed'][()]
                            pd = h5[NXentries[-1] + '/data/pd_processsed'][()]
                            clk = h5[NXentries[-1] + '/data/clock_processsed'][()]
                        except KeyError as e:
                            print("Trouble loading data for %s, skipping. %s" % (sample, e))
                            continue
                    elif NXentries[-1] + '/data/sdd1_processed' in h5:
                        try:
                            en = h5[NXentries[-1] + '/data/en_processed'][()]
                            sdd1 = h5[NXentries[-1] + '/data/sdd1_processed'][()]
                            sdd2 = h5[NXentries[-1] + '/data/sdd2_processed'][()]
                            sdd3 = h5[NXentries[-1] + '/data/sdd3_processed'][()]
                            sdd4 = h5[NXentries[-1] + '/data/sdd4_processed'][()]
                            tey = h5[NXentries[-1] + '/data/tey_processed'][()]
                            i0 = h5[NXentries[-1] + '/data/i0_processed'][()]
                            pd = h5[NXentries[-1] + '/data/pd_processed'][()]
                            clk = h5[NXentries[-1] + '/data/clock_processed'][()]
                        except KeyError as e:
                            print("Trouble loading data for %s, skipping. %s" % (sample, e))
                            continue
                    else:
                        print("No data for %s, skipping. " % (sample))
                        continue
                    peak_guess = (en.min() + 0.33 * (en.max() - en.min())) / 10
                    ROI = (int(round(peak_guess - 5)), int(round(peak_guess + 5)))
                    sdd1 = np.sum(sdd1[:, ROI[0]:ROI[1]], axis=1)
                    sdd1 = sdd1 / np.max(sdd1)
                    sdd2 = np.sum(sdd2[:, ROI[0]:ROI[1]], axis=1)
                    sdd2 = sdd2 / np.max(sdd2)
                    sdd3 = np.sum(sdd3[:, ROI[0]:ROI[1]], axis=1)
                    sdd3 = sdd3 / np.max(sdd3)
                    sdd4 = np.sum(sdd4[:, ROI[0]:ROI[1]], axis=1)
                    sdd4 = sdd4 / np.max(sdd4)
                    tey = tey / np.max(tey)
                    pd = pd / np.max(pd)
                    clk = clk / np.max(clk)
                    i0 = i0 / np.max(i0)
                    data = {'en': en,
                            'sdd1': sdd1,
                            'sdd2': sdd2,
                            'sdd3': sdd3,
                            'sdd4': sdd4,
                            'tey': tey,
                            'i0': i0,
                            'pd': pd,
                            'clock': clk}
                    path = self.make_scan_png(k, sample, data)
                scans_list.update({sample: {'image': path,
                                            'url': sgmdata_avg_url + str(v.avg_id[0]),
                                            'num_processed': len(v.processed_ids),
                                            'num_raw': len(v.scan_ids),
                                            'res': np.abs(en[1] - en[0])
                                            }})
            self.scans_log.update({k: scans_list})

    def get_or_process_data(self, process=False, key=None, **kwargs):
        """
            User SGMQuery to find if EEMs and averaged (processed) files exist in SGMLive database.
            Optional Keyword:
                    process (type: bool) - Default=False. If True, and attempt is made to preprocess scans for which no
                                            averaged file currently exists.
        """
        holder_processed = dict()
        process_count = 0
        if key:
            if isinstance(key, str):
                holder_scans = {key: self.holder_scans[key]}
                holders = {key: self.holders[key]}
            elif isinstance(key, list):
                holder_scans = {k:v for k,v in self.holder_scans.items() if any(l in k for l in key)}
                holders = {k:v for k,v in self.holders.items() if any(l in k for l in key)}
        else:
            holder_scans = self.holder_scans
            holders = self.holders
        for k in holder_scans.keys():
            processed = dict()
            for sample in holder_scans[k]:
                init = self.holder_time_init[k]
                sgmq = SGMQuery(
                    sample=sample,
                    user=self.account,
                    processed=True,
                    daterange=(init, datetime.datetime.utcnow()),
                    data=False
                )
                sgmq.connection.close()
                if not sgmq.paths and process:
                    if process_count > 10:
                        self.client.shutdown()
                        self.client = Client()
                        process_count = 0
                    resolution = kwargs.get('resolution', 0.1)
                    sgmq = preprocess(sample,
                                      user=self.account,
                                      resolution=resolution,
                                      client=self.client,
                                      query=True,
                                      daterange=(init, datetime.datetime.utcnow()),
                                      **kwargs
                                      )
                    try:
                        sgmq.connection.close()
                    except AttributeError:
                        pass
                    process_count += 1
                processed.update({sample: sgmq})
            holder_processed.update({k: processed})
        holder_eems = dict()
        for k in holders.keys():
            eems = dict()
            for sample in holders[k]:
                init = self.holder_time_init[k]
                end = init + datetime.timedelta(hours=72)
                sgmq2 = SGMQuery(
                    sample=sample,
                    user=self.account,
                    daterange=(init, end),
                    data=False
                )
                sgmq2.connection.close()
                eems.update({sample: sgmq2})
            holder_eems.update({k: eems})
        return holder_eems, holder_processed

    def setup_tex(self):
        main = './main.tex'
        path = "report/%s/" % self.account
        if not os.path.exists(path):
            os.makedirs(path)
        path = "report/%s/main.tex" % self.account
        copyfile(main, path)
        self.make_header_tex()

    def create_sample_report(self, key=None, plots=True, process=False, **kwargs):
        """
            Core logic to create LaTeX report from confluence experimental log.
        """
        self.setup_tex()
        if key:
            if isinstance(key, str):
                holders = {key: self.holders[key]}
            elif isinstance(key, list):
                holders = {k:v for k,v in self.holders.items() if any(l in k for l in key)}
        else:
            holders = self.holders
        self.holder_time_init = {}
        for k, v in holders.items():
            print("Creating report for %s" % k)
            sgmq = SGMQuery(sample=k, user=self.account, data=False)
            paths = [sgmq.paths[0]]
            init = datetime.datetime.strptime(paths[0].split('/')[-1].split('.')[0], "%Y-%m-%dt%H-%M-%S%z")
            self.holder_time_init.update({k: init})
            paths2 = list()
            sample_list = []
            for j, s in enumerate(v):
                q = [(p, datetime.datetime.strptime(p.split('/')[-1].split('.')[0], "%Y-%m-%dt%H-%M-%S%z") - init)
                     for p in SGMQuery(sample=s, user=self.account, data = False).paths]
                q = sorted(q, key=lambda i: i[1])
                q = [p for p in q if p[1].total_seconds() > 0]
                if q:
                    paths2.append(q[0][0])
                    sample_list.append(s)
                for scanname in [edge for edge in self.holder_scans[k] if s in edge]:
                    q = [(p, datetime.datetime.strptime(p.split('/')[-1].split('.')[0], "%Y-%m-%dt%H-%M-%S%z") - init,
                          scanname)
                         for p in SGMQuery(sample=scanname, user=self.account, data=False).paths]
                    q = sorted(q, key=lambda i: i[1])
                    q = [p for p in q if p[1].total_seconds() > 0]
                    if q:
                        paths2.append(q[0][0])
                        sample_list.append(q[0][2])
            paths += paths2
            if plots:
                data = SGMData(paths)
                try:
                    entries = [k2 for k1, scan in data.scans.items()
                               for k2, entry in scan.__dict__.items() if entry['sample'] != k]
                    positions = self.get_sample_positions(paths2, entries)
                except Exception as e:
                    print("Couldn't get sample positions: %s" % e)
                    positions = []

                image = [entry for k1, scan in data.scans.items() for k2, entry in scan.__dict__.items() if
                         entry['sample'] == k][0]
                command = image['command']
                xrange = (float(command[2]), float(command[3]))
                yrange = (float(command[6]), float(command[7]))
                dx = abs(xrange[0] - xrange[1])/(int(command[4])* 20)
                dy = abs(yrange[0] - yrange[1])/50
                df = image.interpolate(resolution=[dx, dy], start=[min(xrange),min(yrange)], stop=[max(xrange), max(yrange)])
                img_data = self.make_data(df)
                self.make_plot(img_data, positions, k, iter(sample_list))
                del img_data, data, image, df
            self.sample_lists.update({k: sample_list})
        self.make_scan_figures(*self.get_or_process_data(process=process, key=key, **kwargs), plots=plots)
        self.make_holder_table()

