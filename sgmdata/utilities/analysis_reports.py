from sgmdata.xrffit import *
from sgmdata.utilities.util import badscans
import numpy as np
import json

def fit_peaks_once(emission, sdd):
    """
    ### Description:
    Method for summing, normalizing and then finding relavent peaks.
    ### Args:
    >**emission** *(ndarray)*  -- labels for xrf bins
    >**sdd** *(list)* -- list of sdd detector signals filtered from dataframe.
    ### Keywords:
    >**bounds** *(list)* -- list of len 2, included start and stop bin of mcas to be fit.
    """
    if not isinstance(sdd, list):
        sdd = [sdd]
    names = [list(s)[0].split('-')[0] for s in sdd]
    y = sdd[0].sum(axis=0).to_numpy() / len(sdd[0])
    for i, n in enumerate(names[1:]):
        y = y + sdd[i + 1].sum(axis=0).to_numpy() / len(sdd[i + 1])
    pks, hgts = find_peaks(y, distance=10, height=3)
    guess = []
    for i in range(0, len(pks)):
        guess += [pks[i] * 10, hgts['peak_heights'][i], 50]
    pcalc, pcov = fit_xrf(emission, y, constrain_peaks(guess, 2, 2 * np.amax(y), 5))
    wid = pcalc[2::3]
    ctr = pcalc[0::3]
    return pks, hgts, wid, ctr


def high_var(avg, emission=[], detector='sdd3'):
    """
    ### Description:
    Find high variance peaks in detector placed at low scattering position.
    ### Returns:
    (peaks (list), heights (list), widths (list))
    """
    sig = avg.get_arr(detector)
    if not len(emission):
        emission = np.linspace(10, sig.shape[1] * 10, sig.shape[1])
    xrf_var = np.var(sig, axis=0)
    var_peaks, _ = find_peaks(xrf_var, distance=10, height=np.mean(xrf_var))
    peaks = [emission[v] for v in var_peaks]
    heights = [max(sig[:, var]) for var in var_peaks]
    widths = [50 for i in range(len(var_peaks))]
    return peaks, heights, widths


def sel_roi(avg, emission=[], depth=20):
    """
    ### Description:
    Finds XRF peaks at beginning and end of the XANES spectrum, then takes the difference set of final peaks from the initial set.
    Additionally find XRF regions of high variance throughout the scan range, and append these peaks to the above set.
    ### Args:
    >**avg** *(sgmdata.Processed)* -- Averaged object from running sgmdata.mean().
    ### Keywords:
    >**emission** *(ndarray)*  -- labels for xrf bins
    >**depth** *(int)* -- number of points to take from the beginning and end of the XANES scan.
    >**bounds** *(list)* -- list of len 2, included start and stop bin of mcas to be fit.
    """
    detectors = ['sdd1', 'sdd2', 'sdd3', 'sdd4']
    if not len(emission):
        sig = avg.get_arr("sdd1")
        emission = np.linspace(10, sig.shape[1] * 10, sig.shape[1])
    start_end = [depth, -1 * depth]
    fit = {"peaks": [], "heights": [], "widths": []}
    for selection in start_end:
        df = avg.data
        if selection > 0:
            df = df.head(selection).copy()
        if selection < 0:
            df = df.tail(-1 * selection).copy()
        max_en = max(df.index) * 1.025  # 2.5% error in mca energy range
        roi_cols = df.filter(regex="sdd[1-4]_[0-2].*").columns
        df.drop(columns=roi_cols, inplace=True)
        data = []
        for det in detectors:
            rgx = "%s.*" % det
            data.append(df.filter(regex=rgx, axis=1))
        pks, hgts, wid, _ = fit_peaks_once(emission, data)
        peaks = [emission[p] for p in pks if 230 < emission[p] < max_en]
        heights = [h for i, h in enumerate(hgts['peak_heights']) if 230 < emission[pks[i]] < max_en]
        widths = [w for i, w in enumerate(wid) if 230 < emission[pks[i]] < max_en]
        fit["peaks"].append(peaks)
        fit["heights"].append(heights)
        fit["widths"].append(widths)

    peaks = list(set(fit['peaks'][-1]).difference(set(fit['peaks'][0])))
    heights = [h for i, h in enumerate(fit['heights'][-1]) if fit['peaks'][-1][i] in peaks]
    widths = [w for i, w in enumerate(fit['widths'][-1]) if fit['peaks'][-1][i] in peaks]
    pk, h, _ = high_var(avg, emission=emission, detector="sdd3")
    close_peaks = [p for p in pk for sp in peaks if abs(p - sp) < 12]
    pks = [p for p in pk if p < max_en and p not in close_peaks]
    h = [hgt for i, hgt in enumerate(h) if pk[i] < max_en and pk[i] not in close_peaks]
    w = [50 for i in range(len(pks))]
    fit = {"peaks": peaks + pks, "heights": heights + h, "widths": widths + w}
    return fit


def norm_arr(a, max):
    a = max * (a / np.amax(a))
    return a


def make_eemsreport(data, emission=None, sample = None, i0=1, bs_args={"report": True}):
    interp = []
    report = []
    bscan_report = {}
    for f, scan in data.scans.items():
        for entry in list(scan.__dict__.values()):
            if 'binned' in entry.keys():
                interp.append(entry['binned']['dataframe'])
    if interp:
        _, bscan_report = badscans(interp, **bs_args)
    if sample:
        averaged = {sample: data.averaged[sample]}
    else:
        averaged = data.averaged
    for sample, avg in averaged.items():
        if not emission:
            sig = avg.get_arr("sdd1")
            emission = np.linspace(10, sig.shape[1] * 10, sig.shape[1]).tolist()
        fit = sel_roi(avg, emission=emission)
        xrf_plot = [{
            "title": "XRF Plot",
            "kind": "lineplot",
            "data": {
                "x": ["Emission Energy (eV)"] + list(emission),
                "y1": [["sdd1"] + list(avg.get_arr("sdd1").sum(axis=0)),
                       ["sdd2"] + list(avg.get_arr("sdd2").sum(axis=0)),
                       ["sdd3"] + list(avg.get_arr("sdd3").sum(axis=0)),
                       ["sdd4"] + list(avg.get_arr("sdd4").sum(axis=0))
                       ],
                "x-label": "Emission Energy (eV)",
                "y1-label": "Fluorescence",
                "aspect-ratio": 1.5,
                "annotations": [{"value": p, "text": f"ROI{i}"} for i, p in enumerate(fit['peaks'])]
            },
            "style": "col-12"
        }]
        tey = avg.get_arr("tey")
        m_tey = max(tey)
        pd = norm_arr(avg.get_arr("pd"), m_tey)
        sdd1 = norm_arr(avg.get_arr("sdd1"), m_tey)
        sdd2 = norm_arr(avg.get_arr("sdd2"), m_tey)
        sdd3 = norm_arr(avg.get_arr("sdd3"), m_tey)
        sdd4 = norm_arr(avg.get_arr("sdd4"), m_tey)
        xas_plots = [{
            "title": f"XAS Plot for ROI{i}",
            "kind": "lineplot",
            "data": {
                "x": ["Energy (eV)"] + list(avg.data.index),
                "y1": [["pd"] + list(pd.sum(axis=1)),
                       ["tey"] + list(tey.sum(axis=1)),
                       ["sdd1"] + list(sdd1[:, int(p/10 - fit['widths'][i]/10):int(p/10 + fit['widths'][i]/10)].sum(axis=1)/i0),
                       ["sdd2"] + list(sdd2[:, int(p/10 - fit['widths'][i]/10):int(p/10 + fit['widths'][i]/10)].sum(axis=1)/i0),
                       ["sdd3"] + list(sdd3[:, int(p/10 - fit['widths'][i]/10):int(p/10 + fit['widths'][i]/10)].sum(axis=1)/i0),
                       ["sdd4"] + list(sdd4[:, int(p/10 - fit['widths'][i]/10):int(p/10 + fit['widths'][i]/10)].sum(axis=1)/i0)
                       ],
                "x-label": "Energy (eV)",
                "y1-label": "Absorption (a.u.)",
                "aspect-ratio": 1.5,
            },
            "style": "col-12"
        } for i, p in enumerate(fit['peaks'])]

        report += [{
                "title": "Fluorescence Region of Interest Selection",
                "style": "row",
                "content": xrf_plot
            },
            {
                "title": "X-ray Absorption Spectra",
                "style": "row",
                "content": xas_plots
            },
            bscan_report
        ]

    return json.dumps(report)
