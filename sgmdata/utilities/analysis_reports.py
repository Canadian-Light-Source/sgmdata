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
    pcalc, pcov = fit_xrf(emission, y, constrain_peaks(guess, 2, 2 * np.amax(y), 5)) #Some sdds have incorrect bin number.
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
    roi_cols = avg.data.filter(regex="sdd[1-4]_[0-2].*").columns
    avg.data.drop(columns=roi_cols, inplace=True)
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
    return fit, emission


def norm_arr(a, max):
    if abs(max) > 0:
        a = max * (a / np.amax(a))
        return a
    a = a /np.amax(a)
    return a


def make_eemsreport(data, emission=[], sample = None, i0=1, bs_args={}):
    interp = []
    report = []
    bs_args.update({'report': [k for k in data.scans.keys()]})
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
        fit, emission = sel_roi(avg, emission=emission)
        xrf_plot = [{
            "title": "XRF Plot",
            "kind": "lineplot",
            "data": {
                "x": ["Emission Energy (eV)"] + list(emission),
                "y1": [["sdd1"] + list(np.nansum(avg.get_arr("sdd1"), axis=0)),
                       ["sdd2"] + list(np.nansum(avg.get_arr("sdd2"), axis=0)),
                       ["sdd3"] + list(np.nansum(avg.get_arr("sdd3"), axis=0)),
                       ["sdd4"] + list(np.nansum(avg.get_arr("sdd4"), axis=0))
                       ],
                "x-label": "Emission Energy (eV)",
                "y1-label": "Fluorescence",
                "aspect-ratio": 1.5,
                "annotations": [{"value": p, "text": f"ROI{i}"} for i, p in enumerate(fit['peaks'])]
            },
            "style": "col-12"
        }]
        tey = np.nan_to_num(avg.get_arr("tey"))
        m_tey = max(tey)
        pd = norm_arr(np.nan_to_num(avg.get_arr("pd")), m_tey)
        sdd1 = norm_arr(avg.get_arr("sdd1"), m_tey)
        sdd2 = norm_arr(avg.get_arr("sdd2"), m_tey)
        sdd3 = norm_arr(avg.get_arr("sdd3"), m_tey)
        sdd4 = norm_arr(avg.get_arr("sdd4"), m_tey)
        xas_plots = [{
            "title": f"XAS Plot for ROI{i}",
            "kind": "lineplot",
            "data": {
                "x": ["Energy (eV)"] + list(avg.data.index),
                "y1": [["pd"] + list(np.nansum(pd, axis=1)),
                       ["tey"] + list(np.nansum(tey, axis=1)),
                       ["sdd1"] + list(np.nansum(sdd1[:, int(p/10 - fit['widths'][i]/10):int(p/10 + fit['widths'][i]/10)], axis=1)/i0),
                       ["sdd2"] + list(np.nansum(sdd2[:, int(p/10 - fit['widths'][i]/10):int(p/10 + fit['widths'][i]/10)], axis=1)/i0),
                       ["sdd3"] + list(np.nansum(sdd3[:, int(p/10 - fit['widths'][i]/10):int(p/10 + fit['widths'][i]/10)], axis=1)/i0),
                       ["sdd4"] + list(np.nansum(sdd4[:, int(p/10 - fit['widths'][i]/10):int(p/10 + fit['widths'][i]/10)], axis=1)/i0)
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

    return report

def sel_map_roi(entry, emission=[], max_en=2000):
    """
    ### Description:
    Finds XRF peaks of high variance throughout the image, and return thier ROIs.
    ### Args:
    >**avg** *(sgmdata.Processed)* -- Averaged object from running sgmdata.mean().
    ### Keywords:
    >**emission** *(ndarray)*  -- labels for xrf bins
    >**depth** *(int)* -- number of points to take from the beginning and end of the XANES scan.
    >**bounds** *(list)* -- list of len 2, included start and stop bin of mcas to be fit.
    """
    detectors = ['sdd1', 'sdd2', 'sdd3', 'sdd4']
    roi_cols = entry['binned']['dataframe'].filter(regex="sdd[1-4]_[0-2].*").columns
    df = entry['binned']['dataframe']
    df.drop(columns=roi_cols, inplace=True)
    if not len(emission):
        sig = df.filter(regex="sdd1.*").to_numpy()
        emission = np.linspace(10, sig.shape[1] * 10, sig.shape[1])
    fit = {"peaks": [], "heights": [], "widths": []}
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
    pk, h, _ = high_var(entry, emission=emission, detector="sdd3")
    close_peaks = [p for p in pk for sp in peaks if abs(p - sp) < 12]
    pks = [p for p in pk if p < max_en and p not in close_peaks]
    h = [hgt for i, hgt in enumerate(h) if pk[i] < max_en and pk[i] not in close_peaks]
    w = [50 for i in range(len(pks))]
    fit = {"peaks": peaks + pks, "heights": heights + h, "widths": widths + w}
    return fit, emission

def make_xrfmapreport(data, emission=[], sample = None, i0=1):
    interp = []
    report = []
    for f, scan in data.scans.items():
        for entry in list(scan.__dict__.values()):
            if 'binned' in entry.keys():
                interp.append(entry)
            else:
                df = entry.interpolate()
                interp.append(entry)
    for entry in interp:
        data = entry['binned']['dataframe']
        fit, emission = sel_map_roi(entry, emission=emission)
        xrf_plot = [{
            "title": "XRF Plot",
            "kind": "lineplot",
            "data": {
                "x": ["Emission Energy (eV)"] + list(emission),
                "y1": [["sdd1"] + list(np.nansum(entry.get_arr("sdd1"), axis=0)),
                       ["sdd2"] + list(np.nansum(entry.get_arr("sdd2"), axis=0)),
                       ["sdd3"] + list(np.nansum(entry.get_arr("sdd3"), axis=0)),
                       ["sdd4"] + list(np.nansum(entry.get_arr("sdd4"), axis=0))
                       ],
                "x-label": "Emission Energy (eV)",
                "y1-label": "Fluorescence",
                "aspect-ratio": 1.5,
                "annotations": [{"value": p, "text": f"ROI{i}"} for i, p in enumerate(fit['peaks'])]
            },
            "style": "col-12"
        }]
        tey = norm_arr(np.nan_to_num(entry.get_arr("tey")), i0)
        pd = norm_arr(np.nan_to_num(entry.get_arr("pd")), i0)
        sdd1 = norm_arr(entry.get_arr("sdd1"), i0)
        sdd2 = norm_arr(entry.get_arr("sdd2"), i0)
        sdd3 = norm_arr(entry.get_arr("sdd3"), i0)
        sdd4 = norm_arr(entry.get_arr("sdd4"), i0)
        xrfm_plots = [{
            "title": f"XRF Map for ROI{i}",
            "kind": "heatmap",
            "data": {
                "x": ["xp (mm)"] + list(data.index.get_level_values('xp')),
                "y": ["yp (mm)"] + list(data.index.get_level_values('yp')),

                "z": [["tey"] + list(np.nansum(tey, axis=1)),
                       ["sdd1"] + list(np.nansum(sdd1[:, int(p/10 - fit['widths'][i]/10):int(p/10 + fit['widths'][i]/10)], axis=1)),
                       ["sdd2"] + list(np.nansum(sdd2[:, int(p/10 - fit['widths'][i]/10):int(p/10 + fit['widths'][i]/10)], axis=1)),
                       ["sdd3"] + list(np.nansum(sdd3[:, int(p/10 - fit['widths'][i]/10):int(p/10 + fit['widths'][i]/10)], axis=1)),
                       ["sdd4"] + list(np.nansum(sdd4[:, int(p/10 - fit['widths'][i]/10):int(p/10 + fit['widths'][i]/10)], axis=1))
                       ],
                "selected": ["sdd1", "sdd2", "sdd3", "sdd4"],
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
                "title": "X-ray Fluorescence Maps",
                "style": "row",
                "content": xrfm_plots
            },
        ]

    return report