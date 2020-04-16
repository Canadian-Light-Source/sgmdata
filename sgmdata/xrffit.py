import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from functools import partial
from multiprocessing.pool import Pool, ThreadPool

try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
        from tqdm.notebook import tqdm  # Jupyter notebook or qtconsole
    else:
        from tqdm import tqdm  # Other type (?)
except NameError:
    from tqdm import tqdm


def gaussians(x, *params, width=[], centre=[]):
    y = np.zeros_like(x)
    if len(width) and len(centre):
        for i in range(0, len(params)):
            ctr = centre[i]
            amp = params[i]
            wid = width[i]
            y = y + amp * np.exp(-((x - ctr) / wid) ** 2)
    else:
        for i in range(0, len(params), 3):
            ctr = params[i]
            amp = params[i + 1]
            wid = params[i + 2]
            y = y + amp * np.exp(-((x - ctr) / wid) ** 2)
    return y

def constrain_peaks(l_guess, en_width, h_width, w_width, amp_only=False):
    i = 0
    guess = []
    l_bounds = []
    u_bounds = []
    if not amp_only:
        for i in range(0, len(l_guess), 3):
            guess.append(l_guess[i])
            l_bounds.append(l_guess[i] - en_width)
            u_bounds.append(l_guess[i] + en_width)
            guess.append(l_guess[i + 1])
            l_bounds.append(0)
            u_bounds.append(l_guess[i + 1] + h_width)
            guess.append(l_guess[i + 2])
            l_bounds.append(l_guess[i + 2] - w_width)
            u_bounds.append(l_guess[i + 2] + w_width)
        return guess, (l_bounds, u_bounds)

    else:
        for i in range(0, len(l_guess), 3):
            guess.append(l_guess[i + 1])
            l_bounds.append(0)
            u_bounds.append(l_guess[i + 1] + h_width)

        return guess, (l_bounds, u_bounds)


def fit_xrf(emission, xrf, bounds):
    fit = (curve_fit(gaussians, emission, xrf, p0=bounds[0], bounds=bounds[1]))
    return fit


def _fit_amp(emission, xrf, bounds, wid, ctr):
    g = partial(gaussians, width=wid, centre=ctr)
    fit = (curve_fit(g, emission, xrf, p0=bounds[0], bounds=bounds[1]))
    return fit


def fit_amp(args):
    return _fit_amp(*args)


def fit_peaks(emission, sdd, first=True):
    y = sdd.sum(axis=0).to_numpy()
    y = y / len(sdd)
    pks, hgts = find_peaks(y, distance=10, height=3)
    new = np.zeros((len(sdd), len(pks)))
    guess = []
    for i in range(0, len(pks)):
        guess += [pks[i] * 10, hgts['peak_heights'][i], 50]
    pcalc, pcov = fit_xrf(emission, y, constrain_peaks(guess, 2, np.amax(y), 5))
    wid = pcalc[2::3]
    amp = pcalc[1::3]
    ctr = pcalc[0::3]
    xrfs = [[emission, sdd.to_numpy()[i], constrain_peaks(pcalc, 2, np.amax(y), 5, amp_only=True), wid, ctr] for i in
            range(len(sdd))]
    print("Fitting peak amplitudes")
    with Pool(8) as pool:
        L = list(tqdm(pool.imap_unordered(fit_amp, xrfs), total=len(xrfs)))

    return L, xrfs


