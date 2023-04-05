from sgmdata import *
from sgmdata.load import SGMData, SGMScan
from sgmdata.search import SGMQuery
from sgmdata.utilities.util import badscans, preprocess
from sgmdata.interpolate import interpolate
from sgmdata.xrffit import fit_peaks
import sgmdata.sign
from .version import __version__


__doc__ = f"""
# API
## SGMData
-----
{SGMData.__doc__}
## SGMScan
-----
{SGMScan.__doc__}
## SGMQuery
-----
{SGMQuery.__doc__}
## preprocess()
-----
{preprocess.__doc__}
## interpolate()
-----
{interpolate.__doc__}
## fit_peaks()
_____
{fit_peaks.__doc__}
"""
