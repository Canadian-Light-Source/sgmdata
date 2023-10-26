
# API
## SGMData
-----

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

## SGMScan
-----

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
data.scans['file_prefix'].entry1.interpolate()
sdd1 = data.scans['file_prefix'].entry1.get_arr('sdd1')
sdd1.shape # (1290, 256)
```

## SGMQuery
-----

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

## preprocess()
-----

### Description:
>Utility for automating the interpolation and averaging of a sample in the SGMLive website.

### Args:
>**sample** *(str)*:  The name of the sample in your account that you wish to preprocess.
or:
>**data_id** *(int)*: The primary key of the dataset to preprocress.

### Keywords:
>All of the below are optional.
>**proposal** *(str)* -- name of proposal to limit search to.

>**user** *(str)* -- name of user account to limit search to (for use by staff).

>**resolution** *(float)* -- to be passed to interpolation function, this is histogram bin width.

>**start** *(float)* --  start energy to be passed to interpolation function.

>**stop** *(float)* -- stop energy to be passed to interpolation function.

>**sdd_max** *(int)* -- threshold value to determine saturation in SDDs, to determine scan_health (default
is 105000).
>**bscan_thresh** *(tuple)* -- (continuous, dumped, and saturated)  these are the threshold percentages from
scan_health that will label a scan as 'bad'.
>**report** *(str)* -- Analysis report type, e.g. "XAS Report".
>**report_id** *(int)* -- primary key of report to be updated.

### Returns:
>SGMQuery object if query_return is True

### Example Usage:
```python
from sgmdata import preprocess

preprocess(sample="TiO2", user='regiert', resolution=0.1)
```

## interpolate()
-----

### Description:
>Creates the bins required for each independent axes to be histogrammed into for interpolation,
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

## fit_peaks()
_____

### Description:
Method for fitting multiple interpolated SDD numpy arrays with a sum of gaussians.

### Args:
>**emission** *(ndarray)*  -- labels for xrf bins

>**sdd** *(list)* -- list of sdd detector signals filtered from dataframe.

### Keywords:
>**bounds** *(list)* -- list of len 2, included start and stop bin of mcas to be fit.


