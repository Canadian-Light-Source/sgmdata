
# Utilities
## h5tree
-----

### Description:
>A function to output the data-tree from an hdf5 file object.

### Args:
>**h5** *(h5py.File)* -- Any H5File object, from h5py.

### Returns:
>**None**

### Example Usage:
```python
from sgmdata.utilities import h5tree
import h5py

f = h5py.File("Filename.nxs", 'r')
h5tree(f)
```

## plot1d
-----

### Description:
>Convenience function for plotting a bokeh lineplot, assumes Bokeh is already loaded.

### Args:
>**xarr** *(array-like)* --  Independent array-like object, or list of array-like objects.

>**yarr** *(array-like)* -- Dependent array-like object, or list of array-like objects, same shape as xarr

>**title** *(str)* -- Plot title

>**labels** *(list(str))* --  Legend labels for multiple objects, defaults to Curve0, Curve1, etc.

### Returns:
>**None**

## preprocess
-----

### Description:
>Utility for automating the interpolation and averaging of a sample in the SGMLive website.

### Args:
>**sample** *(str)*:  The name of the sample in your account that you wish to preprocess.

### Keywords:
>All of the below are optional.

>**user** *(str)* -- name of user account to limit search to (for use by staff).

>**resolution** *(float)* -- to be passed to interpolation function, this is histogram bin width.

>**start** *(float)* --  start energy to be passed to interpolation function.

>**stop** *(float)* -- stop energy to be passed to interpolation function.

>**sdd_max** *(int)* -- threshold value to determine saturation in SDDs, to determine scan_health (default
is 105000).
>**bscan_thresh** *(tuple)* -- (continuous, dumped, and saturated)  these are the threshold percentages from
scan_health that will label a scan as 'bad'.

### Returns:
>(HTML) hyperlink for preprocessed data stored in SGMLive

### Example Usage:
```python
from sgmdata import preprocess

preprocess(sample="TiO2", user='regiert', resolution=0.1)
```

## badscans
-----

### Description:
>Batch calculation of sgmdata.utilities.scan_health for list of interpolated dataframes.

### Args:
>interp (list) :  list of SGMScan binned dataframes.

### Returns:
>List of indexes for bad scans in interp.


## scan_health
-----

### Description:
>Function takes in a interpolated scan (a pandas DataFrame), and returns the overall health.

### Args:
>**df** *(DataFrame)* --  pandas dataframe from SGMScan.binned.

>**verbose** *(bool)* -- Explain the returned output in plain text.

>**sdd_max** *(int)* -- 105000 (default) - saturation value for total SDD counts/s

### Returns:
>(tuple):  (Discontiunty %,  Beam-dump %,  Saturation %)

## predict_num_scans
-----

### Description:
-----
Takes the SGMData object of a sample and uses a combination of other functions to predict how many additional
scans should be taken of that sample.
### Args:
-----
> **data** *(type: SGMData object)* -- The SGMData object for the sample on which the user would like more
information.
> **verbose** *(type: optional boolean)* -- Default value is False. If set to True, gives user additional data
on how the additional number of scans needed was calculated.
> **percent_of_log** *(type: optional float)* -- Default value is 0.4. The average of the noise values of the
first ten scans is taken, and the log of it is found. Scans continue to be taken, and the average of the
noise values of the most recent ten scans is taken. The log of this average is taken,and if it's less than
percent_of_log multiplied by the log of the first ten averages, then scanning stops.
> **num_scans** *(type: optional int)* -- Default value is 10. The number of scans from the scans provided by
the user, that the user would like to be used to predict the number of additional scans to take.
### Returns:
-----
>*(int)*: The predicted number of additional scans that should be taken of a sample.

## create_csv
-----

### Description:
>Make CSV file from sample(s)

### Args:
>**sample** *(str or list(str))*  -- Sample(s) name(s) from SGMLive that you want to process.

### Keywords:
>**mcas** *(list(str))* -- list of detector names for which the ROI summation should take place.

>**user** *(str)* -- SGMLive account name, defaults to current jupyterhub user.

>**out** *(os.path / str)* -- System path to output directory for csv file(s)

>**I0** *(pandas.DataFrame)** -- Dataframe including an incoming flux profile to be joined to the sample
dataframe and included in the each CSV file.

>**ROI** *(tuple)** --  Set the upper and lower bin number for the Region-of-Interest integration to be used in
reducing the dimensionality of energy MCA data.

### Returns:
>**list(pd.DataFrame)** -- list of dataframes created.

## OneList
-----

### Description:
>List extension that will return the sole item of the list if len(list) == 1

### Usage:
```python
data = {"key":1}
l = OneList([data])
assert l == data
print(l['key'])  #prints 1
l.append(2)
print(l[1]) #prints 2
assert l == data #raises Error
```

## DisplayDict
-----

### Description
>dict class extension that includes repr_html for key,value display in Jupyter.


