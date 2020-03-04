# Helper module for working with data from the [SGM Beamline](https://sgm.lightsource.ca) 
### Installation:
Using pip:
```commandline
pip install sgmdata
```
or from source: 
```commandline
git clone https://github.lightsource.ca/arthurz/sgmdata ./sgmdata
cd sgmdata
python setup.py install
```
### Usage:
First import the package, and select data to load in.
```python
import sgmdata 
data = sgmdata.SGMData(["file1.hdf5", "file2.hdf5", "..."])
```
This will identify the independent axis, signals and other data within the files listed. 
The file load list also works with [hsds](https://github.com/HDFGroup/hsds) domains.
```python
data = sgmdata.SGMData(["file1.data.sgm-hdf5.lightsource.ca", "..."])
```
Useful functions:
```python
data.scans  #contains a dictionary of the identified data arrays loaded from your file list
data.interpolate(start=270, stop=2000, resolution=0.5) #bin the data in scans dictionary and interpolates missing points
data.average(scans=[("file1.hdf5", "entry1"), ("file2.hdf5", "entry3")]) #TODO:  average interpolated data in scans dictionary. 
```

