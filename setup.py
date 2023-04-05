from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        README = f.read()
    return README
exec(open('sgmdata/version.py').read())
setup(
    name="sgm-data",
    version=__version__,
    description="Module for loading, interpolating and plotting data taken at the SGM Beamline at the Canadian Light Source.",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.lightsource.ca/arthurz/sgmdata",
    author="Zachary Arthur",
    author_email="zachary.arthur@lightsource.ca",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7"
    ],
    packages=find_packages(),
    include_package_data=True,
    package_data={"sgmdata": [
        "plots/callbacks/eemscan/*.js",
        "plots/callbacks/xrfmap/*.js",
    ]},
    install_requires=[
        "blosc==1.10.2",
        "lz4==3.1.3",
        "dask[complete]>=2021.03.0",
        "msgpack==1.0.2",
        "pandas>=1.1.5",
        "h5py>=2.10.0",
        "bokeh>=1.4.0",
        "numpy>=1.18.3",
        "scipy>=1.4.1",
        "tqdm",
        "python-slugify",
        "beautifulsoup4",
        "matplotlib",
        "requests",
        "lmfit"
    ]


)
