from setuptools import setup

def readme():
    with open('README.md') as f:
        README = f.read()
    return README

setup(
    name="sgm-data",
    version="0.1",
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
    packages=["sgmdata"],
    include_package_data=True,
    install_requires=[
        "dask>=2.11.0",
        "pandas>=0.25.3"
        "h5py>=2.10.0",
        "h5pyd",
        "numpy",

    ]


)