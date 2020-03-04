import argparse
import asyncio
from glob import glob
from load import SGMData

def main():
    parser = argparse.ArgumentParser(description="SGM Data Analysis Tools")
    parser.add_argument("-i", "--interp", type=str, nargs=1, metavar="interpolation", help="Interpolate on load")
    parser.add_argument("-a", "--avg", type=str, nargs=0, metavar="average", help="Average selected files")
    parser.add_argument("-o", "--out", type=str, nargs =1, metavar="output", help="Output file (HDF5)")


    args = parser.parse_args()