import os
import h5py
import h5pyd
import numpy as np
import dask.dataframe as dd
import pandas as pd


class sgmdata(object):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if not hasattr(self, "data"):
            self.data = {}


    def plot(self, **kwargs):
        pass

    def interpolate(self, **kwargs):
        if 'output_path' not in kwargs.keys():
            output_path = "output-dir/"
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # For holding the names of x and y axis
        x_axis_name_1d = ""
        x_axis_name_2d = ""
        y_axis_name_2d = ""

        # A list to hold data which will be processed
        all_entries = {}

        for entry in self.input_data:
            results_list = []

            # Get the file name and path
            file_name = entry["file_name"]
            file_path = entry["file_path"]

            # Construct a full path to the input file
            full_file_path = file_path + file_name

            # Read the input file into a hdf5 object
            hdf5file = h5py.File(full_file_path, "r")
            # Get the dimension of the independent variable name(s) (1d or 2d)
            indep_dimension = len(entry["indep"])

            # A list to store the independent variable values
            indep_value = []

            # List to store number of bins for xas, or number of grids for map
            num_of_bins = []

            # A flag to check if the starting point in the independent
            # variable array is greater than the stopping point
            flip = False
            # The input scan is xas
            if (indep_dimension == 1):

                # Get the independent variable range and resolution
                indep_range = entry["range"].split(" ")
                if "resolution" in entry.keys():
                    resolution_1d = entry["resolution"]

                    # Calculate the number of regularly spaced bins to create
                    num_of_bins = int(abs(int(indep_range[1]) - int(indep_range[0])) / float(resolution_1d))
                    offset = float(resolution_1d) / 2
                    start = min(int(indep_range[0]), int(indep_range[1]))
                    stop = max(int(indep_range[0]), int(indep_range[1]))
                    bin_edges = np.linspace(start - offset, stop + offset, num_of_bins + 1)
                    bins = np.linspace(start, stop, num_of_bins)
                    # bin_edges = np.array([[np.amin(bin_edges[i:i+2]), np.amax(bin_edges[i:i+2])] for i in range(len(bins))])
                elif "bins" in entry.keys():
                    bins = np.array(entry["bins"])
                    offset1 = (bins[0] + bins[1]) / 2
                    offset2 = (bins[-2] + bins[-1]) / 2
                    bin_edges = [np.mean(bins[b:b + 1]) for b in range(len(bins[1:-1]))]
                    bin_edges.append(bins[-1] + offset2)
                    bin_edges.insert(0, bins[0] - offset1)

                # Get the independent variable name

                bin_value = np.array(hdf5file[entry["indep"][0]][()], dtype=np.float32)
                x_axis_name_1d = entry["indep"][0].split("/")[-1]
                nxentry = entry["indep"][0].split("/")[0]
                bin_label = np.full(bin_value.shape[0], np.nan)
                for i, b in enumerate(bins):
                    bin_label[np.where(np.logical_and(bin_value >= bin_edges[i], bin_value <= bin_edges[i + 1]))] = b

                axis = [x_axis_name_1d]
                idx = pd.Index(bins, name=['x'])
                # bin_label = [label if b < max(bin_edges[i:i+2]) and b > min(bin_edges[i:i+2]) else np.nan for i, label in enumerate(bins) for b in bin_value]
                df = dd.from_pandas(pd.DataFrame({x_axis_name_1d: bin_value, "bins": bin_label}), npartitions=1)
            # The input scan is map
            elif (indep_dimension == 2):

                # Get the x and y names
                x_axis_name_2d = entry["indep"][0].split("/")[-1]
                y_axis_name_2d = entry["indep"][1].split("/")[-1]

                # Get the x and y data
                x_value = np.array(hdf5file[entry["indep"][0]][()], dtype=np.float32)
                y_value = np.array(hdf5file[entry["indep"][1]][()], dtype=np.float32)

                # Pre-process the x values. The data needs to be shifted due to how they were collected
                shift = 0.5
                shifted_data = np.zeros(len(x_value), dtype=np.float32)
                shifted_data[0] = x_value[0]
                for i in range(1, len(x_value)):
                    shifted_data[i] = x_value[i] + shift * (x_value[i] - x_value[i - 1])

                # x_value is now shifted
                x_value = shifted_data

                # Get the independent variable range and resolution
                indep_range = [float(item) for item in entry["range"].split(" ")]
                if "resolution" in entry.keys():
                    resolution_2d = entry["resolution"]

                    # Split the resolution to get x and y resolution separately
                    resolution_2d = resolution_2d.split(" ")

                    # Calculate the number of grids to create for x and y
                    x_bins = int((indep_range[1] - indep_range[0]) / float(resolution_2d[0]))
                    y_bins = int((indep_range[3] - indep_range[2]) / float(resolution_2d[1]))

                    if (x_bins < 0 or y_bins < 0):
                        flip = True

                    num_of_bins = [abs(x_bins), abs(y_bins)]
                    startx = min(indep_range[:2])
                    stopx = max(indep_range[:2])
                    xoffset = float(resolution_2d[0]) / 2
                    starty = min(indep_range[2:])
                    stopy = max(indep_range[2:])
                    yoffset = float(resolution_2d[1]) / 2
                    bin_edges_x = np.linspace(startx - xoffset, stopx + xoffset, num_of_bins[0] + 1, dtype=np.float32)
                    bin_edges_y = np.linspace(starty - yoffset, stopy + yoffset, num_of_bins[1] + 1, dtype=np.float32)
                    regular_x = np.linspace(min(indep_range[:2]), max(indep_range[:2]), num_of_bins[0],
                                            dtype=np.float32)
                    regular_y = np.linspace(min(indep_range[2:]), max(indep_range[2:]), num_of_bins[1],
                                            dtype=np.float32)

                else:
                    print("Not implimented yet, skipping file %s %s" % (file_name, entry_name))
                    continue

                bin_value_x = x_value
                bin_label_x = np.full(bin_value_x.shape[0], np.nan)
                bin_value_y = y_value
                bin_label_y = np.full(bin_value_y.shape[0], np.nan)

                bins = [regular_x, regular_y]
                for i, b in enumerate(bins[0]):
                    bin_label_x[
                        np.where(np.logical_and(bin_value_x >= bin_edges_x[i], bin_value_x <= bin_edges_x[i + 1]))] = b
                for i, b in enumerate(bins[1]):
                    bin_label_y[
                        np.where(np.logical_and(bin_value_y >= bin_edges_y[i], bin_value_y <= bin_edges_y[i + 1]))] = b
                axis = [y_axis_name_2d, x_axis_name_2d]
                _x = np.array([-1 * regular_x if i % 2 else regular_x for i in range(len(regular_y))]).flatten()
                _y = np.array([[regular_y[j] for i in range(len(regular_x))] for j in range(len(regular_y))]).flatten()
                array = [_y, _x]
                idx = pd.MultiIndex.from_tuples(list(zip(*array)), names=['x', 'y'])
                df = dd.from_pandas(pd.DataFrame(
                    {x_axis_name_2d: bin_value_x, y_axis_name_2d: bin_value_y, "bins_x": bin_label_x,
                     "bins_y": bin_label_y}), npartitions=1)

            # Get all the dependent variables path
            dep_keys = [k for (k, v) in entry.items() if
                        k not in ["file_name", "file_path", "entry", "indep", "command", "range", "resolution"]]

            for k in dep_keys:
                # Get the full path of the dependent variable
                path = entry[k]

                # Get the entry name
                entry_name = path.split("/")[0]

                # Read the dependent variable data
                dep_value = np.array(hdf5file[path][()], dtype=np.float32)

                if flip and dep_value.ndim == 1:
                    dep_value = np.flip(dep_value, 0)
                    # print(file_name+"_"+entry_name+"_"+k, "flipped")

                # Add file and entry name, dependent variable name, independent variable value, dependent
                # variable value, independent variable dimension, and the number of bins to create
                if len(dep_value.shape) == 2:
                    temp = dd.from_pandas(
                        pd.DataFrame(dep_value, columns=[k + "_{}".format(i) for i in range(dep_value.shape[1])]),
                        npartitions=1)
                    new_df = df.join(temp)
                elif len(dep_value.shape) == 1:
                    temp = dd.from_pandas(pd.DataFrame(dep_value, columns=[k]), npartitions=1)
                    new_df = df.join(temp)
                if indep_dimension == 1:
                    final = new_df.groupby(by=["bins"]).mean()  # .reindex(index=f_index, method='nearest')
                elif indep_dimension == 2:
                    final = new_df.groupby(by=["bins_y",
                                               "bins_x"]).mean()  # .reindex(index=idx, inplace=True).interpolate(inplace=True).drop(columns=axis)
                results_list.append(final)

            all_entries.update({file_name + "/" + entry_name: [result.compute().reindex(index=idx).interpolate() for
                                                               result in results_list]})

            # Close connection to the "hdf5file"
            hdf5file.close()
        self.data = all_entries
        return all_entries

    def write(self, file, **kwargs):
        pass