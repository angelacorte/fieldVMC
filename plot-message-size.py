#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 19:27:30 2025

@author: gienna
"""

from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import seaborn as sns
import pandas as pd
import matplotlib
import re
import os
import argparse

def distance(val, ref):
    return abs(ref - val)

vectDistance = np.vectorize(distance)

def cmap_xmap(function, cmap):
    """ Applies function, on the indices of colormap cmap. Beware, function
    should map the [0, 1] segment to itself, or you are in for surprises.

    See also cmap_xmap.
    ""    """
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('viridis')
    cdict = cmap._segmentdata
    function_to_map = lambda x: (function(x[0]), x[1], x[2])
    for key in ('red', 'green', 'blue'):
        cdict[key] = map(function_to_map, cdict[key])
    #        cdict[key].sort()
    #        assert (cdict[key][0]<0 or cdict[key][-1]>1), "Resulting indices extend out of the [0, 1] segment."
    return matplotlib.colors.LinearSegmentedColormap('colormap', cdict, 1024)

def getClosest(sortedMatrix, column, val):
    while len(sortedMatrix) > 3:
        half = int(len(sortedMatrix) / 2)
        sortedMatrix = sortedMatrix[-half - 1:] if sortedMatrix[half, column] < val else sortedMatrix[: half + 1]
    if len(sortedMatrix) == 1:
        result = sortedMatrix[0].copy()
        result[column] = val
        return result
    else:
        safecopy = sortedMatrix.copy()
        safecopy[:, column] = vectDistance(safecopy[:, column], val)
        minidx = np.argmin(safecopy[:, column])
        safecopy = safecopy[minidx, :].A1
        safecopy[column] = val
        return safecopy

def convert(column, samples, matrix):
    return np.matrix([getClosest(matrix, column, t) for t in samples])

def valueOrEmptySet(k, d):
    return (d[k] if isinstance(d[k], set) else {d[k]}) if k in d else set()

def mergeDicts(d1, d2):
    """
    Creates a new dictionary whose keys are the union of the keys of two
    dictionaries, and whose values are the union of values.

    Parameters
    ----------
    d1: dict
        dictionary whose values are sets
    d2: dict
        dictionary whose values are sets

    Returns
    -------
    dict
        A dict whose keys are the union of the keys of two dictionaries,
    and whose values are the union of values

    """
    res = {}
    for k in d1.keys() | d2.keys():
        res[k] = valueOrEmptySet(k, d1) | valueOrEmptySet(k, d2)
    return res

def extractCoordinates(filename):
    """
    Scans the header of an Alchemist file in search of the variables.

    Parameters
    ----------
    filename : str
        path to the target file
    mergewith : dict
        a dictionary whose dimensions will be merged with the returned one

    Returns
    -------
    dict
        A dictionary whose keys are strings (coordinate name) and values are
        lists (set of variable values)

    """
    with open(filename, 'r') as file:
        #        regex = re.compile(' (?P<varName>[a-zA-Z._-]+) = (?P<varValue>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),?')
        regex = r"(?P<varName>[a-zA-Z._-]+) = (?P<varValue>(?:\[[^\]]*\]|[^,]*)),?"
        dataBegin = r"\d"
        is_float = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
        for line in file:
            match = re.findall(regex, line.replace('Infinity', '1e30000'))
            if match:
                return {
                    var: float(value) if re.match(is_float, value)
                    else bool(re.match(r".*?true.*?", value.lower())) if re.match(r".*?(true|false).*?", value.lower())
                    else value
                    for var, value in match
                }
            elif re.match(dataBegin, line[0]):
                return {}

def extractVariableNames(filename):
    """
    Gets the variable names from the Alchemist data files header.

    Parameters
    ----------
    filename : str
        path to the target file

    Returns
    -------
    list of list
        A matrix with the values of the csv file

    """
    with open(filename, 'r') as file:
        dataBegin = re.compile('\d')
        lastHeaderLine = ''
        for line in file:
            if dataBegin.match(line[0]):
                break
            else:
                lastHeaderLine = line
        if lastHeaderLine:
            regex = re.compile(" (?P<varName>\S+)")
            return regex.findall(lastHeaderLine)
        return []


def openCsv(path):
    """
    Converts an Alchemist export file into a list of lists representing the matrix of values.

    Parameters
    ----------
    path : str
        path to the target file

    Returns
    -------
    list of list
        A matrix with the values of the csv file

    """
    regex = re.compile('\d')
    with open(path, 'r') as file:
        lines = filter(lambda x: regex.match(x[0]), file.readlines())
        return [[float(x) for x in line.split()] for line in lines]

def beautifyValue(v):
    """
    Converts an object to a better version for printing, in particular:
        - if the object converts to float, then its float value is used
        - if the object can be rounded to int, then the int value is preferred

    Parameters
    ----------
    v : object
        the object to try to beautify

    Returns
    -------
    object or float or int
        the beautified value
    """
    try:
        v = float(v)
        if v.is_integer():
            return int(v)
        return v
    except:
        return v

    class Measure:
        def __init__(self, description, unit=None):
            self.__description = description
            self.__unit = unit

        def description(self):
            return self.__description

        def unit(self):
            return '' if self.__unit is None else f'({self.__unit})'

        def derivative(self, new_description=None, new_unit=None):
            def cleanMathMode(s):
                return s[1:-1] if s[0] == '$' and s[-1] == '$' else s

            def deriveString(s):
                return r'$d ' + cleanMathMode(s) + r'/{dt}$'

            def deriveUnit(s):
                return f'${cleanMathMode(s)}' + '/{s}$' if s else None

            result = Measure(
                new_description if new_description else deriveString(self.__description),
                new_unit if new_unit else deriveUnit(self.__unit),
            )
            return result

        def __str__(self):
            return f'{self.description()} {self.unit()}'

    centrality_label = 'H_a(x)'

    def expected(x):
        return r'\mathbf{E}[' + x + ']'

    def stdev_of(x):
        return r'\sigma{}[' + x + ']'

    def mse(x):
        return 'MSE[' + x + ']'

    def cardinality(x):
        return r'\|' + x + r'\|'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default="leader-election", help='Type of experiment to plot (leader-election or fixed-leader)')
    args = parser.parse_args()
    print(args.experiment)

    current_experiment = args.experiment

    directory = 'data'
    # Where to save charts
    current_datetime = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_directory = f'charts/{current_experiment}_{current_datetime}'
    os.makedirs(output_directory, exist_ok=True)
    # How to name the summary of the processed data
    pickleOutput = f'messages-self-construction-{current_experiment}'
    # Experiment prefixes: one per experiment (root of the file name)
    experiments = [f'messages-self-construction-{current_experiment}'] #'messages-self-construction-fixed-leader',
    floatPrecision = '{: 0.3f}'
    # Number of time samples
    timeSamples = 200
    # time management
    minTime = 0
    maxTime = 50000
    if current_experiment == 'fixed-leader':
        maxTime = 200
    timeColumnName = 'time'
    logarithmicTime = False
    # One or more variables are considered random and "flattened"
    seedVars = ['seed'] #, 'maxResource', 'maxSuccess', 'resourceLowerBound'
    # Label mapping

    # Setup libraries
    np.set_printoptions(formatter={'float': floatPrecision.format})
    # Read the last time the data was processed, reprocess only if new data exists, otherwise just load
    import pickle
    import os

    if os.path.exists(directory):
        newestFileTime = max([os.path.getmtime(directory + '/' + file) for file in os.listdir(directory)], default=0.0)
        try:
            lastTimeProcessed = pickle.load(open('timeprocessed', 'rb'))
        except:
            lastTimeProcessed = -1
        shouldRecompute = not os.path.exists(".skip_data_process") and newestFileTime != lastTimeProcessed
        if not shouldRecompute:
            try:
                means = pickle.load(open(pickleOutput + '_mean', 'rb'))
                stdevs = pickle.load(open(pickleOutput + '_std', 'rb'))
            except:
                shouldRecompute = True
        if shouldRecompute:
            timefun = np.logspace if logarithmicTime else np.linspace
            means = {}
            stdevs = {}
            for experiment in experiments:
                # Collect all files for the experiment of interest
                import fnmatch
                allfiles = filter(lambda file: fnmatch.fnmatch(file, experiment + '_*.csv'), os.listdir(f'{directory}/{experiment}'))
                allfiles = [directory + f'/{experiment}/' + name for name in allfiles]
                allfiles.sort()
                # From the file name, extract the independent variables
                dimensions = {}
                for file in allfiles:
                    dimensions = mergeDicts(dimensions, extractCoordinates(file))
                dimensions = {k: sorted(v) for k, v in dimensions.items()}
                # Add time to the independent variables
                dimensions[timeColumnName] = range(0, timeSamples)
                # Compute the matrix shape
                shape = tuple(len(v) for k, v in dimensions.items())
                # Prepare the Dataset
                dataset = xr.Dataset()
                for k, v in dimensions.items():
                    dataset.coords[k] = v
                if len(allfiles) == 0:
                    print("WARNING: No data for experiment " + experiment)
                    means[experiment] = dataset
                    stdevs[experiment] = xr.Dataset()
                else:
                    varNames = extractVariableNames(allfiles[0])
                    for v in varNames:
                        if v != timeColumnName:
                            novals = np.ndarray(shape)
                            novals.fill(float('nan'))
                            dataset[v] = (dimensions.keys(), novals)
                    # Compute maximum and minimum time, create the resample
                    timeColumn = varNames.index(timeColumnName)
                    allData = {file: np.matrix(openCsv(file)) for file in allfiles}
                    computeMin = minTime is None
                    computeMax = maxTime is None
                    if computeMax:
                        maxTime = float('-inf')
                        for data in allData.values():
                            maxTime = max(maxTime, data[-1, timeColumn])
                    if computeMin:
                        minTime = float('inf')
                        for data in allData.values():
                            minTime = min(minTime, data[0, timeColumn])
                    timeline = timefun(minTime, maxTime, timeSamples)
                    # Resample
                    for file in allData:
                        #                    print(file)
                        allData[file] = convert(timeColumn, timeline, allData[file])
                    # Populate the dataset
                    for file, data in allData.items():
                        dataset[timeColumnName] = timeline
                        for idx, v in enumerate(varNames):
                            if v != timeColumnName:
                                darray = dataset[v]
                                experimentVars = extractCoordinates(file)
                                darray.loc[experimentVars] = data[:, idx].A1


                    # Fold the dataset along the seed variables, producing the mean and stdev datasets
                    mergingVariables = [seed for seed in seedVars if seed in dataset.coords]
                    means[experiment] = dataset.mean(dim=mergingVariables, skipna=True)
                    stdevs[experiment] = dataset.std(dim=mergingVariables, skipna=True)
            # Save the datasets
            pickle.dump(means, open(pickleOutput + '_mean', 'wb'), protocol=-1)
            pickle.dump(stdevs, open(pickleOutput + '_std', 'wb'), protocol=-1)
            pickle.dump(newestFileTime, open('timeprocessed', 'wb'))
    else:
        means = {experiment: xr.Dataset() for experiment in experiments}
        stdevs = {experiment: xr.Dataset() for experiment in experiments}

    for experiment in experiments:
        current_experiment_means = means[experiment]
        current_experiment_errors = stdevs[experiment]


    def beautify_experiment_name(name):
        if name == 'messages-self-construction-leader-election':
            return 'Leader Election'
        if name == 'messages-self-construction-fixed-leader':
            return 'Fixed Leader'
        else:
            raise Exception(f'Unknown experiment name {name}.')

    def plot_selfs_dual_axis(data, experiment, metric):
        plt.rcParams.update({'font.size': 15})
        plt.rcParams.update({'legend.loc': 0})
        colors = sns.color_palette("viridis", n_colors=len(data) + 2)

        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax2 = ax1.twinx()  # secondo asse Y
        children = list(data.keys())[0][1][1]

        for j, (((_, resources), (_, children)), (mean_df, std_df)) in enumerate(data.items()):
            sns.lineplot(
                data=mean_df,
                x='time',
                y=metric,
                label=f'{resources}',
                ax=ax1,
                color=colors[j+2],
            )
            sns.lineplot(
                data=mean_df,
                x='time',
                y='nodes',
                label='_nolegend_',
                ax=ax2,
                linestyle="--",
                color=colors[j+2],
            )

        ax1.set_xlim(0, 50000)
        if experiment == 'Fixed Leader':
            ax1.set_xlim(0, 100)

        ax1.set_xlabel('Simulated seconds')
        ax1.set_ylabel('Message Size (KB)')
        ax2.set_ylabel('Number of nodes')

        # Legenda unica: solo maxResources; stile spiegato nel titolo
        handles, labels = ax1.get_legend_handles_labels()
        if handles:
            ax1.legend(handles, labels, title='env resources (solid=KB, dashed=nodes)', loc='best')

        plt.title(f'{experiment} Spawnable Children = {children}')
        plt.tight_layout()
        plt.savefig(f'{output_directory}/{experiment}-dualaxis_{metric}-children{children}.pdf', dpi=300)
        plt.close()

    def plot_selfs(data, experiment, metric):
        i = len(data)+2
        plt.rcParams.update({'font.size': 15})
        plt.rcParams.update({'legend.loc': 0})
        colors = sns.color_palette("viridis", n_colors=i)
        plt.figure(figsize=(9, 5))
        children = list(data.keys())[0][1][1]
        for j, ((exp, resources), (mean_df, std_df)) in enumerate(data.items()):
            sns.lineplot(
                data=mean_df,
                x='time',
                y=metric,
                label=f'env resources = {exp[1]}$',
                color=colors[j+2],
            )
            # upper_bound = mean_df[metric] + std_df[f'{metric}-std']
            # lower_bound = mean_df[metric] - std_df[f'{metric}-std']
            # plt.fill_between(mean_df['time'], lower_bound, upper_bound, color=colors[j+2], alpha=0.2)
            plt.title(f'{experiment} Spawnable Children = {children}')
        plt.xlim(0, 50000)
        if experiment == 'Fixed Leader':
            plt.xlim(0, 100)
        plt.xlabel('Simulated seconds')
        plt.ylabel(f'{metric} KB')
        if metric == 'nodes':
            plt.ylabel('Number of nodes')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_directory}/{experiment}-{metric}-children{children}.pdf', dpi=300)
        plt.close()

    data_dict = {}
    # maxResource = [ 30, 60, 120, 240, 300 ]
    maxResource = [50, 100, 250, 500, 1000, 1500]
    maxChildren = [2, 3, 4, 5]
    metric_to_plot = ['MessageSize[mean]', 'nodes', 'MessageSize[Sum]']

    for metric in metric_to_plot:
        for experiment in experiments:
            for children in maxChildren:
                data_dict = {}
                for resources in maxResource:
                    metric_series_mean = means[experiment][metric].sel(dict(maxResource=resources, maxChildren=children))
                    metric_series_std = stdevs[experiment][metric].sel(dict(maxResource=resources, maxChildren=children))
                    time_series = metric_series_mean['time'].values
                    if metric == 'MessageSize[mean]' or metric == 'MessageSize[Sum]':
                        metric_series_mean = metric_series_mean.values / 1024 # convert to KB
                    nodes_series = means[experiment]['nodes'].sel(dict(maxResource=resources, maxChildren=children)).values

                    df_mean = pd.DataFrame({
                        'time': time_series,
                        metric: metric_series_mean,
                        'nodes': nodes_series,
                    })
                    df_std = pd.DataFrame({
                        'time': time_series,
                        f'{metric}-std': metric_series_std,
                    })

                    data_dict[(f"res-{resources}", resources),(f"ch-{children}",children)] = (df_mean, df_std)
                exp_name = beautify_experiment_name(experiment)
                if metric in ['MessageSize[mean]', 'MessageSize[Sum]']:
                    plot_selfs_dual_axis(data_dict, experiment=exp_name, metric=metric)
                else:
                    plot_selfs(data_dict, experiment=exp_name, metric=metric)
