import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import re
from pathlib import Path
import seaborn as sns
import pandas as pd
import matplotlib
import glob
import re
from datetime import datetime

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


# =============================================================================
#     def derivativeOrMeasure(variable_name):
#         if variable_name.endswith('dt'):
#             return labels.get(variable_name[:-2], Measure(variable_name)).derivative()
#         return Measure(variable_name)
# 
# 
#     def label_for(variable_name):
#         return labels.get(variable_name, derivativeOrMeasure(variable_name)).description()
# 
# 
#     def unit_for(variable_name):
#         return str(labels.get(variable_name, derivativeOrMeasure(variable_name)))
# =============================================================================

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
    # CONFIGURE SCRIPT
    # Where to find Alchemist data files
    directory = 'data'
    # Where to save charts
    output_directory = 'charts/good'
    # How to name the summary of the processed data
    pickleOutput = 'data_summary'
    # Experiment prefixes: one per experiment (root of the file name)
# =============================================================================
    experiments = ['self-repair', 'cutting-field-vmc-fixed-leader', 'cutting-classic-vmc', 'field-vmc-fixed-leader', 'classic-vmc','self-integration', 'self-optimization']
# =============================================================================
 #   experiments = ['self-integration',]
    #'self-integration', 'self-repair', 'self-optimization'
    floatPrecision = '{: 0.3f}'
    # Number of time samples
    timeSamples = 200
    # time management
    minTime = 0
    maxTime = 6000
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



# Custom charting
# plot in a single boxplot chart by using seaborn, the data of both experiments "classic-vmc" and "field-vmc-fixed-leader",
# comparing the stabilization time of the two experiments. The x-axis should represent the experiment name,
# and the y-axis should represent the stabilization time. The title of the chart should be "Stabilization Time Comparison".
# the stabilization time is the amount of time elapsed from the start of the experiment to the end of the experiment.


def beautify_experiment_name(name):
    if name == 'classic-vmc':
        return 'Classic VMC'
    if name == 'field-vmc-fixed-leader':
        return 'Field VMC'
    if name == 'cutting-classic-vmc':
        return 'Classic VMC'
    if name == 'cutting-field-vmc-fixed-leader':
        return 'Field VMC'
    if name == 'self-integration':
        return 'Field VMC Self-Integration'
    if name == 'self-optimization':
        return 'Field VMC Self-Optimization'
    if name == 'self-repair':
        return 'Field VMC Self-Repair'
    else:
        raise Exception(f'Unknown experiment name {name}.')


# def plot_cutting(data, origin):
#     i = len(data)+1
#     colors = sns.color_palette("viridis", n_colors=i)
#     metric = 'nodes'
#     plt.figure(figsize=(10, 6))
#     for j, (algorithm, (mean_df, variance_df)) in enumerate(data.items()):
#         sns.lineplot(
#             data = mean_df,
#             x = 'time',
#             y = metric,
#             label = algorithm,
#             color = colors[j],
#         )
#         mean = mean_df[metric]
#         variance = variance_df[metric]
#         upper_bound = mean + np.sqrt(variance)
#         lower_bound = mean - np.sqrt(variance)
#         plt.fill_between(mean.index, lower_bound, upper_bound, color=colors[j], alpha=0.2)
#     plt.axvline(x=500, color=colors[0], linestyle='dotted', linewidth=1, label='Cut event')
#     plt.title(f'Cut event from y={origin}')
#     plt.xlabel('Seconds simulated')
#     plt.ylabel('Number of nodes')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f'{output_directory}/cut-at-{origin}.pdf', dpi=300)
#     plt.close()

def plot_selfs(data, experiment, metric, y_label='Number of roots', cut=True):
    i = len(data)+2
    plt.rcParams.update({'font.size': 15})
    colors = sns.color_palette("viridis", n_colors=i)
    plt.figure(figsize=(9, 5))
    for j, ((exp, nodes), (mean_df, std_df)) in enumerate(data.items()):
        sns.lineplot(
            data=mean_df,
            x='time',
            y=metric,
            label=f'initial nodes: {nodes}',
            color=colors[j+2],
        )
        upper_bound = mean_df[metric] + std_df['std']
        lower_bound = mean_df[metric] - std_df['std']
        plt.fill_between(mean_df['time'], lower_bound, upper_bound, color=colors[j+2], alpha=0.2)

    if cut:
        y_target = 2
        if experiment == beautify_experiment_name('self-integration'):
            plt.xlim(180, 230)
            plt.ylim(0, 5)
            y_target = 1
        plt.axhline(y=y_target, color=colors[1], linestyle='--', linewidth=2, label='Target')
    plt.axvline(x=200, color=colors[0], linestyle='dotted', linewidth=2, label='Event')
    
    plt.title(experiment)
    plt.xlabel('Seconds simulated')
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    now = datetime.now()
    now = now.strftime("%Y-%m-%d-%H-%M-%S")
    plt.savefig(f'{output_directory}/{experiment}{now}.pdf', dpi=300)
    plt.show()  
    plt.close() 
    
    
def plot_cutting(data, origin):
    i = len(data)+1
    plt.rcParams.update({'font.size': 15})
    colors = sns.color_palette("viridis", n_colors=i)
    metric = 'nodes'
    plt.figure(figsize=(9, 5))
    for j, (algorithm, (mean_df, variance_df)) in enumerate(data.items()):
        sns.lineplot(
            data = mean_df,
            x = 'time',
            y = metric,
            label = algorithm,
            color = colors[j],
        )
        mean = mean_df[metric]
        variance = variance_df[metric]
        upper_bound = mean + np.sqrt(variance)
        lower_bound = mean - np.sqrt(variance)
        plt.fill_between(mean.index, lower_bound, upper_bound, color=colors[j+1], alpha=0.2)
    plt.xlim(0, 1000)
    plt.axvline(x=500, color=colors[0], linestyle='--', linewidth=2, label='Cut event')
    plt.title(f'Cut event from y={origin}')
    plt.xlabel('Simulated seconds')
    plt.ylabel('Number of nodes')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{directory}/cut-at-{origin}.pdf', dpi=300)
    plt.close()


    # def plot_selfs_twinx(data, experiment, metric, secondary_metric=None, y_label='Number of roots', secondary_y_label='Success per node', cut=True):
    #     i = (len(data) + 2)*2
    #     plt.rcParams.update({'font.size': 15})
    #     colors = sns.color_palette("viridis", n_colors=i)
    #     plt.figure(figsize=(20, 12))

    #     ax1 = plt.gca()
    #     for j, ((exp, nodes), (mean_df, variance_df)) in enumerate(data.items()):
    #         sns.lineplot(
    #             ax=ax1,
    #             data=mean_df,
    #             x='time',
    #             y=metric,
    #             label=f'initial nodes: {nodes}',
    #             color=colors[(j * 2) + 1],
    #         )
    #         mean = mean_df[metric]
    #         variance = variance_df[metric]
    #         upper_bound = mean + np.sqrt(variance)
    #         lower_bound = mean - np.sqrt(variance)
    #         ax1.fill_between(mean.index, lower_bound, upper_bound, color=colors[(j * 2) + 1], alpha=0.2)

    #     ax1.set_xlabel('Simulated seconds')
    #     ax1.set_ylabel(y_label)
    #     ax1.set_title(f'{experiment}')

    #     if secondary_metric:
    #         ax2 = ax1.twinx()
    #         for j, ((exp, nodes), (mean_df, variance_df)) in enumerate(data.items()):
    #             mean_df['success_per_node'] = mean_df[secondary_metric] / mean_df['nodes']
    #             sns.lineplot(
    #                 ax=ax2,
    #                 data=mean_df,
    #                 x='time',
    #                 y=secondary_metric,
    #                 label=f'Success per node - nodes: {nodes}',
    #                 color=colors[(j * 2) + 2],
    #             )
    #         ax2.set_ylabel(secondary_y_label)

    #     if cut:
    #         ax1.set_ylim(0, 15)
    #         if experiment == beautify_experiment_name('self-integration'):
    #             ax1.set_xlim(-10, 300)
    #             ax1.axhline(y=2, color=colors[1], linestyle='--', linewidth=1, label='Target')
    #         ax1.axvline(x=200, color=colors[0], linestyle='dotted', linewidth=1, label='Event')

    #     ax1.legend(loc='upper left')
    #     if secondary_metric:
    #         ax2.legend(loc='upper right')

    #     plt.tight_layout()
    #     now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    #     plt.savefig(f'{directory}/{experiment}WithSuccessPerNode{now}.pdf', dpi=300)
    #     plt.close()


    # def plot_single_selfs(data, experiment, metric, y_label='Number of roots', cut = True):
    #     i = len(data)+2
    #     plt.rcParams.update({'font.size': 13})
    #     colors = sns.color_palette("viridis", n_colors=i)
    #     plt.figure(figsize=(20, 12))
    #     for j, ((exp, nodes), (mean_df, variance_df)) in enumerate(data.items()):
    #         sns.lineplot(
    #             data = mean_df,
    #             x = 'time',
    #             y = metric,
    #             label = f'initial nodes: {nodes}',
    #             color = colors[j+2],
    #         )
    #         mean = mean_df[metric]
    #         variance = variance_df[metric]
    #         upper_bound = mean + np.sqrt(variance)
    #         lower_bound = mean - np.sqrt(variance)
    #         plt.fill_between(mean.index, lower_bound, upper_bound, color=colors[j+2], alpha=0.2)
    #         if cut:
    #             plt.ylim(0, 15)
    #             if experiment == 'self-integration':
    #                 plt.xlim(-10, 300)
    #             plt.axhline(y=1, color=colors[1], linestyle='--', linewidth=1, label='Target')
    #         plt.axvline(x=200, color=colors[0], linestyle='dotted', linewidth=1, label='Event')
    #         plt.title(f'{experiment}')
    #         plt.xlabel('Simulated seconds')
    #         plt.ylabel(y_label)
    #         plt.legend()
    #         plt.tight_layout()
    #         now = datetime.now()
    #         now = now.strftime("%Y-%m-%d-%H-%M-%S")
    #         plt.savefig(f'{directory}/{experiment}_initialNodes-{nodes}-{now}.pdf', dpi=300)
    #         plt.close()

    def box_plot(dataframes):
        plt.rcParams.update({'font.size': 16})
        df = pd.DataFrame(dataframes)
        df_melted = df.melt(var_name="Experiment", value_name="Stabilisation time")
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(x="Experiment", y="Stabilisation time", data=df_melted, hue="Experiment", palette="viridis", legend=True)
        plt.ylabel("Simulated seconds")
        plt.title("Stabilisation time")
        handles = []
        for i, category in enumerate(df_melted["Experiment"].unique()):
            color = ax.patches[i].get_facecolor()  # Get color from boxplot
            handles.append(plt.Line2D([0], [0], color=color, lw=4))
        plt.legend(handles, df_melted["Experiment"].unique(), title="Legend", loc="upper right")
        plt.savefig(f'{directory}/stabilization-time.pdf', dpi=300)
        plt.close()

    def violin_plot(dataframes):
        plt.rcParams.update({'font.size': 16})
        df = pd.DataFrame(dataframes)
        df_melted = df.melt(var_name="Experiment", value_name="Stabilisation time")
        plt.figure(figsize=(10, 6))
        sns.violinplot(x="Experiment", y="Stabilisation time", data=df_melted, palette="viridis", hue="Experiment")
        plt.ylabel("Simulated seconds")
        plt.title("Stabilisation time")
        plt.savefig(f'{directory}/stabilization-time-violin.pdf', dpi=300)
        plt.close()

# =============================================================================
# 
# 
# def box_plot(dataframes):
#     df = pd.DataFrame(dataframes)
#     df_melted = df.melt(var_name="Experiment", value_name="Stabilization time")
#     plt.figure(figsize=(10, 6))
#     ax = sns.boxplot(x="Experiment", y="Stabilization time", data=df_melted, hue="Experiment", palette="viridis", legend=True)
#     plt.xlabel("Experiment Type")
#     plt.ylabel("Simulated seconds")
#     plt.title("Stabilization time")
#     handles = []
#     for i, category in enumerate(df_melted["Experiment"].unique()):
#         color = ax.patches[i].get_facecolor()  # Get color from boxplot
#         handles.append(plt.Line2D([0], [0], color=color, lw=4))
#     plt.legend(handles, df_melted["Experiment"].unique(), title="Legend", loc="upper right")
#     plt.savefig(f'{output_directory}/stabilization-time.pdf', dpi=300)
#     plt.close()
# 
# def violin_plot(dataframes):
#     df = pd.DataFrame(dataframes)
#     df_melted = df.melt(var_name="Experiment", value_name="Stabilization time")
#     plt.figure(figsize=(10, 6))
#     sns.violinplot(x="Experiment", y="Stabilization time", data=df_melted, palette="viridis", hue="Experiment")
#     plt.xlabel("Experiment Type")
#     plt.ylabel("Simulated seconds")
#     plt.title("Stabilization time")
#     plt.savefig(f'{output_directory}/stabilization-time-violin.pdf', dpi=300)
#     plt.close()
# 
# 
# =============================================================================
def check_stability(dataset, metrics, window_size):
    for i in range(len(dataset) - window_size + 1):
        window = dataset.iloc[i : i + window_size]  
        if all((window[col] == window[col].iloc[0]).all() for col in metrics):
            first_index = i
            break  
    return first_index


def beautify_experiment_name(name):
    if name == 'classic-vmc':
        return 'Classic VMC'
    if name == 'field-vmc-fixed-leader':
        return 'Field VMC'
    if name == 'cutting-classic-vmc':
        return 'Classic VMC'
    if name == 'cutting-field-vmc-fixed-leader':
        return 'Field VMC'
    if name == 'self-integration':
        return 'Field VMC Self-Integration'
    if name == 'self-optimization':
        return 'Field VMC Self-Optimization'
    if name == 'self-repair':
        return 'Field VMC Self-Repair'
    else:
        raise Exception(f'Unknown experiment name {name}.')

from matplotlib import pyplot as plt

metric_name = 'ifit1@leader[Sum]'
data_dict = {}
initialNodes = [100, 300, 500]
experiments = ['self-integration'] #'self-repair', 
metric_to_plot = 'ifit1@leader[Sum]'

for experiment in experiments:
    for nodes in initialNodes:
        metric_series_mean = means[experiment][metric_to_plot].sel(dict(initialNodes=nodes))
        metric_series_std = stdevs[experiment][metric_to_plot].sel(dict(initialNodes=nodes))
        time_series = metric_series_mean['time'].values
        df_mean = pd.DataFrame({
            'time': time_series,
            metric_name: metric_series_mean
        })
        df_std = pd.DataFrame({
            'time': time_series,
            'std': metric_series_std
        })
        
        data_dict[(f"{nodes}", nodes * 2)] = (df_mean, df_std)
        exp_name = beautify_experiment_name(experiment)
    plot_selfs(data_dict, experiment=exp_name, metric=metric_name, cut=True)



# experiments = ['classic-vmc', 'field-vmc-fixed-leader']
# metrics = ['network-hub-xCoord', 'network-hub-yCoord', 'nodes', 'network-diameter', 'network-density', 'nodes-degree[mean]']
# dataframes = {beautify_experiment_name(experiment): [] for experiment in experiments}

# # all_time = { experiment : []  for experiment in experiments }
# for experiment in experiments:
#     current_experiment_means = means[experiment]
#     times = []
#     seeds = current_experiment_means.seed.values
#     for seed in seeds:
#         metrics_to_check = {metric: [-1000] for metric in metrics}
#         current_metrics_to_check = {metric: [-1000] for metric in metrics}
#         equals_for = 0
#         current_experiment_data = data[seed]
#         index = check_stability(current_experiment_data, metrics, 30)
#         time = current_experiment_data.loc[index, 'time']
#         times.append(time)
#     dataframes[beautify_experiment_name(experiment)].extend(times)
# box_plot(dataframes)
# violin_plot(dataframes)

# # origin = [0.1, 3.1, 6.1, 9.1, 12.1, 15.1, 18.1, 21.1, 24.1, 27.1, 30.1, 33.1, 36.1, 39.1, 42.1, 45.1, 48.1]
# origin = [0.1, 9.1, 18.1, 27.1]
# experiments = ['cutting-classic-vmc', 'cutting-field-vmc-fixed-leader']
# for o in origin:
#     dataframes = {}
#     for experiment in experiments:
#         data = load_data_from_csv(f'data/{experiment}/{experiment}_origin-{o}_*', experiment)
#         mean, variance = compute_mean_variance(data)
#         dataframes[beautify_experiment_name(experiment)] = (mean, variance)
#     plot_cutting(dataframes, o)
