import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import glob
import re

def extractVariableNames(filename):
    with open(filename, 'r') as file:
        dataBegin = re.compile(r'\d')
        lastHeaderLine = ''
        for line in file:
            if dataBegin.match(line[0]):
                break
            else:
                lastHeaderLine = line
        if lastHeaderLine:
            regex = re.compile(r' (?P<varName>\S+)')
            return regex.findall(lastHeaderLine)
        return []


def openCsv(path):
    regex = re.compile(r'\d')
    with open(path, 'r') as file:
        lines = filter(lambda x: regex.match(x[0]), file.readlines())
        return [[float(x) for x in line.split()] for line in lines]

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


def load_data_from_csv(path, experiment):  
    files = glob.glob(path)
    dataframes = []
    print(f'For experiment {experiment} loaded {len(files)} files')
    # print(files)
    for file in files:
        columns = extractVariableNames(file)
        data = openCsv(file)
        df = pd.DataFrame(data, columns=columns)
        dataframes.append(df)
    return dataframes

def compute_mean_variance(dfs):
    stacked = pd.concat(dfs, axis=0).groupby(level=0)
    mean_df = stacked.mean()
    variance_df = stacked.var(ddof=0)
    return mean_df, variance_df

def check_stability(dataset, metrics, window_size):
    # Iterate through the DataFrame with a sliding window
    for i in range(len(dataset) - window_size + 1):
        window = dataset.iloc[i : i + window_size]  # Extract window
    
        # Check if all values in the window are equal for all selected metrics
        if all((window[col] == window[col].iloc[0]).all() for col in metrics):
            first_index = i  # Store the first valid index
            break  # Stop at the first occurrence    
    return first_index


def plot_cutting(data, origin):
    colors = sns.color_palette("viridis", n_colors=len(data))
    metric = 'nodes'
    plt.figure(figsize=(10, 6))
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
        plt.fill_between(mean.index, lower_bound, upper_bound, color=colors[j], alpha=0.2)
    plt.axvline(x=500, color='black', linestyle='dotted', linewidth=2)
    plt.title(f'TODO')
    plt.xlabel('Seconds simulated')
    plt.ylabel('Number of nodes')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'charts/cut-at-{origin}.pdf', dpi=300)
    plt.close()

def plot_selfs(data, experiment, metric, y_label='Number of roots', cut = True):
    colors = sns.color_palette("viridis", n_colors=len(data))
    plt.figure(figsize=(10, 6))
    for j, ((exp, nodes), (mean_df, variance_df)) in enumerate(data.items()):
        sns.lineplot(
            data = mean_df,
            x = 'time',
            y = metric,
            label = f'initial nodes: {nodes}',
            color = colors[j],
        )
        mean = mean_df[metric]
        variance = variance_df[metric]
        upper_bound = mean + np.sqrt(variance)
        lower_bound = mean - np.sqrt(variance)
        plt.fill_between(mean.index, lower_bound, upper_bound, color=colors[j], alpha=0.2)
    if cut:
        plt.axvline(x=200, color='black', linestyle='dotted', linewidth=2)
    plt.title(f'{experiment}')
    plt.xlabel('Seconds simulated')
    plt.ylabel('Number of roots')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'charts/{experiment}.pdf', dpi=300)
    plt.close()

def plot_single_selfs(data, experiment, metric, y_label='Number of roots', cut = True):
    colors = sns.color_palette("viridis", n_colors=len(data))
    plt.figure(figsize=(10, 6))
    for j, ((exp, nodes), (mean_df, variance_df)) in enumerate(data.items()):
        sns.lineplot(
            data = mean_df,
            x = 'time',
            y = metric,
            label = f'initial nodes: {nodes}',
            color = colors[j],
        )
        mean = mean_df[metric]
        variance = variance_df[metric]
        upper_bound = mean + np.sqrt(variance)
        lower_bound = mean - np.sqrt(variance)
        plt.fill_between(mean.index, lower_bound, upper_bound, color=colors[j], alpha=0.2)
        if cut:
            plt.axvline(x=200, color='black', linestyle='dotted', linewidth=2)
        plt.title(f'{experiment}')
        plt.xlabel('Seconds simulated')
        plt.ylabel(y_label)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'charts/{experiment}_initialNodes-{nodes}.pdf', dpi=300)
        plt.close()

def box_plot(dataframes):
    df = pd.DataFrame(dataframes)
    df_melted = df.melt(var_name="Experiment", value_name="Stabilization time")
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x="Experiment", y="Stabilization time", data=df_melted, hue="Experiment", palette="viridis", legend=True)
    plt.xlabel("Experiment Type")
    plt.ylabel("Simulated seconds")
    plt.title("Stabilization time")
    handles = []
    for i, category in enumerate(df_melted["Experiment"].unique()):
        color = ax.patches[i].get_facecolor()  # Get color from boxplot
        handles.append(plt.Line2D([0], [0], color=color, lw=4))
    plt.legend(handles, df_melted["Experiment"].unique(), title="Legend", loc="upper right")
    plt.savefig(f'charts/stabilization-time.pdf', dpi=300)
    plt.close()

if __name__ == '__main__':

    experiments = ['classic-vmc', 'field-vmc-fixed-leader']
    metrics = ['network-hub-xCoord', 'network-hub-yCoord', 'nodes', 'network-diameter', 'network-density', 'nodes-degree[mean]']
    dataframes = {beautify_experiment_name(experiment): [] for experiment in experiments}

    for experiment in experiments:
        data = load_data_from_csv(f'data/{experiment}/{experiment}_*', experiment)
        times = []
        for seed in range(len(data)):
            metrics_to_check = {metric: [-1000] for metric in metrics}
            current_metrics_to_check = {metric: [-1000] for metric in metrics}
            equals_for = 0
            current_experiment_data = data[seed]
            index = check_stability(current_experiment_data, metrics, 30)
            time = current_experiment_data.loc[index, 'time']
            times.append(time)
        dataframes[beautify_experiment_name(experiment)].extend(times)
    box_plot(dataframes)
    
    origin = [0.1, 3.1, 6.1, 9.1, 12.1, 15.1, 18.1, 21.1, 24.1, 27.1, 30.1, 33.1, 36.1, 39.1, 42.1, 45.1, 48.1]
    experiments = ['cutting-classic-vmc', 'cutting-field-vmc-fixed-leader']
    for o in origin:
        dataframes = {}
        for experiment in experiments:
            data = load_data_from_csv(f'data/{experiment}/{experiment}_origin-{o}_*', experiment)
            mean, variance = compute_mean_variance(data)
            dataframes[beautify_experiment_name(experiment)] = (mean, variance)
        plot_cutting(dataframes, o)

    initialNodes = [1, 2, 10, 100, 300, 500]
    experiments = ['self-integration', 'self-repair']
    # experiment = 'self-integration'
    for experiment in experiments:
        dataframes = {}
        for n in initialNodes:
            data = load_data_from_csv(f'data/{experiment}/{experiment}_*_initialNodes-{n}.csv', experiment)
            mean, variance = compute_mean_variance(data)
            dataframes[beautify_experiment_name(experiment), n] = (mean, variance)
            # dataframes[f'{n}'] = (mean, variance)
        plot_selfs(dataframes, experiment, metric = 'ifit1@leader[Sum]', y_label='Number of roots', cut = True)
        plot_single_selfs(dataframes, experiment, metric = 'ifit1@leader[Sum]', cut = True)

    initialNodes = [1.0, 10, 100, 300, 500, 1000]
    experiment = 'self-optimization'
    dataframes = {}
    for n in initialNodes:
        data = load_data_from_csv(f'data/{experiment}/{experiment}_*_initialNodes-{n}.csv', experiment)
        mean, variance = compute_mean_variance(data)
        dataframes[beautify_experiment_name(experiment), n] = (mean, variance)
        # dataframes[f'{n}'] = (mean, variance)
    plot_selfs(dataframes, experiment, 'nodes', 'Number of nodes', False)
    plot_single_selfs(dataframes, experiment, 'nodes', 'Number of nodes', False)