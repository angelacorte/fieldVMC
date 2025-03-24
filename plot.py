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


def load_data_from_csv(path, experiment, metric):  
    files = glob.glob(path)
    dataframes = []
    print(f'For experiment {experiment} and metric {metric} loaded {len(files)} files')
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
    plt.title(f'Origin {origin}')
    plt.xlabel('Time')
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'charts/origin-{origin}.pdf', dpi=300)
    plt.close()

def plot_selfs(data, origin, experiment, metric, y_label='Number of roots', cut = True):
    colors = sns.color_palette("viridis", n_colors=len(data))
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
    if cut:
        plt.axvline(x=200, color='black', linestyle='dotted', linewidth=2)
    plt.title(f'{experiment} - initial nodes: {origin}')
    plt.xlabel('Time')
    plt.ylabel(y_label)
    plt.ylim(0, 14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'charts/{experiment}.pdf', dpi=300)
    plt.close()


if __name__ == '__main__':
    
    # origin = [0.1, 3.1, 6.1, 9.1, 12.1, 15.1, 18.1, 21.1, 24.1, 27.1, 30.1, 33.1, 36.1, 39.1, 42.1, 45.1, 48.1]
    # experiments = ['cutting-classic-vmc', 'cutting-field-vmc-fixed-leader']
    # for o in origin:
    #     dataframes = {}
    #     for experiment in experiments:
    #         data = load_data_from_csv(f'data/{experiment}', experiment, o)
    #         mean, variance = compute_mean_variance(data)
    #         dataframes[beautify_experiment_name(experiment)] = (mean, variance)
    #     plot_cutting(dataframes, o)

    initialNodes = [1, 2, 10, 100, 300, 500]
    experiments = ['self-integration', 'self-repair']
    # experiment = 'self-integration'
    for experiment in experiments:
        dataframes = {}
        for n in initialNodes:
            data = load_data_from_csv(f'data/{experiment}/{experiment}_*_initialNodes-{n}.csv', experiment, n)
            mean, variance = compute_mean_variance(data)
            dataframes[f'{n}'] = (mean, variance)
        plot_selfs(dataframes, n, experiment, metric = 'ifit1@leader[Sum]', cut = False)

    initialNodes = [1.0, 10, 100, 300, 500, 1000]
    experiment = 'self-optimization'
    dataframes = {}
    for n in initialNodes:
        data = load_data_from_csv(f'data/{experiment}/{experiment}_*_initialNodes-{n}.csv', experiment, n)
        mean, variance = compute_mean_variance(data)
        dataframes[f'{n}'] = (mean, variance)
    plot_selfs(dataframes, n, experiment, 'nodes', 'Number of nodes', False)