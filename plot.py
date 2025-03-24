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
    else:
        raise Exception(f'Unknown experiment name {name}.')


def load_data_from_csv(path, experiment, metric):  
    files = glob.glob(f'{path}/{experiment}*-{metric}*.csv')
    dataframes = []
    print(f'For experiment {experiment} and metric {metric} loaded {len(files)} files')
    for file in files:
        columns = extractVariableNames(file)
        data = openCsv(file)
        df = pd.DataFrame(data, columns=columns)
        # df['Experiment'] = beautify_experiment_name(experiment)
        dataframes.append(df)
    return dataframes


def compute_mean_variance(dfs):
    stacked = pd.concat(dfs, axis=0).groupby(level=0)
    mean_df = stacked.mean()
    variance_df = stacked.var(ddof=0)
    return mean_df, variance_df


def plot(data, origin):
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


if __name__ == '__main__':
    
    origin = [0.1, 3.1, 6.1, 9.1, 12.1, 15.1, 18.1, 21.1, 24.1, 27.1, 30.1, 33.1, 36.1, 39.1, 42.1, 45.1, 48.1]
    experiments = ['cutting-classic-vmc', 'cutting-field-vmc-fixed-leader']
    for o in origin:
        dataframes = {}
        for experiment in experiments:
            data = load_data_from_csv(f'data/{experiment}', experiment, o)
            mean, variance = compute_mean_variance(data)
            dataframes[beautify_experiment_name(experiment)] = (mean, variance)
        plot(dataframes, o)