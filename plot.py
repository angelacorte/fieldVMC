import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import glob
import re
from datetime import datetime

directory = 'charts/good'

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
        return 'Field VMC Self-Optimisation'
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

def plot_selfs(data, experiment, metric, y_label='Number of roots', cut = True):
    i = len(data)+2
    plt.rcParams.update({'font.size': 15})
    colors = sns.color_palette("viridis", n_colors=i)
    plt.figure(figsize=(9, 5))
    for j, ((exp, nodes), (mean_df, variance_df)) in enumerate(data.items()):
        sns.lineplot(
            data = mean_df,
            x = 'time',
            y = metric,
            label = f'initial nodes: {nodes}',
            color = colors[j+2],
        )
        mean = mean_df[metric]
        variance = variance_df[metric]
        upper_bound = mean + np.sqrt(variance)
        lower_bound = mean - np.sqrt(variance)
        plt.fill_between(mean.index, lower_bound, upper_bound, color=colors[j+2], alpha=0.2)
    if cut:
        y_target = 2
        if experiment == beautify_experiment_name('self-integration'):
            plt.xlim(199, 210)
            plt.ylim(0, 5)
            y_target = 1
        plt.axhline(y=y_target, color=colors[1], linestyle='--', linewidth=2, label='Target')
        plt.axvline(x=200, color=colors[0], linestyle='dotted', linewidth=2, label='Event')

    plt.title(f'{experiment}')
    plt.xlabel('Simulated seconds')
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    now = datetime.now()
    now = now.strftime("%Y-%m-%d-%H-%M-%S")
    plt.savefig(f'{directory}/{experiment}{now}.pdf', dpi=300)
    plt.close()

def plot_selfs_twinx(data, experiment, metric, secondary_metric=None, y_label='Number of roots', secondary_y_label='Success per node', cut=True):
    i = (len(data) + 2)*2
    plt.rcParams.update({'font.size': 15})
    colors = sns.color_palette("viridis", n_colors=i)
    plt.figure(figsize=(20, 12))

    ax1 = plt.gca()
    for j, ((exp, nodes), (mean_df, variance_df)) in enumerate(data.items()):
        sns.lineplot(
            ax=ax1,
            data=mean_df,
            x='time',
            y=metric,
            label=f'initial nodes: {nodes}',
            color=colors[(j * 2) + 1],
        )
        mean = mean_df[metric]
        variance = variance_df[metric]
        upper_bound = mean + np.sqrt(variance)
        lower_bound = mean - np.sqrt(variance)
        ax1.fill_between(mean.index, lower_bound, upper_bound, color=colors[(j * 2) + 1], alpha=0.2)

    ax1.set_xlabel('Simulated seconds')
    ax1.set_ylabel(y_label)
    ax1.set_title(f'{experiment}')

    if secondary_metric:
        ax2 = ax1.twinx()
        for j, ((exp, nodes), (mean_df, variance_df)) in enumerate(data.items()):
            mean_df['success_per_node'] = mean_df[secondary_metric] / mean_df['nodes']
            sns.lineplot(
                ax=ax2,
                data=mean_df,
                x='time',
                y=secondary_metric,
                label=f'Success per node - nodes: {nodes}',
                color=colors[(j * 2) + 2],
            )
        ax2.set_ylabel(secondary_y_label)

    if cut:
        ax1.set_ylim(0, 15)
        if experiment == beautify_experiment_name('self-integration'):
            ax1.set_xlim(-10, 300)
            ax1.axhline(y=2, color=colors[1], linestyle='--', linewidth=1, label='Target')
        ax1.axvline(x=200, color=colors[0], linestyle='dotted', linewidth=1, label='Event')

    ax1.legend(loc='upper left')
    if secondary_metric:
        ax2.legend(loc='upper right')

    plt.tight_layout()
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    plt.savefig(f'{directory}/{experiment}WithSuccessPerNode{now}.pdf', dpi=300)
    plt.close()


def plot_single_selfs(data, experiment, metric, y_label='Number of roots', cut = True):
    i = len(data)+2
    plt.rcParams.update({'font.size': 13})
    colors = sns.color_palette("viridis", n_colors=i)
    plt.figure(figsize=(20, 12))
    for j, ((exp, nodes), (mean_df, variance_df)) in enumerate(data.items()):
        sns.lineplot(
            data = mean_df,
            x = 'time',
            y = metric,
            label = f'initial nodes: {nodes}',
            color = colors[j+2],
        )
        mean = mean_df[metric]
        variance = variance_df[metric]
        upper_bound = mean + np.sqrt(variance)
        lower_bound = mean - np.sqrt(variance)
        plt.fill_between(mean.index, lower_bound, upper_bound, color=colors[j+2], alpha=0.2)
        if cut:
            plt.ylim(0, 15)
            if experiment == 'self-integration':
                plt.xlim(-10, 300)
            plt.axhline(y=1, color=colors[1], linestyle='--', linewidth=1, label='Target')
        plt.axvline(x=200, color=colors[0], linestyle='dotted', linewidth=1, label='Event')
        plt.title(f'{experiment}')
        plt.xlabel('Simulated seconds')
        plt.ylabel(y_label)
        plt.legend()
        plt.tight_layout()
        now = datetime.now()
        now = now.strftime("%Y-%m-%d-%H-%M-%S")
        plt.savefig(f'{directory}/{experiment}_initialNodes-{nodes}-{now}.pdf', dpi=300)
        plt.close()

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

if __name__ == '__main__':

# =============================================================================
    # experiments = ['classic-vmc', 'field-vmc-fixed-leader']
    # metrics = ['network-hub-xCoord', 'network-hub-yCoord', 'nodes', 'network-diameter', 'network-density', 'nodes-degree[mean]']
    # dataframes = {beautify_experiment_name(experiment): [] for experiment in experiments}

    # for experiment in experiments:
    #     data = load_data_from_csv(f'data/{experiment}/{experiment}_*', experiment)
    #     times = []
    #     for seed in range(len(data)):
    #         metrics_to_check = {metric: [-1000] for metric in metrics}
    #         current_metrics_to_check = {metric: [-1000] for metric in metrics}
    #         equals_for = 0
    #         current_experiment_data = data[seed]
    #         index = check_stability(current_experiment_data, metrics, 30)
    #         time = current_experiment_data.loc[index, 'time']
    #         times.append(time)
    #     dataframes[beautify_experiment_name(experiment)].extend(times)
    # #box_plot(dataframes)
    # violin_plot(dataframes)

    # origin = [0.1, 9.1, 18.1, 24.1, 27.1]
    # experiments = ['cutting-classic-vmc', 'cutting-field-vmc-fixed-leader']
    # for o in origin:
    #     dataframes = {}
    #     for experiment in experiments:
    #         data = load_data_from_csv(f'data/{experiment}/{experiment}_origin-{o}_*', experiment)
    #         mean, variance = compute_mean_variance(data)
    #         dataframes[beautify_experiment_name(experiment)] = (mean, variance)
    #     plot_cutting(dataframes, o)

    # initialNodes = [100, 300, 500]
    # experiments = [ 'self-integration'] #, 'self-integration'
    # for experiment in experiments:
    #     dataframes = {}
    #     for n in initialNodes:
    #         data = load_data_from_csv(f'data/{experiment}/{experiment}_*_initialNodes-{n}.csv', experiment)
    #         mean, variance = compute_mean_variance(data)
    #         mean['time'] = mean['time'].astype(int)
    #         exp_name = beautify_experiment_name(experiment)
    #         dataframes[exp_name, n * 2] = (mean, variance)
    #     plot_selfs(dataframes, exp_name, metric = 'ifit1@leader[Sum]', y_label='Number of roots', cut = True)
# =============================================================================

    initialNodes = [1.0, 10, 100, 300, 500, 1000]
    experiment = 'self-optimization'
    dataframes = {}
    for n in initialNodes:
        data = load_data_from_csv(f'data/{experiment}/{experiment}_*_initialNodes-{n}.csv', experiment)
        mean, variance = compute_mean_variance(data)
        if n == 1.0:
            n = 1
        exp_name = beautify_experiment_name(experiment)
        dataframes[exp_name, n] = (mean, variance)
    #plot_selfs(dataframes, exp_name, 'nodes', 'Number of nodes', False)
    plot_selfs(dataframes, exp_name, 'success[mean]', 'Average maximum success', False)
    #plot_selfs_twinx(dataframes, exp_name, 'nodes', 'success[max]', 'Number of nodes', 'Average success', False)
    #plot_single_selfs(dataframes, exp_name, 'nodes', 'Number of nodes', False)
