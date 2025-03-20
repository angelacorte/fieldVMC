import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Path to the data folder
DATA_PATH = "data"

# Function to extract variable values from file names
def extract_variable(filename, variable_name):
    parts = filename.split("_")
    for part in parts:
        if part.startswith(variable_name + "-"):
            return part.split("-")[1]
    return None

# Function to get stabilization time (last row's time)
def get_stabilization_times(folder):
    times = []
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, file))
            last_time = df["time"].iloc[-1]
            times.append({"experiment": folder.split("/")[-1], "stabilization_time": last_time})
    return pd.DataFrame(times)

# **A1: Boxplot for Stabilization Time**
def plot_stabilization_time():
    exp1 = get_stabilization_times(os.path.join(DATA_PATH, "classic-vmc"))
    exp2 = get_stabilization_times(os.path.join(DATA_PATH, "field-vmc-fixed-leader"))
    df = pd.concat([exp1, exp2])

    plt.figure(figsize=(8, 6))
    sns.boxplot(x="experiment", y="stabilization_time", data=df)
    plt.title("Stabilization Time for A1 Experiments")
    plt.xlabel("Experiment")
    plt.ylabel("Stabilization Time")
    plt.show()

# **A2: Line Plot for Nodes Over Time**
def plot_nodes_over_time():
    experiments = ["cutting-classic-vmc", "cutting-field-vmc-fixed-leader"]
    plt.figure(figsize=(10, 6))

    for exp in experiments:
        folder = os.path.join(DATA_PATH, exp)
        for file in os.listdir(folder):
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(folder, file))
                origin = extract_variable(file, "origin")
                sns.lineplot(x=df["time"], y=df["nodes"], label=f"{exp} (origin {origin})", alpha=0.5)

    plt.title("Number of Nodes Over Time (A2)")
    plt.xlabel("Time")
    plt.ylabel("Number of Nodes")
    plt.legend()
    plt.show()

# **B1 & B2: Number of Roots Over Time with Vertical Line at 500**
def plot_roots_over_time(experiment):
    folder = os.path.join(DATA_PATH, experiment)
    plt.figure(figsize=(10, 6))

    for file in os.listdir(folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, file))
            initial_nodes = extract_variable(file, "initialNodes")
            sns.lineplot(x=df["time"], y=df["ifit1@leader[Sum]"], label=f"{experiment} (initialNodes {initial_nodes})", alpha=0.5)

    plt.axvline(x=500, color="red", linestyle="--", label="Time = 500")
    plt.title(f"Number of Roots Over Time ({experiment})")
    plt.xlabel("Time")
    plt.ylabel("Number of Roots")
    plt.legend()
    plt.show()

# **B3: Tracking Initial Nodes Over Time**
def plot_initial_nodes_over_time():
    experiment = "self-integration"  # Example
    folder = os.path.join(DATA_PATH, experiment)
    plt.figure(figsize=(10, 6))

    for file in os.listdir(folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, file))
            initial_nodes = extract_variable(file, "initialNodes")
            sns.lineplot(x=df["time"], y=[initial_nodes] * len(df), label=f"initialNodes {initial_nodes}", alpha=0.5)

    plt.title("Initial Nodes Over Time (B3)")
    plt.xlabel("Time")
    plt.ylabel("Initial Nodes")
    plt.legend()
    plt.show()

# Run all plots
plot_stabilization_time()
plot_nodes_over_time()
plot_roots_over_time("self-integration")  # B1
plot_roots_over_time("self-segmentation")  # B2
plot_roots_over_time("self-repair")  # B2
plot_initial_nodes_over_time()  # B3