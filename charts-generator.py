import subprocess
import os

if __name__ == '__main__':

    self_division_script = "plot-self-division.py"
    self_integration_script = "plot-self-integration.py"
    generic_plotter_script = "plot.py"
    message_script = "plot-message-size.py"

    output_directory = "charts"
    os.makedirs(output_directory, exist_ok=True)

    process1 = subprocess.Popen(["python3", self_division_script])
    print("Launched self-division plotter script.")
    process2 = subprocess.Popen(["python3", self_integration_script])
    print("Launched self-integration plotter script.")
    process3 = subprocess.Popen(["python3", generic_plotter_script])
    print("Launched generic plotter script.")
    process4 = subprocess.Popen(["python3", message_script, "--experiment", "leader-election"])
    print("Launched message size plotter script for leader-election.")
    process5 = subprocess.Popen(["python3", message_script, "--experiment", "fixed-leader"])
    print("Launched message size plotter script for fixed-leader.")

    print("All chart generator scripts have been launched.")
