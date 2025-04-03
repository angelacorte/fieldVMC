import subproces

if __name__ == '__main__':
    
    self_repair_script = "plot-self-repair.py"
    self_integration_script = "plot-self-integration.py"
    generic_plotter_script = "plot.py"
    
    process1 = subprocess.Popen()(["python3", self_repair_script])
    process2 = subprocess.Popen()(["python3", self_integration_script])
    process3 = subprocess.Popen()(["python3", generic_plotter_script])
    
    print("All chart generator scripts have been launched.")
    