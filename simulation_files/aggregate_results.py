from pathlib import Path
import pandas as pd

from metrics import aggregate_recall_plots

# parameters for evaluation

stop_at_n = 100 

out_dir = Path(r'C:\\Users\\timov\\Desktop\\Utrecht\\Utrecht\\MSBBSS\\thesis_timo\\simulation_results\\correct_trials')

datasets = [p.name for p in out_dir.iterdir() if p.is_dir()]

print(datasets) 

aggregate_recall_plots(
    datasets=datasets, 
    out_dir=out_dir, 
    stop_at_n=stop_at_n
)

print("Aggregation complete.")