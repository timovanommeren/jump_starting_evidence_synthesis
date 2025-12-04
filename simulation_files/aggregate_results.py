from pathlib import Path
import pandas as pd

from metrics import aggregate_recall_plots

# parameters for evaluation

stop_at_n = 100 

in_dir = Path(r"C:\\Users\\timov\\Desktop\\Utrecht\\Utrecht\\MSBBSS\\thesis_timo\\Synergy\\synergy_dataset")
out_dir = Path(r'C:\\Users\\timov\\Desktop\\Utrecht\\Utrecht\\MSBBSS\\thesis_timo\\simulation_results\\inclusion_only')

# import the paths of all files in the input directory
data_paths = [f for f in Path(in_dir).iterdir() if f.is_file()]

# import all the of the datasets
datasets = {file.stem: pd.read_csv(file) for file in data_paths}

for dataset in datasets:
    (out_dir / dataset).mkdir(parents=True, exist_ok=True)

aggregate_recall_plots(
    datasets=datasets, 
    out_dir=out_dir, 
    stop_at_n=stop_at_n
)