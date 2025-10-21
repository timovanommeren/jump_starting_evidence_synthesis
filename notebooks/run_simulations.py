import typer 
import pathlib
from pathlib import Path
from typing import Optional, List
import os
import pandas as pd

from simulation import pad_labels

app = typer.Typer()

@app.command()
def run(
    
    in_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True,
                                  help="Folder containing datasets."),
    # out_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True,
    #                               help="Root folder for all outputs.")  
    ):
    
    files = [f for f in Path(in_dir).iterdir() if f.is_file()]

    dataset_names = []
    datasets = []
    for i in range(0, len(files)):
        dataset_names.append(pathlib.PurePath(files[i]).name)
        datasets.append(pd.read_csv(files[i]))
    
    print(datasets[0].head())

    return


    # output_path = os.path.join(os.getcwd(), f"output\{name_dataset}")
    # os.makedirs(out_dir, exist_ok=True)
    # print("Directory created at:", out_dir)

if __name__ == "__main__":
    app()