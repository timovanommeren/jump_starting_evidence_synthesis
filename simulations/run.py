import typer 
import pathlib
from pathlib import Path
from typing import Optional, List
import os
import pandas as pd
import numpy as np

from stimulus import select_criteria
from llm import generate_abstracts

#from simulation import pad_labels
stimulus_for_llm = ['inclusion_criteria', 'exclusion_criteria']
synergy_metadata = pd.read_excel(r'C:\Users\timov\Desktop\Utrecht\Utrecht\MSBBSS\thesis_timo\Synergy\synergy_dataset_overview.xlsx')

app = typer.Typer()

@app.command()
def run(
    
    in_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True,
                                  help="Folder containing datasets."),
    out_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True,
                                  help="Root folder for all outputs.")  
    ):
    
    ### Retieve datasets and their names from given directory ###
    
    # import the paths of all files in the input directory
    data_paths = [f for f in Path(in_dir).iterdir() if f.is_file()]
    
    # import all the of the datasets
    datasets = {file.stem: pd.read_csv(file) for file in data_paths}

    # create output directories for each dataset
    for dataset in datasets:
        (out_dir / dataset).mkdir(parents=True, exist_ok=True)



    ### Retrieve llm stimulus ###
    
    stimulus = []
    
    #return list with the stimulus for the llm
    for dataset_names in datasets.keys():
        stimulus.append(select_criteria(dataset_names, stimulus_for_llm, synergy_metadata))
    
    print(stimulus[0]) 
    
    
    ### Generate abstracts ###
    
    #return series with a df of generated abstracts per dataset
    generated_abstracts = generate_abstracts(datasets, n_abstracts= 1, stimulus = stimulus) # -- A bit cumbersome to addd datasets and dataset names seperately, should perhaps be joined in a list or series at import

    print(f'print: {generated_abstracts[0]}')

    ### sample minimal training set
    
    
    
    
    ### add generated abstracts to datasets
    
    datasets_llms = []
    
    for i, dataset_names in enumerate(datasets.keys()):
        dataset_llm = pd.concat([datasets[dataset_names], generated_abstracts[i]], ignore_index=True) # concatenate original dataset with generated abstracts
        llm_prior_idx = np.array(range(len(datasets[dataset_names]), len(dataset_llm))) # get indices of generated abstracts to use as priors
        datasets_llms.append({'dataset': dataset_llm, 'prior_index': llm_prior_idx}) # store dataset with llm priors and prior indices in list

    # save dataset with LLM priors to file to output_path and with dataset specific name
    for dataset in datasets_llms:
        dataset_llm_path = out_dir / Path(dataset["dataset"].name) / "llm_priors.csv"
        dataset["dataset"].to_csv(dataset_llm_path, index=False)
        print(f'Dataset with LLM priors saved to: {dataset_llm_path}')
        
        
        
    ### simulate data!
    
 
    

    return  

@app.command()
def hello():
    print('hello')

    # output_path = os.path.join(os.getcwd(), f"output\{name_dataset}")
    # os.makedirs(out_dir, exist_ok=True)
    # print("Directory created at:", out_dir)

if __name__ == "__main__":
    app()