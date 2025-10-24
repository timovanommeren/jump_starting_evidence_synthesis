import typer 
from pathlib import Path
import pandas as pd


from minimal import sample_minimal_priors
from llm import prepare_llm_datasets
from simulation import run_simulation
from metrics import evaluate_simulation

#from simulation import pad_labels
#stimulus_for_llm = ['inclusion_criteria', 'exclusion_criteria']

app = typer.Typer()

@app.command()
def run(
    
    in_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True,
                                  help="Folder containing datasets."),
    out_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True,
                                  help="Root folder for all outputs."),
    stimulus_for_llm: str = typer.Argument(..., help="Space-separated list of stimulus for LLM.")
):
  
  
    ### RETRIEVE INPUT #########################################################################################
    
    # import the paths of all files in the input directory
    data_paths = [f for f in Path(in_dir).iterdir() if f.is_file()]
    
    # import all the of the datasets
    datasets = {file.stem: pd.read_csv(file) for file in data_paths}

    # create output directories for each dataset
    for dataset in datasets:
        (out_dir / dataset).mkdir(parents=True, exist_ok=True)
        
    # load synergy metadata
    synergy_metadata = pd.read_excel(r'C:\Users\timov\Desktop\Utrecht\Utrecht\MSBBSS\thesis_timo\Synergy\synergy_dataset_overview.xlsx')

    #convert string of stimulus for llm to list
    stimulus_for_llm = stimulus_for_llm.split(' ')

    ##########################################################################################################



    ### Create smaller subset of datasets for testing ########################################################
    selected_keys = ['Brouwer_2019']
    subset_datasets = {k: datasets[k] for k in selected_keys if k in datasets}
    datasets = subset_datasets
    ##########################################################################################################



    ### SIMULATE AND EVALUATE #################################################################################
    simulation_results = run_simulation(datasets, criterium=stimulus_for_llm, out_dir=out_dir, metadata=synergy_metadata, n_abstracts=1, stop_at_n=None)

    print(simulation_results['Brouwer_2019']['llm'].head())

    #evaluate_simulation(simulation_results, datasets, datasets_llms, minimal_prior_idx, out_dir, tdd_threshold=100, threshold=.95)
    ############################################################################################################


    return  

@app.command()
def hello():
    print('hello')

    # output_path = os.path.join(os.getcwd(), f"output\{name_dataset}")
    # os.makedirs(out_dir, exist_ok=True)
    # print("Directory created at:", out_dir)

if __name__ == "__main__":
    app()