import typer 
from pathlib import Path
import pandas as pd


from simulation import run_simulation
from metrics import aggregate_recall_plots

#from simulation import pad_labels
#stimulus_for_llm = ['inclusion_criteria', 'exclusion_criteria']

# Parameters for running simulations
n_simulations = 50
stop_at_n = 100 # set to -1 to stop when all relevant records are found

# Parameters for simulation (IVs)
n_abstracts = 1
length_abstracts = 500
typicality = .90
degree_jargon = .10
llm_temperature = .7

# Parameters for evaluation (DVs)
tdd_threshold = 100
wss_threshold = .95

# Parameters for reproducibility
seed = 42

app = typer.Typer()

@app.command()
def run(
    
    in_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True,
                                  help="Folder containing datasets."),
    out_dir: Path = typer.Argument(..., exists=False, file_okay=False, dir_okay=True, readable=True,
                                  help="Root folder for all outputs."),
    stimulus_for_llm: str = typer.Argument(..., help="Space-separated list of stimulus for LLM.")
):
  
  
    ### RETRIEVE INPUT #########################################################################################
    
    # import the paths of all files in the input directory
    data_paths = [f for f in Path(in_dir).iterdir() if f.is_file()]
    
    # import all the of the datasets
    datasets = {file.stem: pd.read_csv(file) for file in data_paths}

    # ### Create smaller subset of datasets for testing ########################################################
    # selected_keys = ['Walker_2018', 'Wassenaar_2017', 'Hall_2012', 'Brouwer_2019', 'Moran_2021', 'Leenaars_2020', 'Muthu_2021']
    # subset_datasets = {k: datasets[k] for k in selected_keys if k in datasets}
    # datasets = subset_datasets
    
    # print(f"Running simulations on datasets: {list(datasets.keys())}")

    ##########################################################################################################

    ### CREATE OUTPUT DIRECTORIES #############################################################################

    # create output directories for each dataset
    for dataset in datasets:
        (out_dir / dataset).mkdir(parents=True, exist_ok=True)
        

        
    # load synergy metadata
    synergy_metadata = pd.read_excel(r'C:\Users\timov\Desktop\Utrecht\Utrecht\MSBBSS\thesis_timo\Synergy\synergy_dataset_overview.xlsx')

    #convert string of stimulus for llm to list
    stimulus_for_llm = stimulus_for_llm.split(' ')

    ##########################################################################################################







    ### SIMULATE AND EVALUATE #################################################################################
    
    for sim in range(n_simulations):
        run_simulation(
            datasets=datasets,
            criterium=stimulus_for_llm,
            out_dir=out_dir,
            metadata=synergy_metadata,
            n_abstracts=n_abstracts,
            length_abstracts=length_abstracts,
            typicality=typicality,
            degree_jargon=degree_jargon,
            llm_temperature=llm_temperature,
            tdd_threshold=tdd_threshold,
            wss_threshold=wss_threshold,
            seed=seed,
            run=sim + 1,  # 1-based replicate index
            stop_at_n=stop_at_n
        )

    ############################################################################################################


    ### RETURN AGGREGATE RECALL PLOTS ##########################################################################
    
    aggregate_recall_plots(
        datasets=datasets, 
        out_dir=out_dir, 
        stop_at_n=stop_at_n
    )
    
    ############################################################################################################

    return  

@app.command()
def hello():
    print('hello')

if __name__ == "__main__":
    app()