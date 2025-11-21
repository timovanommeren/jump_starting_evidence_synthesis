import typer 
from pathlib import Path
import pandas as pd
import itertools


from simulation import run_simulation
from metrics import aggregate_recall_plots

#from simulation import pad_labels
#stimulus_for_llm = ['inclusion_criteria', 'exclusion_criteria']

# Parameters for running simulations
n_simulations = 10
stop_at_n = 100 # set to -1 to stop when all relevant records are found

# Parameters for simulation (IVs)
n_abstracts = [1, 3, 5, 7, 10]
length_abstracts = [200, 500, 1000]
typicality = [0.90]
degree_jargon = [0.10]
llm_temperature = [0.0, 0.4, 0.8]

# Parameters for evaluation (DVs)
tdd_threshold = stop_at_n if stop_at_n != -1 else 100
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
  
  
    ### RETRIEVE INPUT #######################################################################################
    
    # Resolve in_dir relative to project root (parent of simulation_files/) !!! DOESNT WORK YET
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    in_dir = project_root / in_dir
    
    # import the paths of all files in the input directory
    data_paths = [f for f in Path(in_dir).iterdir() if f.is_file()]
    
    # import all the of the datasets
    datasets = {file.stem: pd.read_csv(file) for file in data_paths}

    # # ### Create smaller subset of datasets for testing ####################################################
    # selected_keys = ['Brouwer_2019']
    # subset_datasets = {k: datasets[k] for k in selected_keys if k in datasets}
    # datasets = subset_datasets
    
    # print(f"Running simulations on datasets: {list(datasets.keys())}")

    ##########################################################################################################

    ### CREATE OUTPUT DIRECTORIES ############################################################################

    # create output directories for each dataset
    for dataset in datasets:
        (out_dir / dataset).mkdir(parents=True, exist_ok=True)
        

        
    # load synergy metadata (path relative to this script's location)
    script_dir = Path(__file__).parent
    synergy_path = script_dir.parent / 'Synergy' / 'synergy_dataset_overview.xlsx'
    synergy_metadata = pd.read_excel(synergy_path)

    #convert string of stimulus for llm to list
    stimulus_for_llm = stimulus_for_llm.split(' ')

    ##########################################################################################################







    ### SIMULATE AND EVALUATE ###############################################################################
    
    # Generate all combinations of IVs
    iv_combinations = list(itertools.product(
        n_abstracts,
        length_abstracts,
        typicality,
        degree_jargon,
        llm_temperature
    ))
    
    print(f"Running {n_simulations} simulations for each of {len(iv_combinations)} IV combinations")
    print(f"Total simulations: {n_simulations * len(iv_combinations)}")
    
    
    for sim in range(n_simulations):
       
        for combo_idx, (n_abs, len_abs, typ, jargon, temp) in enumerate(iv_combinations, 1):
            print(f"\nIV Combination {combo_idx}/{len(iv_combinations)}: "
                f"n_abstracts={n_abs}, length={len_abs}, typicality={typ}, "
                f"jargon={jargon}, temperature={temp}")
        
            run_simulation(
                datasets=datasets,
                criterium=stimulus_for_llm,
                out_dir=out_dir,
                metadata=synergy_metadata,
                n_abstracts=n_abs,
                length_abstracts=len_abs,
                typicality=typ,
                degree_jargon=jargon,
                llm_temperature=temp,
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