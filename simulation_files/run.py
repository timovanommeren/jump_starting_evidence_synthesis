import typer 
from pathlib import Path
import pandas as pd
import itertools


from simulation import run_simulation
from metrics import aggregate_recall_plots
from config import load_pyproject_config

app = typer.Typer()

@app.command()
def run(
    
    in_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True,
                                  help="Folder containing datasets."),
    out_dir: Path = typer.Argument(..., exists=False, file_okay=False, dir_okay=True, readable=True,
                                  help="Root folder for all outputs."),
    criteria_path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True,
                                  help="Path to criteria file for LLM."),
    #stimulus_for_llm: str = typer.Argument(..., help="Space-separated list of stimulus for LLM.")
):
  
    ### LOAD CONFIG FROM TOML FILE ##########################################################################
    
    config = load_pyproject_config()
    
    #from simulation import pad_labels
    stimulus_for_llm = config.get("stimulus_for_llm")

    # Parameters for running simulations
    n_simulations = config.get("n_simulations")
    stop_at_n = config.get("stop_at_n") # set to -1 to stop when all relevant records are found
    
    # Parameters for simulation (IVs)
    n_abstracts = config.get("n_abstracts")
    length_abstracts = config.get("length_abstracts")
    llm_temperature = config.get("llm_temperature")

    # Parameters for evaluation (DVs)
    papers_screened = stop_at_n if stop_at_n != -1 else None  


  
    ### RETRIEVE INPUT #######################################################################################
    
    # # Resolve in_dir relative to project root (parent of simulation_files/) !!! DOESNT WORK YET
    # script_dir = Path(__file__).parent
    # project_root = script_dir.parent
    # in_dir = project_root / in_dir
    
    # import the paths of all files in the input directory
    data_paths = [f for f in Path(in_dir).iterdir() if f.is_file()]
    
    # import all the of the datasets
    datasets = {file.stem: pd.read_csv(file) for file in data_paths}

    ### Create smaller subset of datasets for testing ####################################################
    subset_keys = config.get("subset_datasets", None)
    if subset_keys is not None:
        datasets = {k: datasets[k] for k in subset_keys if k in datasets}
    
    print(f"Running simulations on datasets: {list(datasets.keys())}")

    ##########################################################################################################





    ### CREATE OUTPUT DIRECTORIES ############################################################################

    # create output directories for each dataset
    for dataset in datasets:
        (out_dir / dataset).mkdir(parents=True, exist_ok=True)
        

        
    # load synergy metadata (path relative to this script's location)
    synergy_metadata = pd.read_excel(criteria_path)

    #convert string of stimulus for llm to list
    #stimulus_for_llm = stimulus_for_llm.split(' ')

    ##########################################################################################################






    ### SIMULATE AND EVALUATE ###############################################################################
    
    # Generate all combinations of IVs
    iv_combinations = list(itertools.product(
        n_abstracts,
        length_abstracts,
        llm_temperature
    ))
    
    print(f"Running {n_simulations} simulations for each of {len(iv_combinations)} IV combinations for {len(datasets)} datasets for all four conditions")
    print(f"Total simulations: {n_simulations * len(iv_combinations) * len(datasets) * 4}")
    
    
    for run in range(n_simulations):
       
        for combo_idx, (n_abs, len_abs, temp) in enumerate(iv_combinations):
            print(f"\nIV Combination {combo_idx + 1}/{len(iv_combinations)}: "
                f"n_abstracts={n_abs}, length={len_abs}, temperature={temp}."
                f"From simulation {run + 1} of {n_simulations}.")
        
            run_simulation(
                datasets=datasets,
                criterium=stimulus_for_llm,
                out_dir=out_dir,
                metadata=synergy_metadata,
                n_abstracts=n_abs,
                length_abstracts=len_abs,
                llm_temperature=temp,
                papers_screened=papers_screened,
                run=run * len(iv_combinations) + combo_idx + 1,  # global run counter starting from 1
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

if __name__ == "__main__":
    app()