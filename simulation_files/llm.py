import pandas as pd
import numpy as np
from pathlib import Path

from stimulus import select_criteria
from prompting import generate_abstracts


### Prepare the llm datasets ###

def prepare_datasets(dataset: pd.DataFrame, name: str, criterium: list, out_dir: Path, metadata: pd.ExcelFile, n_abstracts: int, length_abstracts: int, llm_temperature: float, run: int) -> dict:

    ### RETRIEVE CRITERIA FROM METADATA ##########################################################

    stimuli = select_criteria(name, criterium, metadata)
    
    # Skip if no stimuli found (dataset missing from metadata)
    if not stimuli:
        return None


    ### GENERATE ABSTRACTS #################################################################

    generated_abstracts = generate_abstracts(name=name, stimulus=stimuli, out_dir=out_dir, n_abstracts=n_abstracts, length_abstracts=length_abstracts, llm_temperature=llm_temperature, run=run)
     
    # # Ensure exactly n_abstracts included and n_abstracts excluded (1:1 ratio)
    # # If not, regenerate up to max_retries times
    # max_retries = 10
    # retry_count = 0
    
    # n_included = generated_abstracts['label_included'].sum()
    # n_excluded = len(generated_abstracts) - n_included
    
    # while (n_included != n_abstracts or n_excluded != n_abstracts) and retry_count < max_retries:
    #     retry_count += 1
    #     print(f"Regenerating abstracts for dataset {name} (attempt {retry_count}/{max_retries}): got {n_included} included and {n_excluded} excluded, expected {n_abstracts} of each.")
    #     generated_abstracts = generate_abstracts(name=name, stimulus=stimuli, out_dir=out_dir, n_abstracts=n_abstracts, length_abstracts=length_abstracts, typicality=typicality, degree_jargon=degree_jargon, llm_temperature=llm_temperature, run=run)
    #     n_included = generated_abstracts['label_included'].sum()
    #     n_excluded = len(generated_abstracts) - n_included
    

    # if n_included != n_abstracts or n_excluded != n_abstracts:
    #     print(f"WARNING: Dataset {name} failed to generate correct ratio after {max_retries} attempts. Proceeding with {n_included} included and {n_excluded} excluded.")
 
 
    ### APPEND GENERATED ABSTRACTS TO DATASET ##########################################################
             
    # concatenate original dataset with generated abstracts for simulation
    dataset_llm = pd.concat([dataset, generated_abstracts.drop(columns=['reasoning'])], ignore_index=True) # concatenate original dataset with generated abstracts
    llm_prior_idx = np.array(range(len(dataset), len(dataset_llm))) # get indices of generated abstracts to use as priors
    
    #store dataset with llm priors and prior indices in dictionary
    datasets_llms = {
        'dataset': dataset_llm,
        'prior_idx': llm_prior_idx
    }
 

 

    ### APPEND CRITERIA TO DATASET (AS CONTROL CONDITION) ##########################################################   
    
    # Create dataframe with 2 prior rows: inclusion criteria (label=1) and exclusion criteria (label=0)
    included_row = {col: '' for col in dataset.columns}
    included_row['abstract'] = stimuli['inclusion_criteria']
    included_row['label_included'] = 1
    criteria_data = pd.DataFrame([included_row])

        
    # concatenate original dataset with criteria for simulation
    dataset_criteria = pd.concat([dataset, criteria_data], ignore_index=True) # concatenate original dataset with criteria
    criteria_idx = np.array(range(len(dataset), len(dataset_criteria))) # get indices of criteria to use as priors
    
    #store dataset with criteria and prior indices in dictionary
    datasets_criteria = {
        'dataset': dataset_criteria,
        'prior_idx': criteria_idx
    }
        
    return datasets_llms, datasets_criteria