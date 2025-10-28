import pandas as pd
import numpy as np
from pathlib import Path

from stimulus import select_criteria
from prompting import generate_abstracts


### Prepare the llm datasets ###

def prepare_llm_datasets(dataset: pd.DataFrame, name: str, criterium: list, out_dir: Path, metadata: pd.ExcelFile, n_abstracts: int, length_abstracts: int, typicality: float, degree_jargon: float, llm_temperature: float, run: int) -> dict:

    stimuli = select_criteria(name, criterium, metadata)
    
    # Skip if no stimuli found (dataset missing from metadata)
    if not stimuli:
        return None

    generated_abstracts = generate_abstracts(name=name, stimulus=stimuli, out_dir=out_dir, n_abstracts=n_abstracts, length_abstracts=length_abstracts, typicality=typicality, degree_jargon=degree_jargon, llm_temperature=llm_temperature, run=run)
             
    # concatenate original dataset with generated abstracts for simulation
    dataset_llm = pd.concat([dataset, generated_abstracts.drop(columns=['reasoning'])], ignore_index=True) # concatenate original dataset with generated abstracts
    llm_prior_idx = np.array(range(len(dataset), len(dataset_llm))) # get indices of generated abstracts to use as priors
    
    #store dataset with llm priors and prior indices in dictionary
    datasets_llms = {
        'dataset': dataset_llm,
        'prior_idx': llm_prior_idx
    }
        
    return datasets_llms