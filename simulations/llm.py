import pandas as pd
import numpy as np
import dspy
from dotenv import load_dotenv
from pathlib import Path

### Select metadata criteria to use of stimulus for llm ###

def select_criteria(datasets: dict, criterium: list, metadata: pd.ExcelFile) -> list:
    
    stimuli = []

    for dataset_name in datasets.keys():

        stimulus = metadata.set_index("dataset_ID").loc[dataset_name, criterium]
        stimuli.append(stimulus)

    return stimuli



### Create signature ###

load_dotenv()  # Load environment variables from .env file

lm = dspy.LM("openai/gpt-4o-mini", temperature=0.7)
dspy.configure(lm=lm)

class MakeAbstract(dspy.Signature):
    """Generate a fake abstract based on search terms and whether it should be included or not."""
    criteria: str = dspy.InputField(desc="The inclusion or exclusion criteria of the review")
    label_included: int = dspy.InputField(desc="1 if it would perfectly fit the review; 0 if it would be returned by the given search terms but not fit the review")
    nonce: str = dspy.InputField() 
    jsonl: str = dspy.OutputField(desc='One-line JSON object: {"doi":"None","title":"...","abstract":"...","label_included":"1/0","reasoning":"..."}')

make_abstract = dspy.ChainOfThought(MakeAbstract)



### Generate abstracts ###

def generate_abstracts(datasets: dict, stimulus: list, out_dir: Path, n_abstracts: int) -> list:

    generated_abstracts = []

    for i, dataset_names in enumerate(datasets.keys()):
        
        #print(f'Generating abstracts for dataset: {dataset_names}')
        
        df_generated = pd.DataFrame()

        # loop to generate multiple abstracts
        for i in range(n_abstracts):
            
            #generate included abstract
            included = make_abstract(
                criteria = stimulus[i]['inclusion_criteria'],
                label_included=1,
                nonce=f"run-{i}",
                extra_instructions=""
            ).jsonl

            #generate excluded abstract
            excluded = make_abstract(
                criteria = stimulus[i]['exclusion_criteria'],
                label_included=0,
                nonce=f"run-{i}",
                extra_instructions=""
            ).jsonl

            #combine included and excluded abstracts into one pandas dataframe
            data = [included, excluded]
            data_dicts = [eval(item) for item in data]
            df_generated = pd.concat([df_generated, pd.DataFrame(data_dicts)], ignore_index=True)
            df_generated['label_included'] = df_generated['label_included'].astype(int)
            
            #save generated abstracts to csv file
            df_generated.to_csv(out_dir / dataset_names / "llm_priors.csv", index=False)

        generated_abstracts.append(df_generated)

    return generated_abstracts


### Prepare the llm datasets ###

def prepare_llm_datasets(datasets: dict, criterium: list, out_dir: Path, metadata: pd.ExcelFile, n_abstracts: int) -> dict:

    stimuli = select_criteria(datasets, criterium, metadata)

    generated_abstracts = generate_abstracts(datasets, stimulus=stimuli, out_dir=out_dir, n_abstracts=n_abstracts)

    datasets_llms = {}
    for i, dataset_names in enumerate(datasets.keys()):  
             
        # concatenate original dataset with generated abstracts for simulation
        dataset_llm = pd.concat([datasets[dataset_names], generated_abstracts[i].drop(columns=['reasoning'])], ignore_index=True) # concatenate original dataset with generated abstracts
        llm_prior_idx = np.array(range(len(datasets[dataset_names]), len(dataset_llm))) # get indices of generated abstracts to use as priors
        
        #store dataset with llm priors and prior indices in dictionary
        datasets_llms[dataset_names] = {
            'dataset': dataset_llm,
            'prior_idx': llm_prior_idx
        }
        
    return datasets_llms