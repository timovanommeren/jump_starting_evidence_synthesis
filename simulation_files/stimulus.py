import pandas as pd

### Select metadata criteria to use as stimulus for llm or as priors directly ###

def select_criteria(name: str, criterium: list, metadata: pd.ExcelFile):

    #create dictionary with the criteria depending on the selected criterium
    stimulus = {}
    
    # Check if dataset exists first
    indexed_metadata = metadata.set_index("dataset_ID")
    if name not in indexed_metadata.index:
        print(f"Warning: Dataset '{name}' not found in metadata.")
        return None
    
    for criteria in criterium:
        
        try:
            indexed_metadata.loc[name, criteria]
        except KeyError:
            print(f"Warning: Criteria '{criteria}' not found for dataset '{name}'. Skipping.")
            continue
        stimulus[criteria] = indexed_metadata.loc[name, criteria]    
    
    return stimulus