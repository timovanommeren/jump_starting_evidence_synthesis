import pandas as pd

def select_criteria(dataset_id: str, criterium: list, metadata) -> pd.Series:
    
    df = pd.read_excel(metadata) if not isinstance(metadata, pd.DataFrame) else metadata
    
    return df.set_index("dataset_ID").loc[dataset_id, criterium]
