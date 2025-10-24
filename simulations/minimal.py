import numpy as np
import random
from typing import Dict, List
import pandas as pd

def sample_minimal_priors(datasets: pd.DataFrame) -> List[List[int]]:

    minimal_prior_idx = []

    indices_w0 = np.where(datasets['label_included'].to_numpy() == 0)[0]
    indices_w1 = np.where(datasets['label_included'].to_numpy() == 1)[0]

    if len(indices_w0) == 0 or len(indices_w1) == 0:
        raise ValueError(f"Need at least one row with label_included==0 and one with ==1, for {dataset_names}")

    # select two random indices from indices_w0 and one from indices_w1
    i0 = random.sample(list(indices_w0), 1)
    i1 = random.sample(list(indices_w1), 1)

    #convert to int
    i0 = [int(i) for i in i0]
    i1 = [int(i) for i in i1]
    minimal_priors = [i0, i1]
    
    minimal_prior_idx = [item for sublist in minimal_priors for item in sublist] #flatten list of lists and append to all dataset list
        
    return minimal_prior_idx