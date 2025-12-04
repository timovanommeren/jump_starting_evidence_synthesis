import numpy as np
from typing import Dict, List
import pandas as pd

def sample_priors(dataset: pd.DataFrame, seed: int) -> List[int]:

    indices_w1 = np.where(dataset['label_included'].to_numpy() == 1)[0]

    if len(indices_w1) == 0:
        raise ValueError("Need at least one row with label_included==1 to sample minimal priors.")

    # Create a local random number generator with the seed (doesn't affect global state)
    rng = np.random.default_rng(seed)

    return int(rng.choice(indices_w1))