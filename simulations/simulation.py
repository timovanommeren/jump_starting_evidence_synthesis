from pathlib import Path
import pandas as pd

import asreview
from asreview.models.balancers import Balanced
from asreview.models.classifiers import SVM
from asreview.models.feature_extractors import Tfidf
from asreview.models.queriers import TopDown, Max
from asreview.models.stoppers import IsFittable
from asreview.models.stoppers import NLabeled

from minimal import sample_minimal_priors
from llm import prepare_llm_datasets
from metrics import evaluate_simulation

def run_simulation(datasets, criterium: list, out_dir: Path, metadata: pd.ExcelFile, n_abstracts: int, stop_at_n = None) -> dict:

    ### PREPARE SIMULATION ###################################################################################
    
    
    ### Generate abstracts and add them to datasets ###
    datasets_llms = prepare_llm_datasets(datasets, criterium=criterium, out_dir=out_dir, metadata=metadata, n_abstracts=n_abstracts)
    ############################################################################################################

    simulation_results = {}

    for i, dataset_names in enumerate(datasets.keys()):
        
        print(f"Running simulation for dataset: {dataset_names}")
        
        ### PREPARE SIMULATION DATA ###################################################################################
        
        # sample minimal training set priors
        minimal_prior_idx = sample_minimal_priors(datasets[dataset_names])

        ### SET UP ACTIVE LEARNING CYCLES #############################################################################

        tfidf_kwargs = {
        "ngram_range": (1, 2),
        "sublinear_tf": True,
        "max_df": 0.95,
        "min_df": 1,
        }

        alc_no_prior = [
            asreview.ActiveLearningCycle(querier=TopDown(), stopper=IsFittable()),
            asreview.ActiveLearningCycle(
                querier=Max(),
                classifier=SVM(C=0.11, loss="squared_hinge"),
                balancer=Balanced(ratio=9.8),
                feature_extractor=Tfidf(**tfidf_kwargs),
            )
        ]

        alc = [
            asreview.ActiveLearningCycle(
                querier=Max(),
                classifier=SVM(C=0.11, loss="squared_hinge"),
                balancer=Balanced(ratio=9.8),
                feature_extractor=Tfidf(**tfidf_kwargs),
            )
        ]
        
        ### RUN SIMULATION #########################################################################################

        # Run simulation with minimal priors
        simulate_minimal = asreview.Simulate(X=datasets[dataset_names], labels=datasets[dataset_names]["label_included"], cycles=alc, stopper=stop_at_n)
        simulate_minimal.label(minimal_prior_idx)
        simulate_minimal.review()

        # # Run simulation with LLM priors
        simulate_llm = asreview.Simulate(X=datasets_llms[dataset_names]['dataset'], labels=datasets_llms[dataset_names]['dataset']["label_included"], cycles=alc, stopper=stop_at_n)
        simulate_llm.label(datasets_llms[dataset_names]['prior_idx'])
        simulate_llm.review()

        # # Run simulation without priors (random start)
        simulate_no_priors = asreview.Simulate(X=datasets[dataset_names], labels=datasets[dataset_names]["label_included"], cycles=alc_no_prior, stopper=stop_at_n)
        simulate_no_priors.review()
        
        # This line drops priors. To access the dataframe before this, just use simulate._results
        df_results_minimal = simulate_minimal._results.dropna(axis=0, subset="training_set")
        df_results_llm = simulate_llm._results.dropna(axis=0, subset="training_set")
        df_results_no_priors = simulate_no_priors._results.dropna(axis=0, subset="training_set")


        simulation_results[dataset_names] = {
            'minimal': df_results_minimal,
            'llm': df_results_llm,
            'no_priors': df_results_no_priors
        }
        
        print(f"Evaluating simulation results: {simulation_results}")
        
        evaluate_simulation(simulation_results, datasets[dataset_names], datasets_llms[dataset_names], minimal_prior_idx, out_dir, tdd_threshold=100, threshold=.95)

    return simulation_results