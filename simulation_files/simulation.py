from pathlib import Path
import pandas as pd
import pickle

import asreview
from asreview.models.balancers import Balanced
from asreview.models.classifiers import SVM
from asreview.models.feature_extractors import Tfidf
from asreview.models.queriers import Random, Max
from asreview.models.stoppers import IsFittable
from asreview.models.stoppers import NLabeled

from minimal import sample_minimal_priors
from llm import prepare_llm_datasets
from metrics import evaluate_simulation

def run_simulation(datasets: dict, criterium: list, out_dir: Path, metadata: pd.ExcelFile, n_abstracts: int, length_abstracts: int, typicality: float, degree_jargon: float, llm_temperature: float, tdd_threshold: int, wss_threshold: float, seed: int, run: int, stop_at_n = int) -> dict:

    ############################################################################################################

    for dataset_names in datasets.keys():
        
        ### PREPARE SIMULATION DATA ###################################################################################
        
        simulation_results = {} # clear dictionary for each dataset
        
        print(f"Generating LLM priors for dataset: {dataset_names}")
    
        dataset_llm = prepare_llm_datasets(datasets[dataset_names], name=dataset_names, criterium=criterium, out_dir=out_dir, metadata=metadata, n_abstracts=n_abstracts, length_abstracts=length_abstracts, typicality=typicality, degree_jargon=degree_jargon, llm_temperature=llm_temperature, run=run) # Generate abstracts and add them to datasets
  
        # Skip this dataset if metadata not found
        if dataset_llm is None:
            print(f"Skipping simulation for dataset '{dataset_names}' because no metadata was found.")
            continue
        
        minimal_prior_idx = sample_minimal_priors(datasets[dataset_names], seed=seed + run) # sample minimal training set priors

        ### SET UP ACTIVE LEARNING CYCLES #############################################################################

        tfidf_kwargs = {
        "ngram_range": (1, 2),
        "sublinear_tf": True,
        "max_df": 0.95,
        "min_df": 1,
        }

        alc_no_prior = [
            asreview.ActiveLearningCycle(
                querier=Random(random_state=seed + run), 
                stopper=IsFittable()),
            asreview.ActiveLearningCycle(
                querier=Max(),
                classifier=SVM(C=0.11, loss="squared_hinge", random_state=seed + run),
                balancer=Balanced(ratio=9.8),
                feature_extractor=Tfidf(**tfidf_kwargs),
                stopper=NLabeled(stop_at_n)
            )
        ]

        alc = [
            asreview.ActiveLearningCycle(
                querier=Max(),
                classifier=SVM(C=0.11, loss="squared_hinge", random_state=seed + run),
                balancer=Balanced(ratio=9.8),  
                feature_extractor=Tfidf(**tfidf_kwargs),
                stopper=NLabeled(stop_at_n if stop_at_n == -1 else stop_at_n + 2 * n_abstracts) # account for added LLM priors
            )
        ]
        
        ### RUN SIMULATION #########################################################################################

        print(f"Running simulation for dataset: {dataset_names}")

        # Run simulation with minimal priors
        simulate_minimal = asreview.Simulate(X=datasets[dataset_names], labels=datasets[dataset_names]["label_included"], cycles=alc)
        simulate_minimal.label(minimal_prior_idx)
        simulate_minimal.review()

        # # Run simulation with LLM priors
        simulate_llm = asreview.Simulate(X=dataset_llm['dataset'], labels=dataset_llm['dataset']["label_included"], cycles=alc)
        simulate_llm.label(dataset_llm['prior_idx'])
        simulate_llm.review()

        # # Run simulation without priors (random start)
        simulate_no_priors = asreview.Simulate(X=datasets[dataset_names], labels=datasets[dataset_names]["label_included"], cycles=alc_no_prior)
        simulate_no_priors.review()
        
        # Create raw_simulations directory if it doesn't exist
        raw_sim_dir = out_dir / dataset_names / 'raw_simulations'
        raw_sim_dir.mkdir(parents=True, exist_ok=True)
        
        #save all results to csv files
        for sim, condition in zip([simulate_minimal, simulate_llm, simulate_no_priors], ['minimal', 'llm', 'no_priors']):
            sim._results.to_csv(raw_sim_dir / f'{condition}_run_{run}_IVs_{n_abstracts}_{length_abstracts}_{typicality}_{degree_jargon}_{llm_temperature}.csv', index=False)
        
        # This line drops priors. To access the dataframe before this, just use simulate._results
        df_results_minimal = simulate_minimal._results.dropna(axis=0, subset="training_set")
        df_results_llm = simulate_llm._results.dropna(axis=0, subset="training_set")
        df_results_no_priors = simulate_no_priors._results.dropna(axis=0, subset="training_set")


        simulation_results[dataset_names] = {
            'minimal': df_results_minimal,
            'llm': df_results_llm,
            'no_priors': df_results_no_priors
        }
        
        # # Save each simulation so far using pickle
        # with open(out_dir / dataset_names / 'simulation_results.pkl', 'wb') as f:
        #     pickle.dump(simulation_results, f)
            
        # #load simulation results for evaluation
        # with open(out_dir / dataset_names / 'simulation_results.pkl', 'rb') as f:
        #     simulation_results = pickle.load(f)
        
        ### EVALUATE SIMULATION RUN #####################################################################################
        evaluate_simulation(simulation_results, datasets[dataset_names], dataset_llm, minimal_prior_idx, n_abstracts=n_abstracts, length_abstracts=length_abstracts, typicality=typicality, degree_jargon=degree_jargon, llm_temperature=llm_temperature, tdd_threshold=tdd_threshold, wss_threshold=wss_threshold, seed=seed + run, out_dir=out_dir, run=run, stop_at_n=stop_at_n)

    return