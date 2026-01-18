from pathlib import Path
import pandas as pd

import asreview
from asreview.models.balancers import Balanced
from asreview.models.classifiers import SVM
from asreview.models.feature_extractors import Tfidf
from asreview.models.queriers import Random, Max
from asreview.models.stoppers import IsFittable
from asreview.models.stoppers import NLabeled

from priors import sample_priors
from llm import prepare_datasets
from metrics import evaluate_simulation



def run_simulation(datasets: dict, criterium: list, out_dir: Path, metadata: pd.ExcelFile, n_abstracts: int, length_abstracts: int, llm_temperature: float, papers_screened: int, run: int, stop_at_n: int) -> dict:

    for dataset_names in datasets.keys():
        
        ### PREPARE SIMULATION DATA ###################################################################################
        
        # Clear dictionary for each dataset
        simulation_results = {}
        
        # Generate LLM priors and add them to dataset
        print(f"Generating LLM priors for dataset: {dataset_names}")
        dataset_llm, dataset_criteria = prepare_datasets(datasets[dataset_names], name=dataset_names, criterium=criterium, out_dir=out_dir, metadata=metadata, n_abstracts=n_abstracts, length_abstracts=length_abstracts, llm_temperature=llm_temperature, run=run) # Generate abstracts and add them to datasets

        # Sample priors for random initialization condition
        prior_idx = sample_priors(datasets[dataset_names], seed = run) 

        ###############################################################################################################
        
        
        
        

        ### SET UP ACTIVE LEARNING CYCLES #############################################################################

        tfidf_kwargs = {
        "ngram_range": (1, 2),
        "sublinear_tf": True,
        "max_df": 0.95,
        "min_df": 1,
        }

        alc = [
            asreview.ActiveLearningCycle(
                querier=Random(random_state=run), 
                stopper=IsFittable()),
            asreview.ActiveLearningCycle(
                querier=Max(),
                classifier=SVM(C=0.11, loss="squared_hinge", random_state=run),
                balancer=Balanced(ratio=9.8),
                feature_extractor=Tfidf(**tfidf_kwargs),
                stopper=NLabeled(stop_at_n + len(dataset_criteria['prior_idx']))
            )
        ]
        
        ###############################################################################################################
        
        
        
        
        
        
        ### RUN SIMULATION ############################################################################################

        print(f"Running simulations for dataset: {dataset_names}")
        
        # Run simulation with LLM priors
        simulate_llm = asreview.Simulate(X=dataset_llm['dataset'], labels=dataset_llm['dataset']["label_included"], cycles=alc)
        simulate_llm.label(dataset_llm['prior_idx'])
        simulate_llm.review()

        # Run simulation with criteria as priors
        simulate_criteria = asreview.Simulate(X=dataset_criteria['dataset'], labels=dataset_criteria['dataset']["label_included"], cycles=alc)
        simulate_criteria.label(dataset_criteria['prior_idx'])
        simulate_criteria.review()

        # Run simulation without priors (random start)
        simulate_no_initialisation = asreview.Simulate(X=datasets[dataset_names], labels=datasets[dataset_names]["label_included"], cycles=alc)
        simulate_no_initialisation.review()
        
        # Run simulation with random initialization (one relevant and one irrelevant prior)
        simulate_random = asreview.Simulate(X=datasets[dataset_names], labels=datasets[dataset_names]["label_included"], cycles=alc)
        simulate_random.label([prior_idx])
        simulate_random.review()
        
        ###############################################################################################################
    
    
    
        
        
        ### SAVE SIMULATION RESULTS ####################################################################################
        
        # Create raw_simulations directory if it doesn't exist
        raw_sim_dir = out_dir / dataset_names / 'raw_simulations'
        raw_sim_dir.mkdir(parents=True, exist_ok=True)
        
        #save all results to csv files
        for sim, condition in zip([simulate_random, simulate_llm, simulate_criteria, simulate_no_initialisation], ['random', 'llm', 'criteria', 'no_initialisation']):
            sim._results.to_csv(raw_sim_dir / f'{condition}_run_{run}_IVs_{n_abstracts}_{length_abstracts}_{llm_temperature}.csv', index=False)
        
        # This line drops priors. To access the dataframe before this, just use simulate._results
        df_results_random = simulate_random._results.dropna(axis=0, subset="training_set")
        df_results_llm = simulate_llm._results.dropna(axis=0, subset="training_set")
        df_results_criteria = simulate_criteria._results.dropna(axis=0, subset="training_set")
        df_results_no_initialisation = simulate_no_initialisation._results.dropna(axis=0, subset="training_set")

        simulation_results[dataset_names] = {
            'random': df_results_random,
            'llm': df_results_llm,
            'criteria': df_results_criteria,
            'no_initialisation': df_results_no_initialisation
        }
        
        #################################################################################################################
        
        
        
        
        
        ### EVALUATE SIMULATION RUN #####################################################################################
        
        evaluate_simulation(simulation_results, 
                            datasets[dataset_names], 
                            dataset_llm, 
                            dataset_criteria, 
                            prior_idx, 
                            n_abstracts=n_abstracts, 
                            length_abstracts=length_abstracts, 
                            llm_temperature=llm_temperature, 
                            papers_screened=papers_screened, 
                            out_dir=out_dir, 
                            run=run, 
                            stop_at_n=stop_at_n)

        #################################################################################################################
        
    return