import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless, no Tk
import matplotlib.pyplot as plt


from asreview.metrics import loss
from asreview.metrics import ndcg
from asreviewcontrib.insights import algorithms
from asreviewcontrib.insights import metrics


def evaluate_simulation(simulation_results: dict, dataset: pd.DataFrame, dataset_llms: pd.DataFrame, dataset_criteria: pd.DataFrame, prior_idx: list, n_abstracts: int, length_abstracts: int, llm_temperature: float, papers_screened: int, out_dir: Path, run: int, stop_at_n: int) -> None:

    ### PREPARE DATA FOR EVALUATION ############################################################################################################

    (dataset_names, simulation_results), = simulation_results.items() # comma enforces unpacking single item (since were doing only one dataset at a time)

    # pad the labels to ensure accurate simulation results (see Report section 4.2.1)
    # padded_labels_random = pad_labels(simulation_results['random']["label"].reset_index(drop=True), len([prior_idx]), len(dataset), stop_at_n)
    # padded_labels_llm = pad_labels(simulation_results['llm']["label"].reset_index(drop=True), 0, len(dataset_llms['dataset']), stop_at_n) # note that in the llm condition, the priors aren't part of the analyzed set so should not be considered in padding
    # padded_labels_criteria = pad_labels(simulation_results['criteria']["label"].reset_index(drop=True), 0, len(dataset_criteria['dataset']), stop_at_n) # idem for criteria condition
    # padded_labels_no_initialisation = pad_labels(simulation_results['no_initialisation']["label"].reset_index(drop=True), 0, len(dataset), stop_at_n)

    # concatenate the three cumulative sum results in one dataframe for adding metadata and plottingq
    df_cumsum = pd.DataFrame({
        'Random Initialization': simulation_results['random']["label"].reset_index(drop=True).iloc[:100].cumsum(),
        'LLM Initialization': simulation_results['llm']["label"].reset_index(drop=True).iloc[:100].cumsum(),
        'Criteria Initialization': simulation_results['criteria']["label"].reset_index(drop=True).iloc[:100].cumsum(),
        'No Initialization': simulation_results['no_initialisation']["label"].reset_index(drop=True).iloc[:100].cumsum()
    })
    
    # Calculate the actual number of runs (retrievals) performed
    n_trials = len(simulation_results['random']["label"].reset_index(drop=True).iloc[:100])
    
    ############################################################################################################################################
    
    
    
    

    ### GENERATE PLOTS #########################################################################################################################
    
    recall_plot(
        df_cumsum=df_cumsum,
        dataset_names=dataset_names,
        n_abstracts=n_abstracts,
        length_abstracts=length_abstracts,
        llm_temperature=llm_temperature,
        out_dir=out_dir,
        run=run,
        stop_at_n=stop_at_n
    )
    
    ############################################################################################################################################
 
 
 
 
 
    #### CALCULATE OUTCOME METRICS ############################################################################################################

    # Calculate the number of relevant records found at TDD threshold (capped at 100 rows)
    td_random = tdd_at({'record_id': simulation_results['random']['record_id'].iloc[:100], 'label': simulation_results['random']['label'].iloc[:100]}, papers_screened)[1]
    td_llm = tdd_at({'record_id': simulation_results['llm']['record_id'].iloc[:100], 'label': simulation_results['llm']['label'].iloc[:100]}, papers_screened)[1]
    td_criteria = tdd_at({'record_id': simulation_results['criteria']['record_id'].iloc[:100], 'label': simulation_results['criteria']['label'].iloc[:100]}, papers_screened)[1]
    td_no_initialisation = tdd_at({'record_id': simulation_results['no_initialisation']['record_id'].iloc[:100], 'label': simulation_results['no_initialisation']['label'].iloc[:100]}, papers_screened)[1]

    # Calculate the number of records that need to be screened to find the first relevant record (ATD)
    atd_random = metrics._average_time_to_discovery(tdd_at({'record_id': simulation_results['random']['record_id'].iloc[:100], 'label': simulation_results['random']['label'].iloc[:100]}, papers_screened)[0])
    atd_llm = metrics._average_time_to_discovery(tdd_at({'record_id': simulation_results['llm']['record_id'].iloc[:100], 'label': simulation_results['llm']['label'].iloc[:100]}, papers_screened)[0])
    atd_criteria = metrics._average_time_to_discovery(tdd_at({'record_id': simulation_results['criteria']['record_id'].iloc[:100], 'label': simulation_results['criteria']['label'].iloc[:100]}, papers_screened)[0])
    atd_no_initialisation = metrics._average_time_to_discovery(tdd_at({'record_id': simulation_results['no_initialisation']['record_id'].iloc[:100], 'label': simulation_results['no_initialisation']['label'].iloc[:100]}, papers_screened)[0])

    ############################################################################################################################################





    ### SAVE METRICS TO MASTER RESULTS FILE ###################################################################################################

    results_row = []

    for condition, metrics_dict in [
        ('random', {'papers_found': td_random, 
                    'atd': atd_random}),
        ('llm', {'papers_found': td_llm,
                'atd': atd_llm}),
        ('criteria', {'papers_found': td_criteria,
                      'atd': atd_criteria}),
        ('no_initialisation', {'papers_found': td_no_initialisation,
                    'atd': atd_no_initialisation})
    ]:
        for metric_name, metric_value in metrics_dict.items():
            
            # Determine if parameters apply to this condition
            is_llm = (condition == 'llm')
            
            results_row.append({
                'dataset': dataset_names,
                'condition': condition,
                'metric': metric_name,
                'value': metric_value,
                'n_abstracts': n_abstracts if is_llm else np.nan,
                'length_abstracts': length_abstracts if is_llm else np.nan,
                'llm_temperature': llm_temperature if is_llm else np.nan,
                'tdd@': papers_screened,
                'timestamp': pd.Timestamp.now().isoformat(),
                'run': run,  # replicate ID
                'n_trials': n_trials,  # number of attempted retrievals
            })
            
    # Append to master results file
    df_results = pd.DataFrame(results_row)
    master_file = out_dir / 'all_simulation_results.csv'
    df_results.to_csv(master_file, mode='a', header=not master_file.exists(), index=False)
    
    ############################################################################################################################################











### HELPER FUNCTIONS ##############################################################################################################################################



def pad_labels(labels, num_priors, num_records, stop_at_n):
    
    # if there is a stopping criterion, then only pad until stop_at_n   
    if stop_at_n != -1: 
        
        #first check whether len(labels) is already >= stop_at_n (this may occur in no initialization condition, runs at least until the first relevant is found). If true truncate to stop_at_n
        if len(labels) >= stop_at_n:
            return pd.Series(labels.tolist()[:stop_at_n])
        else:
            return pd.Series(labels.tolist() + np.zeros(stop_at_n - len(labels)).tolist())
    
    else:
        return pd.Series(labels.tolist() + np.zeros(num_records - len(labels) - num_priors).tolist())


    
def tdd_at(results, threshold):
    all_tdd = metrics._time_to_discovery(results['record_id'], results['label'])
    count = sum(iter_idx <= threshold for _, iter_idx in all_tdd)
    return all_tdd, count



def recall_plot(df_cumsum: pd.DataFrame, dataset_names: str, n_abstracts: int, length_abstracts: int, llm_temperature: float, out_dir: Path, run: int, stop_at_n: int):
    
    plt.figure(figsize=(10, 6))

    # Use 1-based x-axis: screening 1 through stop_at_n
    x_axis = range(1, len(df_cumsum['Random Initialization']) + 1)
    plt.plot(x_axis, df_cumsum['Random Initialization'], label='Random Initialization', color='blue')
    plt.plot(x_axis, df_cumsum['LLM Initialization'], label='LLM Initialization', color='green')
    plt.plot(x_axis, df_cumsum['Criteria Initialization'], label='Criteria Initialization', color='orange')
    plt.plot(x_axis, df_cumsum['No Initialization'], label='No Initialization', color='red')

    # Add dashed line at stop_at_n
    if stop_at_n != -1:
        plt.axvline(x=stop_at_n, color='black', linestyle='--', label='stopped screening')

    plt.xlabel('Number of Records Screened')
    plt.ylabel('Number of Relevant Records Found')
    plt.title('Number of Relevant Records Found vs. Number of Records Screened')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    #create subfolder for recalls plots
    recalls_folder = out_dir / dataset_names / 'recalls_plots'
    recalls_folder.mkdir(parents=True, exist_ok=True)
    
    # save plot to output_path
    plot_path = recalls_folder / f'recall_plot_run_{run}_IVs_{n_abstracts}_{length_abstracts}_{llm_temperature}.png'
    plt.savefig(plot_path)
    plt.close()




### CREATE AGGREGATED RECALL PLOTS FUNCTION ##############################################################################################################################################

def aggregate_recall_plots(datasets: dict, out_dir: Path, stop_at_n: int) -> None:
  
  for name in datasets:
      
    raw_output = out_dir / name / 'raw_simulations'
        
    random_runs = []
    llm_runs = []
    criteria_runs = []
    no_initialisation_runs = []

    # loop over all csv files in raw_output
    for file in Path(raw_output).glob('*.csv'):
        df = pd.read_csv(file)

        #drop the first rows if it contains NaN
        df = df.dropna(axis=0, subset=["training_set"])
        
        # check if the file is random priors, llm or no priors
        if 'random' in file.name:
            random_runs.append(df)
        elif 'llm' in file.name:
            llm_runs.append(df)
        elif 'criteria' in file.name:
            criteria_runs.append(df)
        elif 'no_initialisation' in file.name:
            no_initialisation_runs.append(df)   
            
    # For each method, first calculate cumsum for each run, then compute mean and SEM across runs
    random_cumsums = [df.loc[df["querier"].notna(), "label"].iloc[:stop_at_n].cumsum().reset_index(drop=True) for df in random_runs]
    agg_random = pd.DataFrame({
        'Cumulative Sum': pd.concat(random_cumsums, axis=1).mean(axis=1),
        'SE': pd.concat(random_cumsums, axis=1).sem(axis=1)
    })

    llm_cumsums = [df.loc[df["querier"].notna(), "label"].iloc[:stop_at_n].cumsum().reset_index(drop=True) for df in llm_runs]
    agg_llm = pd.DataFrame({
        'Cumulative Sum': pd.concat(llm_cumsums, axis=1).mean(axis=1),
        'SE': pd.concat(llm_cumsums, axis=1).sem(axis=1)
    })
    
    criteria_cumsums = [df.loc[df["querier"].notna(), "label"].iloc[:stop_at_n].cumsum().reset_index(drop=True) for df in criteria_runs]
    agg_criteria = pd.DataFrame({
        'Cumulative Sum': pd.concat(criteria_cumsums, axis=1).mean(axis=1),
        'SE': pd.concat(criteria_cumsums, axis=1).sem(axis=1)
    })

    no_initialisation_cumsums = [df.loc[df["querier"].notna(), "label"].iloc[:stop_at_n].cumsum().reset_index(drop=True) for df in no_initialisation_runs]
    agg_no_initialisation = pd.DataFrame({
        'Cumulative Sum': pd.concat(no_initialisation_cumsums, axis=1).mean(axis=1),
        'SE': pd.concat(no_initialisation_cumsums, axis=1).sem(axis=1)
    })


    ### PLOT THE AGGREGATED RECALL CURVES ############################################################################################
    
    # Note that this function does not work if stop_at_n == -1 (no stopping criterion)
    
    plt.figure(figsize=(10, 6))

    x_axis = range(1, stop_at_n + 1)

    # Use 1-based x-axis: screening 1 through stop_at_n
    plt.plot(x_axis, agg_random['Cumulative Sum'][:stop_at_n], label='True Example Condition', color='blue')
    plt.fill_between(x_axis, agg_random['Cumulative Sum'][:stop_at_n] - agg_random['SE'][:stop_at_n], agg_random['Cumulative Sum'][:stop_at_n] + agg_random['SE'][:stop_at_n], color='royalblue', alpha=0.3)

    plt.plot(x_axis, agg_llm['Cumulative Sum'][:stop_at_n], label='LLM Condition', color='green')
    plt.fill_between(x_axis, agg_llm['Cumulative Sum'][:stop_at_n] - agg_llm['SE'][:stop_at_n], agg_llm['Cumulative Sum'][:stop_at_n] + agg_llm['SE'][:stop_at_n], color='forestgreen', alpha=0.3)
    
    plt.plot(x_axis, agg_criteria['Cumulative Sum'][:stop_at_n], label='Inclusion Criteria Condition', color='orange')
    plt.fill_between(x_axis, agg_criteria['Cumulative Sum'][:stop_at_n] - agg_criteria['SE'][:stop_at_n], agg_criteria['Cumulative Sum'][:stop_at_n] + agg_criteria['SE'][:stop_at_n], color='goldenrod', alpha=0.3)
    
    plt.plot(x_axis, agg_no_initialisation['Cumulative Sum'][:stop_at_n], label='Cold Start Condition', color='red')
    plt.fill_between(x_axis, agg_no_initialisation['Cumulative Sum'][:stop_at_n] - agg_no_initialisation['SE'][:stop_at_n], agg_no_initialisation['Cumulative Sum'][:stop_at_n] + agg_no_initialisation['SE'][:stop_at_n], color='lightsalmon', alpha=0.3)

    # Add dashed line at stop_at_n
    plt.axvline(x=stop_at_n, color='black', linestyle='--', label='stop screening')

    plt.xlabel('Number of Records Screened')
    plt.ylabel('Number of Relevant Records Found')
    plt.title('Number of Relevant Records Found vs. Number of Records Screened')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # save plot to output_path
    plot_path = out_dir / name / 'aggregate_recall_plot.png'
    plt.savefig(plot_path)
    plt.close()
    
    
    
