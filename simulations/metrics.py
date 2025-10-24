import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from asreview.metrics import loss
from asreview.metrics import ndcg
from asreviewcontrib.insights import algorithms
from asreviewcontrib.insights import metrics

def pad_labels(labels, num_priors, num_records):
    return pd.Series(
        labels.tolist() + np.zeros(num_records - len(labels) - num_priors).tolist()
    )
    
def tdd_at(results, threshold):
    all_tdd = metrics._time_to_discovery(results['record_id'], results['label'])
    count = sum(iter_idx <= threshold for _, iter_idx in all_tdd)
    return all_tdd, count


def evaluate_simulation(simulation_results, datasets, datasets_llms, minimal_prior_idx, out_dir, tdd_threshold = 100, threshold = .95):  


    padded_labels_minimal = pad_labels(simulation_results['minimal']["label"].reset_index(drop=True), len(minimal_prior_idx), len(datasets))
    padded_labels_llm = pad_labels(simulation_results['llm']["label"].reset_index(drop=True), len(datasets_llms['prior_idx']), len(datasets_llms['dataset']))
    padded_labels_no_priors = pad_labels(simulation_results['no_priors']["label"].reset_index(drop=True), 0, len(datasets))
    
    # concatenate the three cumulative sum results in one dataframe for adding metadata and plotting
    df_cumsum = pd.DataFrame({
        'Minimal Priors': padded_labels_minimal.cumsum(),
        'LLM Priors': padded_labels_llm.cumsum(),
        'No Priors': padded_labels_no_priors.cumsum()
    })

    # save cumulative sum dataframe to output_path
    df_cumsum_path = out_dir / dataset_names / 'cumsum_results.csv'
    df_cumsum.to_csv(df_cumsum_path, index=False)
    
    # generate a plot with scaled cumulative sum (divided by total number of relevant records in dataset)
    plt.figure(figsize=(10, 6))
    plt.step(np.arange(1, len(df_cumsum['Minimal Priors'].dropna())+1) / len(df_cumsum['Minimal Priors'].dropna()), (df_cumsum['Minimal Priors'] / (datasets[dataset_names]['label_included'].sum() - datasets[dataset_names]['label_included'].iloc[minimal_prior_idx[i]].sum())).dropna(), label='Minimal Priors', color='blue', where='post')
    plt.step(np.arange(1, len(df_cumsum['LLM Priors'].dropna())+1) / len(df_cumsum['LLM Priors'].dropna()), (df_cumsum['LLM Priors'] / datasets[dataset_names]['label_included'].sum()).dropna(), label='LLM Priors', color='green', where='post')
    plt.step(np.arange(1, len(df_cumsum['No Priors'].dropna())+1) / len(df_cumsum['No Priors'].dropna()), (df_cumsum['No Priors'] / (datasets[dataset_names]['label_included'].sum())).dropna(), label='No Priors', color='red', where='post')
    plt.xlabel('Proportion of Records Screened')
    plt.ylabel('Proportion of Relevant Records Found')
    plt.title('Proportion of Relevant Records Found vs. Proportion of Records Screened')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # save plot to output_path
    plot_path = out_dir / dataset_names / 'recall_plot.png'
    plt.savefig(plot_path)
    plt.close()
    
    nrr_minimal = loss(list(padded_labels_minimal))
    nrr_llm = loss(list(padded_labels_llm))
    nrr_no_priors = loss(list(padded_labels_no_priors))
    
    ndcg_score_minimal =  ndcg(list(padded_labels_minimal))
    ndcg_score_llm = ndcg(list(padded_labels_llm))
    ndcg_score_no_priors = ndcg(list(padded_labels_no_priors))

    td_minimal = tdd_at(simulation_results[dataset_names]['minimal'], tdd_threshold)[1]
    td_llm = tdd_at(simulation_results[dataset_names]['llm'], tdd_threshold)[1]
    td_no_priors = tdd_at(simulation_results[dataset_names]['no_priors'], tdd_threshold)[1]
    
    atd_minimal = metrics._average_time_to_discovery(tdd_at(simulation_results[dataset_names]['minimal'], tdd_threshold)[0])
    atd_llm = metrics._average_time_to_discovery(tdd_at(simulation_results[dataset_names]['llm'], tdd_threshold)[0])
    atd_no_priors = metrics._average_time_to_discovery(tdd_at(simulation_results[dataset_names]['no_priors'], tdd_threshold)[0])

    all_wss_minimal = algorithms._wss_values(padded_labels_minimal)
    all_wss_llm = algorithms._wss_values(padded_labels_llm)
    all_wss_no_priors = algorithms._wss_values(padded_labels_no_priors)

    idx_minimal = np.searchsorted(all_wss_minimal[0], threshold, side="right") - 1
    idx_minimal = max(idx_minimal, 0)

    idx_llm = np.searchsorted(all_wss_llm[0], threshold, side="right") - 1
    idx_llm = max(idx_llm, 0)

    idx_no_priors = np.searchsorted(all_wss_no_priors[0], threshold, side="right") - 1
    idx_no_priors = max(idx_no_priors, 0)
    
    metric_names = ["nrr", "ndcg", "td", "atd", "wss"]

    df_metrics = (
        pd.DataFrame({
            "metric": metric_names,
            'Minimal Priors': [nrr_minimal, ndcg_score_minimal, td_minimal, atd_minimal, all_wss_minimal[1][idx_minimal]],
            'LLM Priors': [nrr_llm, ndcg_score_llm, td_llm, atd_llm, all_wss_llm[1][idx_llm]],
            'No Priors': [nrr_no_priors, ndcg_score_no_priors, td_no_priors, atd_no_priors, all_wss_no_priors[1][idx_no_priors]]
        })
        .set_index("metric")
    )

    # save metrics dataframe to output_path
    df_metrics.to_csv(out_dir / dataset_names / "simulation_metrics.csv", index=False)
        
    return 
