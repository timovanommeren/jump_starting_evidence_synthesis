@ echo off
COLOR E0

:: version 0.1.dev93+g272efe11b.d20251002

:: Create folder structure. By default, the folder 'output' is used to store output.
mkdir output
mkdir output\simulation
mkdir output\simulation\state_files mkdir output\simulation\metrics
mkdir output\figures
mkdir output\tables\metrics
mkdir output\tables\time_to_discovery

mkdir output\simulation\descriptives
python scripts\data_describe.py generated_data\dataset_custom_priors.csv -o output\simulation\descriptives\data_stats_dataset_custom_priors.json
python scripts\data_describe.py generated_data\dataset_without_priors.csv -o output\simulation\descriptives\data_stats_dataset_without_priors.json


  python -m asreview simulate generated_data\dataset_custom_priors.csv -o output\simulation\state_files\state_dataset_custom_priors__llm_priors.asreview  --n-stop 1000 --prior-idx 5214 5215 && python -m asreview metrics output\simulation\state_files\state_dataset_custom_priors__llm_priors.asreview --quiet -o output\simulation\metrics\dataset_custom_priors__llm_priors_metrics.csv

  python -m asreview simulate generated_data\dataset_without_priors.csv -o output\simulation\state_files\state_dataset_without_priors__minimal_priors.asreview  --n-stop 1000 --n-prior-included 1 --n-prior-excluded 1 && python -m asreview metrics output\simulation\state_files\state_dataset_without_priors__minimal_priors.asreview --quiet -o output\simulation\metrics\dataset_without_priors__minimal_priors_metrics.csv

  python -m asreview simulate generated_data\dataset_without_priors.csv -o output\simulation\state_files\state_dataset_without_priors__no_priors.asreview  --n-stop 1000 && python -m asreview metrics output\simulation\state_files\state_dataset_without_priors__no_priors.asreview --quiet -o output\simulation\metrics\dataset_without_priors__no_priors_metrics.csv

:: Generate plot and tables for dataset
python scripts\get_plot.py -s output\simulation\state_files\ -o output\figures\plot_recall_sim.png -l filename --hide-random
python scripts\merge_metrics.py -s output\simulation\metrics\ -o output\tables\metrics\metrics_sim.csv
python scripts\merge_tds.py -s output\simulation\metrics\ -o output\tables\time_to_discovery\tds_sim.csv
python scripts\merge_descriptives.py  -s output\simulation\descriptives\
