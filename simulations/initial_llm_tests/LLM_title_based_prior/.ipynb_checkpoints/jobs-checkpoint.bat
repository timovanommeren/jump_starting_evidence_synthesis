@ echo off
COLOR E0

:: version 1.1

:: Create folder structure. By default, the folder 'output' is used to store output.
mkdir output
mkdir output\simulation
mkdir output\simulation\metrics
mkdir output\figures
mkdir output\tables\metrics
mkdir output\tables\time_to_discovery

mkdir output\simulation\descriptives
python scripts\data_describe.py generated_data\dataset_custom_priors.csv -o output\simulation\descriptives\data_stats_dataset_custom_priors.json
python scripts\data_describe.py generated_data\dataset_without_priors.csv -o output\simulation\descriptives\data_stats_dataset_without_priors.json

python -m asreview simulate generated_data\dataset_custom_priors.csv -o output\simulation\state_files\sim_custom_priors.asreview --seed 165 --prior-idx 258 259
python -m asreview metrics output\simulation\state_files\sim_custom_priors.asreview -o output\simulation\metrics\metrics_sim_custom_priors.json --quiet

python -m asreview simulate generated_data\dataset_without_priors.csv -o output\simulation\state_files\sim_minimal_priors.asreview --prior-seed 535 --seed 165
python -m asreview metrics output\simulation\state_files\sim_minimal_priors.asreview -o output\simulation\metrics\metrics_sim_minimal_priors.json --quiet

:: Generate plot and tables for dataset
python scripts\get_plot.py -s output\simulation\state_files\ -o output\figures\plot_recall_sim.png -l filename --hide-random
python scripts\merge_metrics.py -s output\simulation\metrics\ -o output\tables\metrics\metrics_sim.csv
python scripts\merge_tds.py -s output\simulation\metrics\ -o output\tables\time_to_discovery\tds_sim.csv
python scripts\merge_descriptives.py  -s output\simulation\descriptives\
