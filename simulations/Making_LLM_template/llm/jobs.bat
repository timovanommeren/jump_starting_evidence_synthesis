@ echo off
COLOR E0

:: version 0.1.dev103+gac2e4ab9d

:: Create folder structure. By default, the folder 'output' is used to store output.
mkdir output
mkdir output\simulation
mkdir output\tables
mkdir output\tables\metrics
mkdir output\tables\time_to_discovery
mkdir output\figures


::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:::::: DATASET: Sep_2021
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

:: Create output folder
mkdir output\simulation\Sep_2021_minimal_prior\
mkdir output\simulation\Sep_2021_minimal_prior\metrics

mkdir output\simulation\Sep_2021_no_prior\
mkdir output\simulation\Sep_2021_no_prior\metrics

mkdir output\simulation\Sep_2021_llm_prior\
mkdir output\simulation\Sep_2021_llm_prior\metrics

:: Collect descriptives about the dataset Sep_2021
mkdir output\simulation\Sep_2021\descriptives
python scripts\data_describe.py data\Sep_2021.csv -o output\simulation\Sep_2021\descriptives\data_stats_Sep_2021.json

:: Simulate runs
mkdir output\simulation\Sep_2021_minimal_prior\state_files
mkdir output\simulation\Sep_2021_no_prior\state_files
mkdir output\simulation\Sep_2021_llm_prior\state_files
python -m asreview simulate data\Sep_2021.csv -o output\simulation\Sep_2021_minimal_prior\state_files\sim_Sep_2021.asreview --prior-seed 535 --seed 165 --prior-idx 72 19

python -m asreview simulate data\Sep_2021.csv -o output\simulation\Sep_2021_no_prior\state_files\sim_Sep_2021.asreview --prior-seed 535 --seed 165
python -m asreview simulate data\Sep_2021.csv -o output\simulation\Sep_2021_llm_prior\state_files\sim_Sep_2021.asreview --prior-seed 535 --seed 165 --prior-idx 283 284

python -m asreview metrics output\simulation\Sep_2021_minimal_prior\state_files\sim_Sep_2021.asreview -o output\simulation\Sep_2021_minimal_prior\metrics\metrics_sim_Sep_2021.json --quiet

python -m asreview metrics output\simulation\Sep_2021_no_prior\state_files\sim_Sep_2021.asreview -o output\simulation\Sep_2021_no_prior\metrics\metrics_sim_Sep_2021.json --quiet

python -m asreview metrics output\simulation\Sep_2021_llm_prior\state_files\sim_Sep_2021.asreview -o output\simulation\Sep_2021_llm_prior\metrics\metrics_sim_Sep_2021.json --quiet



:: Generate plot and tables for dataset
python scripts\get_plot.py -s output\simulation\Sep_2021_minimal_prior\state_files\ -o output\figures\plot_recall_sim_Sep_2021_minimal_prior.png
python scripts\merge_metrics.py -s output\simulation\Sep_2021_minimal_prior\metrics\ -o output\tables\metrics\metrics_sim_Sep_2021_minimal_prior.csv
python scripts\merge_tds.py -s output\simulation\Sep_2021_minimal_prior\metrics\ -o output\tables\time_to_discovery\tds_sim_Sep_2021_minimal_prior.csv

python scripts\get_plot.py -s output\simulation\Sep_2021_no_prior\state_files\ -o output\figures\plot_recall_sim_Sep_2021_no_prior.png
python scripts\merge_metrics.py -s output\simulation\Sep_2021_no_prior\metrics\ -o output\tables\metrics\metrics_sim_Sep_2021_no_prior.csv
python scripts\merge_tds.py -s output\simulation\Sep_2021_no_prior\metrics\ -o output\tables\time_to_discovery\tds_sim_Sep_2021_no_prior.csv

python scripts\get_plot.py -s output\simulation\Sep_2021_llm_prior\state_files\ -o output\figures\plot_recall_sim_Sep_2021_llm_prior.png
python scripts\merge_metrics.py -s output\simulation\Sep_2021_llm_prior\metrics\ -o output\tables\metrics\metrics_sim_Sep_2021_llm_prior.csv
python scripts\merge_tds.py -s output\simulation\Sep_2021_llm_prior\metrics\ -o output\tables\time_to_discovery\tds_sim_Sep_2021_llm_prior.csv

:: Merge descriptives and metrics
python scripts\merge_descriptives.py
python scripts\merge_metrics.py

:: Generate final plot with all simulations
::python scripts\get_plot.py -s output\tables\ -o output\figures\plot_recall_sim_all.png
