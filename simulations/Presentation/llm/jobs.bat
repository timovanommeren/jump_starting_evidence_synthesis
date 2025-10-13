@ echo off
COLOR E0

:: version 0.1.dev94+gb0393b60b.d20251009

:: Create folder structure. By default, the folder 'output' is used to store output.
mkdir output
mkdir output\simulation
mkdir output\tables
mkdir output\tables\metrics
mkdir output\tables\time_to_discovery
mkdir output\figures


::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:::::: DATASET: Brouwer_2019
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

:: Create output folder
mkdir output\simulation\Brouwer_2019\
mkdir output\simulation\Brouwer_2019\metrics

:: Collect descriptives about the dataset Brouwer_2019
mkdir output\simulation\Brouwer_2019\descriptives
python scripts\data_describe.py data\Brouwer_2019.csv -o output\simulation\Brouwer_2019\descriptives\data_stats_Brouwer_2019.json

:: Simulate runs
mkdir output\simulation\Brouwer_2019\state_files
python -m asreview simulate data\Brouwer_2019.csv -o output\simulation\Brouwer_2019\state_files\sim_Brouwer_2019.asreview --prior-seed 535 --seed 165 --n-stop 10 --n-prior-included 1 --n-prior-excluded 1

python -m asreview simulate data\Brouwer_2019.csv -o output\simulation\Brouwer_2019\state_files\sim_Brouwer_2019.asreview --prior-seed 535 --seed 165 --n-stop 10

python -m asreview simulate data\Brouwer_2019_llm.csv -o output\simulation\Brouwer_2019_llm\state_files\sim_Brouwer_2019_llm.asreview --prior-seed 535 --seed 165 --n-stop 10 --prior-idx 38114 38115

python -m asreview metrics output\simulation\Brouwer_2019\state_files\sim_Brouwer_2019.asreview -o output\simulation\Brouwer_2019\metrics\metrics_sim_Brouwer_2019.json --quiet

:: Generate plot and tables for dataset
python scripts\get_plot.py -s output\simulation\Brouwer_2019\state_files\ -o output\figures\plot_recall_sim_Brouwer_2019.png
python scripts\merge_metrics.py -s output\simulation\Brouwer_2019\metrics\ -o output\tables\metrics\metrics_sim_Brouwer_2019.csv
python scripts\merge_tds.py -s output\simulation\Brouwer_2019\metrics\ -o output\tables\time_to_discovery\tds_sim_Brouwer_2019.csv

:: Merge descriptives and metrics
python scripts\merge_descriptives.py
python scripts\merge_metrics.py
