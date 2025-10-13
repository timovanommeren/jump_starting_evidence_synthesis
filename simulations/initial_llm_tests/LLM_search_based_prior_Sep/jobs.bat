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


  python -m asreview simulate generated_data\dataset_custom_priors.csv -o output\simulation\state_files\state_dataset_custom_priors__llm_priors_0.asreview   --prior-idx 0 1 && python -m asreview metrics output\simulation\state_files\state_dataset_custom_priors__llm_priors_0.asreview --quiet -o output\simulation\metrics\dataset_custom_priors__llm_priors_0_metrics.csv

  python -m asreview simulate generated_data\dataset_without_priors.csv -o output\simulation\state_files\state_dataset_without_priors__minimal_priors_0.asreview   --n-prior-included 1 --n-prior-excluded 1 && python -m asreview metrics output\simulation\state_files\state_dataset_without_priors__minimal_priors_0.asreview --quiet -o output\simulation\metrics\dataset_without_priors__minimal_priors_0_metrics.csv

  python -m asreview simulate generated_data\dataset_without_priors.csv -o output\simulation\state_files\state_dataset_without_priors__no_priors_0.asreview   && python -m asreview metrics output\simulation\state_files\state_dataset_without_priors__no_priors_0.asreview --quiet -o output\simulation\metrics\dataset_without_priors__no_priors_0_metrics.csv


  python -m asreview simulate generated_data\dataset_custom_priors.csv -o output\simulation\state_files\state_dataset_custom_priors__llm_priors_1.asreview   --prior-idx 0 1 && python -m asreview metrics output\simulation\state_files\state_dataset_custom_priors__llm_priors_1.asreview --quiet -o output\simulation\metrics\dataset_custom_priors__llm_priors_1_metrics.csv

  python -m asreview simulate generated_data\dataset_without_priors.csv -o output\simulation\state_files\state_dataset_without_priors__minimal_priors_1.asreview   --n-prior-included 1 --n-prior-excluded 1 && python -m asreview metrics output\simulation\state_files\state_dataset_without_priors__minimal_priors_1.asreview --quiet -o output\simulation\metrics\dataset_without_priors__minimal_priors_1_metrics.csv

  python -m asreview simulate generated_data\dataset_without_priors.csv -o output\simulation\state_files\state_dataset_without_priors__no_priors_1.asreview   && python -m asreview metrics output\simulation\state_files\state_dataset_without_priors__no_priors_1.asreview --quiet -o output\simulation\metrics\dataset_without_priors__no_priors_1_metrics.csv


  python -m asreview simulate generated_data\dataset_custom_priors.csv -o output\simulation\state_files\state_dataset_custom_priors__llm_priors_2.asreview   --prior-idx 0 1 && python -m asreview metrics output\simulation\state_files\state_dataset_custom_priors__llm_priors_2.asreview --quiet -o output\simulation\metrics\dataset_custom_priors__llm_priors_2_metrics.csv

  python -m asreview simulate generated_data\dataset_without_priors.csv -o output\simulation\state_files\state_dataset_without_priors__minimal_priors_2.asreview   --n-prior-included 1 --n-prior-excluded 1 && python -m asreview metrics output\simulation\state_files\state_dataset_without_priors__minimal_priors_2.asreview --quiet -o output\simulation\metrics\dataset_without_priors__minimal_priors_2_metrics.csv

  python -m asreview simulate generated_data\dataset_without_priors.csv -o output\simulation\state_files\state_dataset_without_priors__no_priors_2.asreview   && python -m asreview metrics output\simulation\state_files\state_dataset_without_priors__no_priors_2.asreview --quiet -o output\simulation\metrics\dataset_without_priors__no_priors_2_metrics.csv


  python -m asreview simulate generated_data\dataset_custom_priors.csv -o output\simulation\state_files\state_dataset_custom_priors__llm_priors_3.asreview   --prior-idx 0 1 && python -m asreview metrics output\simulation\state_files\state_dataset_custom_priors__llm_priors_3.asreview --quiet -o output\simulation\metrics\dataset_custom_priors__llm_priors_3_metrics.csv

  python -m asreview simulate generated_data\dataset_without_priors.csv -o output\simulation\state_files\state_dataset_without_priors__minimal_priors_3.asreview   --n-prior-included 1 --n-prior-excluded 1 && python -m asreview metrics output\simulation\state_files\state_dataset_without_priors__minimal_priors_3.asreview --quiet -o output\simulation\metrics\dataset_without_priors__minimal_priors_3_metrics.csv

  python -m asreview simulate generated_data\dataset_without_priors.csv -o output\simulation\state_files\state_dataset_without_priors__no_priors_3.asreview   && python -m asreview metrics output\simulation\state_files\state_dataset_without_priors__no_priors_3.asreview --quiet -o output\simulation\metrics\dataset_without_priors__no_priors_3_metrics.csv


  python -m asreview simulate generated_data\dataset_custom_priors.csv -o output\simulation\state_files\state_dataset_custom_priors__llm_priors_4.asreview   --prior-idx 0 1 && python -m asreview metrics output\simulation\state_files\state_dataset_custom_priors__llm_priors_4.asreview --quiet -o output\simulation\metrics\dataset_custom_priors__llm_priors_4_metrics.csv

  python -m asreview simulate generated_data\dataset_without_priors.csv -o output\simulation\state_files\state_dataset_without_priors__minimal_priors_4.asreview   --n-prior-included 1 --n-prior-excluded 1 && python -m asreview metrics output\simulation\state_files\state_dataset_without_priors__minimal_priors_4.asreview --quiet -o output\simulation\metrics\dataset_without_priors__minimal_priors_4_metrics.csv

  python -m asreview simulate generated_data\dataset_without_priors.csv -o output\simulation\state_files\state_dataset_without_priors__no_priors_4.asreview   && python -m asreview metrics output\simulation\state_files\state_dataset_without_priors__no_priors_4.asreview --quiet -o output\simulation\metrics\dataset_without_priors__no_priors_4_metrics.csv


  python -m asreview simulate generated_data\dataset_custom_priors.csv -o output\simulation\state_files\state_dataset_custom_priors__llm_priors_5.asreview   --prior-idx 0 1 && python -m asreview metrics output\simulation\state_files\state_dataset_custom_priors__llm_priors_5.asreview --quiet -o output\simulation\metrics\dataset_custom_priors__llm_priors_5_metrics.csv

  python -m asreview simulate generated_data\dataset_without_priors.csv -o output\simulation\state_files\state_dataset_without_priors__minimal_priors_5.asreview   --n-prior-included 1 --n-prior-excluded 1 && python -m asreview metrics output\simulation\state_files\state_dataset_without_priors__minimal_priors_5.asreview --quiet -o output\simulation\metrics\dataset_without_priors__minimal_priors_5_metrics.csv

  python -m asreview simulate generated_data\dataset_without_priors.csv -o output\simulation\state_files\state_dataset_without_priors__no_priors_5.asreview   && python -m asreview metrics output\simulation\state_files\state_dataset_without_priors__no_priors_5.asreview --quiet -o output\simulation\metrics\dataset_without_priors__no_priors_5_metrics.csv


  python -m asreview simulate generated_data\dataset_custom_priors.csv -o output\simulation\state_files\state_dataset_custom_priors__llm_priors_6.asreview   --prior-idx 0 1 && python -m asreview metrics output\simulation\state_files\state_dataset_custom_priors__llm_priors_6.asreview --quiet -o output\simulation\metrics\dataset_custom_priors__llm_priors_6_metrics.csv

  python -m asreview simulate generated_data\dataset_without_priors.csv -o output\simulation\state_files\state_dataset_without_priors__minimal_priors_6.asreview   --n-prior-included 1 --n-prior-excluded 1 && python -m asreview metrics output\simulation\state_files\state_dataset_without_priors__minimal_priors_6.asreview --quiet -o output\simulation\metrics\dataset_without_priors__minimal_priors_6_metrics.csv

  python -m asreview simulate generated_data\dataset_without_priors.csv -o output\simulation\state_files\state_dataset_without_priors__no_priors_6.asreview   && python -m asreview metrics output\simulation\state_files\state_dataset_without_priors__no_priors_6.asreview --quiet -o output\simulation\metrics\dataset_without_priors__no_priors_6_metrics.csv


  python -m asreview simulate generated_data\dataset_custom_priors.csv -o output\simulation\state_files\state_dataset_custom_priors__llm_priors_7.asreview   --prior-idx 0 1 && python -m asreview metrics output\simulation\state_files\state_dataset_custom_priors__llm_priors_7.asreview --quiet -o output\simulation\metrics\dataset_custom_priors__llm_priors_7_metrics.csv

  python -m asreview simulate generated_data\dataset_without_priors.csv -o output\simulation\state_files\state_dataset_without_priors__minimal_priors_7.asreview   --n-prior-included 1 --n-prior-excluded 1 && python -m asreview metrics output\simulation\state_files\state_dataset_without_priors__minimal_priors_7.asreview --quiet -o output\simulation\metrics\dataset_without_priors__minimal_priors_7_metrics.csv

  python -m asreview simulate generated_data\dataset_without_priors.csv -o output\simulation\state_files\state_dataset_without_priors__no_priors_7.asreview   && python -m asreview metrics output\simulation\state_files\state_dataset_without_priors__no_priors_7.asreview --quiet -o output\simulation\metrics\dataset_without_priors__no_priors_7_metrics.csv


  python -m asreview simulate generated_data\dataset_custom_priors.csv -o output\simulation\state_files\state_dataset_custom_priors__llm_priors_8.asreview   --prior-idx 0 1 && python -m asreview metrics output\simulation\state_files\state_dataset_custom_priors__llm_priors_8.asreview --quiet -o output\simulation\metrics\dataset_custom_priors__llm_priors_8_metrics.csv

  python -m asreview simulate generated_data\dataset_without_priors.csv -o output\simulation\state_files\state_dataset_without_priors__minimal_priors_8.asreview   --n-prior-included 1 --n-prior-excluded 1 && python -m asreview metrics output\simulation\state_files\state_dataset_without_priors__minimal_priors_8.asreview --quiet -o output\simulation\metrics\dataset_without_priors__minimal_priors_8_metrics.csv

  python -m asreview simulate generated_data\dataset_without_priors.csv -o output\simulation\state_files\state_dataset_without_priors__no_priors_8.asreview   && python -m asreview metrics output\simulation\state_files\state_dataset_without_priors__no_priors_8.asreview --quiet -o output\simulation\metrics\dataset_without_priors__no_priors_8_metrics.csv


  python -m asreview simulate generated_data\dataset_custom_priors.csv -o output\simulation\state_files\state_dataset_custom_priors__llm_priors_9.asreview   --prior-idx 0 1 && python -m asreview metrics output\simulation\state_files\state_dataset_custom_priors__llm_priors_9.asreview --quiet -o output\simulation\metrics\dataset_custom_priors__llm_priors_9_metrics.csv

  python -m asreview simulate generated_data\dataset_without_priors.csv -o output\simulation\state_files\state_dataset_without_priors__minimal_priors_9.asreview   --n-prior-included 1 --n-prior-excluded 1 && python -m asreview metrics output\simulation\state_files\state_dataset_without_priors__minimal_priors_9.asreview --quiet -o output\simulation\metrics\dataset_without_priors__minimal_priors_9_metrics.csv

  python -m asreview simulate generated_data\dataset_without_priors.csv -o output\simulation\state_files\state_dataset_without_priors__no_priors_9.asreview   && python -m asreview metrics output\simulation\state_files\state_dataset_without_priors__no_priors_9.asreview --quiet -o output\simulation\metrics\dataset_without_priors__no_priors_9_metrics.csv

:: Generate plot and tables for dataset
python scripts\get_plot.py -s output\simulation\state_files\ -o output\figures\plot_recall_sim.png -l filename --hide-random
python scripts\merge_metrics.py -s output\simulation\metrics\ -o output\tables\metrics\metrics_sim.csv
python scripts\merge_tds.py -s output\simulation\metrics\ -o output\tables\time_to_discovery\tds_sim.csv
python scripts\merge_descriptives.py  -s output\simulation\descriptives\
