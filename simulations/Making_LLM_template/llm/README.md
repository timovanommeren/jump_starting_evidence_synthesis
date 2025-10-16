# Simulation study

*This project was rendered with ASReview-Makita version 0.1.dev103+gac2e4ab9d.*

This project was rendered from the Makita-basic llm template. See [asreview/asreview-makita#templates](https://github.com/asreview/asreview-makita#templates) for template rules and formats.

The template is described as: 'Basic simulation for N runs'.

## Installation

This project depends on Python 3.7 or later (python.org/download), and [ASReview](https://asreview.nl/download/). Install the following dependencies to run the simulation and analysis in this project.

```sh
pip install asreview>=2.0 asreview-insights>=1.6
```

## Data

The performance on the following datasets is evaluated:

- data\Sep_2021.csv

## Run simulation

To start the simulation, run the following command in the project directory.

```
jobs.bat
```

## Structure

The following files are found in this project:

    📦Makita
    ├── 📜README.md
    ├── 📜jobs.bat
    ├── 📂data
    │   ├── 📜Sep_2021.csv
    ├── 📂scripts
    │   ├── 📜data_describe.py
    │   ├── 📜get_plot.py
    │   ├── 📜merge_descriptives.py
    │   ├── 📜merge_metrics.py
    │   ├── 📜merge_tds.py
    │   └── 📜...
    └── 📂output
        ├── 📂simulation
        |   └── 📂Sep_2021
        |       ├── 📂descriptives
        |       |   └── 📜data_stats_Sep_2021.json
        |       ├── 📂state_files
        |       |   ├── 📜sim_Sep_2021_`x`.asreview
        |       |   └── 📜...
        |       └── 📂metrics
        |           ├── 📜metrics_sim_Sep_2021_`x`.json
        |           └── 📜...
        ├── 📂tables
        |   ├── 📜data_descriptives.csv
        |   ├── 📜data_descriptives.xlsx
        |   ├── 📜tds_sim_Sep_2021.csv
        |   ├── 📜tds_sim_Sep_2021.xlsx
        |   ├── 📜tds_summary.csv
        |   ├── 📜tds_summary.xlsx
        |   ├── 📜metrics_sim_Sep_2021_metrics.csv
        |   ├── 📜metrics_sim_Sep_2021_metrics.xlsx
        |   ├── 📜metrics_summary.csv
        |   └── 📜metrics_summary.xlsx
        └── 📂figures
            ├── 📈plot_recall_Sep_2021.png
