# Simulation study

*This project was rendered with ASReview-Makita version 0.1.dev94+gb0393b60b.d20251009.*

This project was rendered from the Makita-basic llm template. See [asreview/asreview-makita#templates](https://github.com/asreview/asreview-makita#templates) for template rules and formats.

The template is described as: 'Basic simulation for N runs'.

## Installation

This project depends on Python 3.7 or later (python.org/download), and [ASReview](https://asreview.nl/download/). Install the following dependencies to run the simulation and analysis in this project.

```sh
pip install asreview>=2.0 asreview-insights>=1.6
```

## Data

The performance on the following datasets is evaluated:

- data\Brouwer_2019.csv

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
    │   ├── 📜Brouwer_2019.csv
    ├── 📂scripts
    │   ├── 📜data_describe.py
    │   ├── 📜get_plot.py
    │   ├── 📜merge_descriptives.py
    │   ├── 📜merge_metrics.py
    │   ├── 📜merge_tds.py
    │   └── 📜...
    └── 📂output
        ├── 📂simulation
        |   └── 📂Brouwer_2019
        |       ├── 📂descriptives
        |       |   └── 📜data_stats_Brouwer_2019.json
        |       ├── 📂state_files
        |       |   ├── 📜sim_Brouwer_2019_`x`.asreview
        |       |   └── 📜...
        |       └── 📂metrics
        |           ├── 📜metrics_sim_Brouwer_2019_`x`.json
        |           └── 📜...
        ├── 📂tables
        |   ├── 📜data_descriptives.csv
        |   ├── 📜data_descriptives.xlsx
        |   ├── 📜tds_sim_Brouwer_2019.csv
        |   ├── 📜tds_sim_Brouwer_2019.xlsx
        |   ├── 📜tds_summary.csv
        |   ├── 📜tds_summary.xlsx
        |   ├── 📜metrics_sim_Brouwer_2019_metrics.csv
        |   ├── 📜metrics_sim_Brouwer_2019_metrics.xlsx
        |   ├── 📜metrics_summary.csv
        |   └── 📜metrics_summary.xlsx
        └── 📂figures
            ├── 📈plot_recall_Brouwer_2019.png
