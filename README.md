# Jump Starting Evidence Synthesis

This repository contains the code to reproduce the results of *Jump Starting Evidence Synthesis: Initializing Active Learning Models for Systematic Reviews using LLM-generated Data*

## Set up

To be able to replicate the results of the simulation study, please follow the instructions below.

### Install python packages

Open a terminal and create a new virtual environment:
```
python -m venv .venv
```

Activate the virtual environment:
```
source .venv/bin/activate
```

Install the required Python dependencies:
```
pip install -r requirements.txt
```

Add your OpenAI API:
```
cp env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_KEY=your_api_key_here
```

### Run the simulation study

First downloaded the datasets from: [the SYNERGY repository](https://github.com/asreview/synergy-dataset) (we of course encourage replications using other datasets). 

(What about meta-data? -- I now have a local fiel from Emily. Will this be published?)

A prepared .bat file? -- but for now:
```
python simulation_files\run.py run 'path to synergy datasets' simulation_results\results 'inclusion_criteria exclusion_criteria'
```


### Render the report

Knit the main.tex file:
```
latexmk -pdf -shell-escape main.tex
```

