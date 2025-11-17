# Jump Starting Evidence Synthesis

This repository contains the code to reproduce the results of *Jump Starting Evidence Synthesis: Initializing Active Learning Models for Systematic Reviews using LLM-generated Data*

## Set up

To be able to replicate the results of the simulation study, please follow the instructions below

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

