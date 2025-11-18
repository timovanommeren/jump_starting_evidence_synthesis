import pandas as pd
from pathlib import Path
import json
import re
import dspy
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

def generate_abstracts(name: str, stimulus: list, out_dir: Path, n_abstracts: int, length_abstracts: int, typicality: float, degree_jargon: float, llm_temperature: float, run: int) -> pd.DataFrame:
    
    # transform typicality to list of integers (1 or 0) with the proportion equal to the inputted float value
    typicality = [1 if i < n_abstracts * typicality else 0 for i in range(n_abstracts)]
    
    ### Create signature ###
    lm = dspy.LM("openai/gpt-4o-mini", temperature=llm_temperature, cache=False)
    dspy.configure(lm=lm)

    class MakeAbstract(dspy.Signature):
        """Generate a fake abstract based on search terms and whether it should be included or not."""
        label_relevant: int = dspy.InputField(desc="1 for an example of an abstract and title relevant to the review; 0 for an example of an abstract and title irrelevant to the review")
        criteria: str = dspy.InputField(desc="The inclusion or exclusion criteria of the review")
        length_abstracts: int = dspy.InputField(desc="The number of words that the generated abstract should approximately contain.")
        typicality: int = dspy.InputField(desc="A binary variable representing whether an abstract should be typical (1) or atypical (0) for the review. The typical abstracts generated should be 'in the center' of the relevant or irrelevant cluster of abstracts classified by reviewers, whereas the atypical abstracts should aim to be on the 'edges' of these clusters. In other words, typical abstracts should be more representative of the review topic, whereas atypical abstracts should be more unusual or unique in their content.")
        degree_jargon: float = dspy.InputField(desc="The degree to which the generated abstracts should exist out of a long list of jargon or rather be written as a true abstract (with 1.00 representing an abstract full of jargon only and 0.00 representing a true abstract)")
        jsonl: str = dspy.OutputField(desc='One-line JSON object: {"doi":"None","title":"...","abstract":"...","label_included":"1/0","reasoning":"..."}')

    make_abstract = dspy.ChainOfThought(MakeAbstract)
  
    ### Generate abstracts ###  
    
    df_generated = pd.DataFrame()
    
    # loop to generate multiple abstracts
    for i in range(n_abstracts):
        
        #generate relevant abstract
        relevant = make_abstract(
            label_relevant=1,
            criteria = stimulus['inclusion_criteria'],
            length_abstracts=length_abstracts,
            typicality=typicality[i],
            degree_jargon=degree_jargon
        ).jsonl

        #generate irrelevant abstract
        irrelevant = make_abstract(
            label_relevant=0,
            criteria = stimulus['exclusion_criteria'],
            length_abstracts=length_abstracts,
            typicality=typicality[i],
            degree_jargon=degree_jargon
        ).jsonl
        
        
        # clean control characters that may cause a JSON parsing errors
        cleaned_relevant = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', relevant)
        cleaned_irrelevant = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', irrelevant)
        
        # add the generated abstracts to the dataframe with error handling
        try:
            parsed_data = [json.loads(item) for item in [cleaned_relevant, cleaned_irrelevant]]
            df_generated = pd.concat([df_generated, pd.DataFrame(parsed_data).astype({"label_included":int})])
            
        except json.JSONDecodeError as error:
            print(f"\n=== JSON Parsing Error on iteration {i} of run {run} ===")
            print(f"Error: {error}")
            print(f"\nRelevant JSON (first 200 chars): {cleaned_relevant[:200]}...")
            print(f"\nIrrelevant JSON (first 200 chars): {cleaned_irrelevant[:200]}...")
            raise
        
    #save generated abstracts to csv file in new directory
    path_abstracts = out_dir / name / f"llm_abstracts/llm_abstracts_run_{run}_IVs_{n_abstracts}_{length_abstracts}_{typicality}_{degree_jargon}_{llm_temperature}.csv"
    path_abstracts.parent.mkdir(parents=True, exist_ok=True)
    df_generated.to_csv(path_abstracts, index=False)


    return df_generated