import pandas as pd
from pathlib import Path
import json
import re
import dspy
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

def generate_abstracts(name: str, stimulus: list, out_dir: Path, n_abstracts: int, length_abstracts: int, llm_temperature: float, run: int) -> pd.DataFrame:
    
    ### Create signature ###
    lm = dspy.LM("openai/gpt-4o-mini",
                temperature=llm_temperature, 
                cache=False
                )
    
    dspy.configure(
        lm=lm,
        adapter=dspy.JSONAdapter(),
    )

    class MakeAbstract(dspy.Signature):
        """Generate a synthetic abstract based on the eligibility criteria of the systematic review."""
        
        # Input fields
        label_relevant: int = dspy.InputField(desc="1 for an example of an abstract and title relevant to the review; 0 for an example of an abstract and title irrelevant to the review")
        criteria: str = dspy.InputField(desc="The inclusion or exclusion criteria of the review")
        length_abstracts: int = dspy.InputField(desc="The number of words that the generated abstract should approximately contain.")
        # typicality: int = dspy.InputField(desc="A binary variable representing whether an abstract should be typical (1) or atypical (0) for the review. The typical abstracts generated should be 'in the center' of the relevant or irrelevant cluster of abstracts classified by reviewers, whereas the atypical abstracts should aim to be on the 'edges' of these clusters. In other words, typical abstracts should be more representative of the review topic, whereas atypical abstracts should be more unusual or unique in their content.")
        # degree_jargon: float = dspy.InputField(desc="The degree to which the generated abstracts should exist out of a long list of jargon or rather be written as a true abstract (with 1.00 representing an abstract full of jargon only and 0.00 representing a true abstract)")
     
        # Output fields   
        doi: str = dspy.OutputField(desc="Should always be 'None' for generated abstracts")
        title: str = dspy.OutputField(desc="The generated title of the abstract")
        abstract: str = dspy.OutputField(desc=f"The generated abstract text of {length_abstracts} words")
        label_included: int = dspy.OutputField(desc="1 if the abstract is included based on the inclusion criteria, 0 if the abstract is excluded based on the exclusion criteria")
        reasoning: str = dspy.OutputField(desc="The reasoning behind inclusion or exclusion of the abstract")
        
    make_abstract = dspy.ChainOfThought(MakeAbstract)
  
    ### Generate abstracts ###  
    
    df_generated = pd.DataFrame(columns=["doi", "title", "abstract", "label_included", "reasoning"])
    
    max_attempts = 10
    
    # loop to generate multiple abstracts
    for i in range(n_abstracts):
    
        success = False
        attempt = 0
        
        while not success and attempt < max_attempts:
            
            if attempt > 1:
                print(f"Regenerating abstracts for dataset {name} (attempt {attempt}/{max_attempts}).")
            attempt += 1
        
            #generate relevant abstract
            relevant = make_abstract(
                label_relevant=1,
                criteria = stimulus['inclusion_criteria'],
                length_abstracts=length_abstracts,
                llm_temperature=llm_temperature,
            )

            relevant_abstract = {
                "doi": relevant.doi,
                "title": relevant.title,
                "abstract": relevant.abstract,
                "label_included": 1,
                "reasoning": relevant.reasoning,
            }
        
            parsed_data = {
                "doi": [relevant_abstract["doi"]],
                "title": [relevant_abstract["title"]],
                "abstract": [relevant_abstract["abstract"]],
                "label_included": [relevant_abstract["label_included"]],
                "reasoning": [relevant_abstract["reasoning"]],
            }
            
            # try to append generated abstracts to dataframe
            df_generated = pd.concat([df_generated, pd.DataFrame(parsed_data)], ignore_index=True)
            
            success = True
            break # exit the attempt loop if successful
        

    
    #ensure that the label is of the generated abstracts is integer
    df_generated = df_generated.astype({"label_included":int})
       
    #save generated abstracts to csv file in new directory
    path_abstracts = out_dir / name / f"llm_abstracts/llm_abstracts_run_{run}_IVs_{n_abstracts}_{length_abstracts}_{llm_temperature}.csv"
    path_abstracts.parent.mkdir(parents=True, exist_ok=True)
    df_generated.to_csv(path_abstracts, index=False)


    return df_generated

