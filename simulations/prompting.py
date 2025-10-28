import pandas as pd
from pathlib import Path
import dspy
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

def generate_abstracts(name: str, stimulus: list, out_dir: Path, n_abstracts: int, length_abstracts: int, typicality: float, degree_jargon: float, llm_temperature: float, run: int) -> pd.DataFrame:
    
    
    ### Create signature ###
    lm = dspy.LM("openai/gpt-4o-mini", temperature=llm_temperature)
    dspy.configure(lm=lm)

    class MakeAbstract(dspy.Signature):
        """Generate a fake abstract based on search terms and whether it should be included or not."""
        criteria: str = dspy.InputField(desc="The inclusion or exclusion criteria of the review")
        length_abstracts: int = dspy.InputField(desc="Desired length of the abstract in words")
        typicality: float = dspy.InputField(desc="The degree to which the generated abstracts should be typical examples or edge cases for the review topic (0-1 scale)")
        degree_jargon: float = dspy.InputField(desc="The degree to which the generated abstracts should exist out of a long list of jargon or rather be written as a true abstract (with 1 representing an abstract full of jargon only and 0 representing a true abstract)")
        label_included: int = dspy.InputField(desc="1 if it would perfectly fit the review; 0 if it would be returned by the given search terms but not fit the review")
        nonce: str = dspy.InputField() 
        jsonl: str = dspy.OutputField(desc='One-line JSON object: {"doi":"None","title":"...","abstract":"...","label_included":"1/0","reasoning":"..."}')

    make_abstract = dspy.ChainOfThought(MakeAbstract)
  
    ### Generate abstracts ###  
    
    df_generated = pd.DataFrame()

    # loop to generate multiple abstracts
    for i in range(n_abstracts):
        
        #generate included abstract
        included = make_abstract(
            criteria = stimulus['inclusion_criteria'],
            label_included=1,
            nonce=f"run-{i}",
            extra_instructions=""
        ).jsonl

        #generate excluded abstract
        excluded = make_abstract(
            criteria = stimulus['exclusion_criteria'],
            label_included=0,
            nonce=f"run-{i}",
            extra_instructions=""
        ).jsonl

        #combine included and excluded abstracts into one pandas dataframe
        data = [included, excluded]
        data_dicts = [eval(item) for item in data]
        df_generated = pd.concat([df_generated, pd.DataFrame(data_dicts)], ignore_index=True)
        df_generated['label_included'] = df_generated['label_included'].astype(int)
        
        #save generated abstracts to csv file
        df_generated.to_csv(out_dir / name / f"llm_priors_run_{run}.csv", index=False)

    return df_generated