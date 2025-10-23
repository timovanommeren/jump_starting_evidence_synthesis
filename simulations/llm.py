import pandas as pd
import dspy
from dotenv import load_dotenv


### Create signature

load_dotenv()  # Load environment variables from .env file

lm = dspy.LM("openai/gpt-4o-mini", temperature=0.7)
dspy.configure(lm=lm)

class MakeAbstract(dspy.Signature):
    """Generate a fake abstract based on search terms and whether it should be included or not."""
    criteria: str = dspy.InputField(desc="The inclusion or exclusion criteria of the review")
    label_included: int = dspy.InputField(desc="1 if it would perfectly fit the review; 0 if it would be returned by the given search terms but not fit the review")
    nonce: str = dspy.InputField() 
    jsonl: str = dspy.OutputField(desc='One-line JSON object: {"doi":"None","title":"...","abstract":"...","label_included":"1/0","reasoning":"..."}')

make_abstract = dspy.ChainOfThought(MakeAbstract)


### Generaye abstracts

def generate_abstracts(datasets: dict, n_abstracts: int, stimulus: list):

    generated_abstracts = []

    for dataset_names in range(len(datasets)):
        
        print(f'Generating abstracts for dataset: {dataset_names}')
        
        df_generated = pd.DataFrame()

        # loop to generate multiple abstracts
        for i in range(n_abstracts):
            
            #generate included abstract
            included = make_abstract(
                criteria = stimulus[dataset_names]['inclusion_criteria'],
                label_included=1,
                nonce=f"run-{i}",
                extra_instructions=""
            ).jsonl

            #generate excluded abstract
            excluded = make_abstract(
                criteria = stimulus[dataset_names]['exclusion_criteria'],
                label_included=0,
                nonce=f"run-{i}",
                extra_instructions=""
            ).jsonl

            #combine included and excluded abstracts into one pandas dataframe
            data = [included, excluded]
            data_dicts = [eval(item) for item in data]
            df_generated = pd.concat([df_generated, pd.DataFrame(data_dicts)], ignore_index=True)
            df_generated['label_included'] = df_generated['label_included'].astype(int)

        generated_abstracts.append(df_generated)

    return generated_abstracts