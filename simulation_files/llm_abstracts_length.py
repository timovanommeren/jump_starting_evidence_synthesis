from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

out_dir = Path(r'C:\\Users\\timov\\Desktop\\Utrecht\\Utrecht\\MSBBSS\\thesis_timo\\simulation_results\\inclusion_only_clean')

#within each folder in out_dir, load all the csv files from the llm_abstracts folder
for dataset in out_dir.iterdir():
    if dataset.is_dir():
        llm_abstracts_folder = dataset / 'llm_abstracts'
        csv_files = list(llm_abstracts_folder.glob('*.csv'))
        abstracts = [pd.read_csv(file) for file in csv_files]
        all_abstracts = pd.concat(abstracts, ignore_index=True)
        print(f"Dataset: {dataset.name}, Total abstracts generated: {len(all_abstracts)}")
        
# generate a plot showing the distribution of abstract lengths over all datasets
        plt.figure(figsize=(10, 6))
        plt.hist(all_abstracts['abstract'].str.len(), bins=30, color='blue', alpha=0.7)
        plt.title(f'Abstract Length Distribution')
        plt.xlabel('Abstract Length (characters)')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt_path = out_dir / f'abstract_length_distribution.png'
        plt.savefig(plt_path)
        plt.close()
        print(f"Saved abstract length distribution plot to {plt_path}")