"""
Generates LIWC results for the 'Story' values in BFI_Story_data.csv

Input: BFI_Story_data.csv
Output: LIWC_results.csv

"""

import os
import re
import liwc
import pandas as pd
import requests
from collections import Counter

def tokenize(text):
    """Tokenizes the input text into words."""
    return re.findall(r'\w+', text.lower())

def download_liwc_dictionary():
    """Downloads the LIWC dictionary from a URL and saves it to a file."""
    liwc_url = "https://raw.githubusercontent.com/chun-hu/conversation-modeling/master/LIWC2007_English100131.dic"
    liwc_filename = 'LIWC2007_English100131.dic'
    response = requests.get(liwc_url)
    with open(liwc_filename, 'wb') as file:
        file.write(response.content)

def analyze_text_with_liwc(text, parse):
    """Analyzes the input text using the LIWC dictionary parser."""
    tokens = tokenize(text)
    return Counter(category for token in tokens for category in parse(token))

def read_csv_and_analyze(file_path, parse):
    """Reads a CSV file and analyzes text with LIWC."""
    data = pd.read_csv(file_path)
    
    results = []
    for index, row in data.iterrows():
        text = row['Story']
        analysis_results = analyze_text_with_liwc(text, parse)
        analysis_results['Subject_ID'] = row['Subject_ID']
        analysis_results['Group'] = row['Group']
        analysis_results['Experimental_condition'] = row['Experimental_condition']
        results.append(analysis_results)
    return results
    
def main(input_file):
    """Main function to execute the LIWC analysis."""
    liwc_filename = 'LIWC2007_English100131.dic'

    parse, category_names = liwc.load_token_parser(liwc_filename)

    # Analyze the CSV file
    results = read_csv_and_analyze(input_file, parse)
    results_df = pd.DataFrame(results)
    
    # Fill empty cells with 0
    results_df.fillna(0, inplace=True)
    
    # Ensure the 'Subject_ID', 'Group', and 'Experimental_condition' columns are first
    fixed_cols = ['Subject_ID', 'Group', 'Experimental_condition']
    liwc_cols = sorted([col for col in results_df if col not in fixed_cols])
    cols = fixed_cols + liwc_cols
    
    # Construct the output file path in the same folder as the input file
    output_file_path = os.path.splitext(input_file)[0].replace('BFI_Story_data', '') + 'LIWC_results.csv'

    results_df = results_df[cols]
    
    # Save LIWC results csv to the dynamically constructed output file path
    results_df.to_csv(output_file_path, index=False)
    

if __name__ == "__main__":
    #Define Paths
    Control_input_file = 'output/Control/BFI_Story_data.csv'
    ANACREA_input_file = 'output/ANACREA/BFI_Story_data.csv'
    CREAANA_input_file = 'output/CREAANA/BFI_Story_data.csv'
    
    files = [Control_input_file, ANACREA_input_file, CREAANA_input_file]
    for file in files:
        print(file)
        main(file)
    
    

