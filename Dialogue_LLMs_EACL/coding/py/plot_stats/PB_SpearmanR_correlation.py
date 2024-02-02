"""
Calculate the PB or SpearmanR correlation for the data.

For PB correlation, use the file: encoded_BFI_data.csv
For SpearmanR correlation, use the file: BFI_Story_data.csv
"""

import pandas as pd
from scipy.stats import pointbiserialr, spearmanr

def load_data(encoded_prompt_path, liwc_results_path):
    """
    Load the datasets, ignoring the first three columns.
    """
    encoded_prompt_df = pd.read_csv(encoded_prompt_path).iloc[:, 3:]
    liwc_results_df = pd.read_csv(liwc_results_path).iloc[:, 3:]
    return encoded_prompt_df, liwc_results_df

def filter_significant_correlations(results_df):
    """
    Remove insignificant p-values and their corresponding correlations.
    """
    for col in results_df.columns:
        if 'p-value' in col:
            results_df.loc[results_df[col] >= 0.05, col.replace(' p-value', ' correlation')] = float('nan')
            results_df.loc[results_df[col] >= 0.05, col] = float('nan')
    return results_df

def save_to_csv(df, file_path):
    """
    Save the DataFrame to a CSV file.
    """
    df.to_csv(file_path, index=False)
    return file_path

def calculate_correlations(encoded_prompt_df, liwc_results_df, correlation_method):
   """
   Calculate point biserial correlations and p-values for each pair of columns.
   """
   results = {liwc_feature: {} for liwc_feature in liwc_results_df.columns}
   for prompt_column in encoded_prompt_df.columns:
        for liwc_column in liwc_results_df.columns:
            if correlation_method == 'PB':
                correlation, p_value = pointbiserialr(encoded_prompt_df[prompt_column], liwc_results_df[liwc_column])
            elif correlation_method == 'SpearmanR':
                correlation, p_value = spearmanr(encoded_prompt_df[prompt_column], liwc_results_df[liwc_column])
            else:
                raise ValueError("Unsupported correlation method. Choose 'PB' or 'SpearmanR'.")
            
            results[liwc_column][f'{prompt_column} correlation'] = correlation
            results[liwc_column][f'{prompt_column} p-value'] = p_value
   return results

def print_top_5_correlations(results_df):
    """
    Prints the top-5 largest absolute correlations for each BFI trait.
    """
    # Extracting BFI trait names from the column headers
    bfi_traits = [col.replace(' correlation', '') for col in results_df.columns if 'correlation' in col]
    print(f" bfi_traits: \n {bfi_traits}")

    for trait in bfi_traits:
        print(f"Top-5 correlations for {trait}:")

        # Filtering the results for the current trait
        trait_correlations = results_df[['LIWC Feature', f'{trait} correlation']].copy()
        trait_correlations.dropna(inplace=True)  # Dropping NaN values

        # Sorting by the absolute value of correlations
        trait_correlations['abs_correlation'] = trait_correlations[f'{trait} correlation'].abs()
        top_5 = trait_correlations.sort_values('abs_correlation', ascending=False).head(5)

        # Printing the top-5 results
        for _, row in top_5.iterrows():
            print(f"  {row['LIWC Feature']}: {row[f'{trait} correlation']:.3f}")

        print()  # Add an empty line for better readability between traits


#------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------PBcorrelation--------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
# Base directory where the folders are located
base_dir = "../output/"

# List of folder names to process
folders = ["CONTROL", "CREAANA", "ANACREA"]  # Adjust folder names as needed

for folder in folders:
    # Update paths for BFI Story data and LIWC results for each folder
    BFI_Story_data_path = f"{base_dir}{folder}/encoded_BFI_data.csv"
    liwc_results_path = f"{base_dir}{folder}/LIWC_results.csv"
    output_file_path = f"{base_dir}{folder}/PB_corr.csv"

    # Load the datasets
    BFI_Story_data_df, liwc_results_df = load_data(BFI_Story_data_path, liwc_results_path)

    # Specify the correlation method here
    correlation_method = 'PB'  

    # Calculate correlations based on the specified method
    results = calculate_correlations(BFI_Story_data_df, liwc_results_df, correlation_method)

    # Convert the results to a DataFrame and reorder columns
    results_df = pd.DataFrame.from_dict(results, orient='index')
    columns_order = [f'{trait} correlation' for trait in BFI_Story_data_df.columns] + \
                    [f'{trait} p-value' for trait in BFI_Story_data_df.columns]
    # Ensure we only include columns that exist in results_df to avoid KeyError
    existing_columns = [col for col in columns_order if col in results_df.columns]
    results_df = results_df.reindex(columns=existing_columns)
    results_df.reset_index(inplace=True)
    results_df.rename(columns={'index': 'LIWC Feature'}, inplace=True)

    # Save the complete results
    save_to_csv(results_df, output_file_path)

    # Print output file path and optionally print top 5 correlations
    print(f"Complete results for {folder}:", output_file_path)
    print_top_5_correlations(results_df)


#------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------SpearmanR correlation--------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------

# Base directory where the folders are located
base_dir = "../output/"

# List of folder names to process
folders = ["CONTROL", "CREAANA", "ANACREA"]  # Adjust folder names as needed

for folder in folders:
    # Update paths for BFI Story data and LIWC results for each folder
    BFI_Story_data_path = f"{base_dir}{folder}/BFI_Story_data.csv"
    liwc_results_path = f"{base_dir}{folder}/LIWC_results.csv"
    output_file_path = f"{base_dir}{folder}/SpearmanR_corr.csv"

    # Load the datasets
    BFI_Story_data_df, liwc_results_df = load_data(BFI_Story_data_path, liwc_results_path)

    # Specify the correlation method here
    correlation_method = 'SpearmanR'  # or 'PB' for Point Biserial

    # Calculate correlations based on the specified method
    results = calculate_correlations(BFI_Story_data_df, liwc_results_df, correlation_method)

    # Convert the results to a DataFrame and reorder columns
    results_df = pd.DataFrame.from_dict(results, orient='index')
    columns_order = [f'{trait} correlation' for trait in BFI_Story_data_df.columns] + \
                    [f'{trait} p-value' for trait in BFI_Story_data_df.columns]
    # Ensure we only include columns that exist in results_df to avoid KeyError
    existing_columns = [col for col in columns_order if col in results_df.columns]
    results_df = results_df.reindex(columns=existing_columns)
    results_df.reset_index(inplace=True)
    results_df.rename(columns={'index': 'LIWC Feature'}, inplace=True)

    # Save the complete results
    save_to_csv(results_df, output_file_path)

    # Print output file path and optionally print top 5 correlations
    print(f"Complete results for {folder}:", output_file_path)
    print_top_5_correlations(results_df)

