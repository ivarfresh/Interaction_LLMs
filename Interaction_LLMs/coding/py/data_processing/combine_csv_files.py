"""
This file:
1. Add BFI writing columns to BFI init data with 'Writing_' prefix.
2. Merges the 'childhood' data with the existing dataframe without duplicating 'Group' and 'Experimental_condition'.
3. Calculates word length for each story and adds it as a new column.

Output file: ../output/{Group_name}/BFI_Story_data.csv

Output file columns: 
    "Subject_ID", "Group", "Experimental_condition",
    "Extraversion", "Agreeableness", "Conscientiousness",
    "Neuroticism", "Openness", "Writing_Extraversion",
    "Writing_Agreeableness", "Writing_Conscientiousness",
    "Writing_Neuroticism", "Writing_Openness", "Word_Length", "Story"

"""

import pandas as pd

def add_writing_columns(bfi_init_path, bfi_writing_path):
    """
    Add BFI writing columns to BFI init data with 'Writing_' prefix.
    """
    bfi_data_init = pd.read_csv(bfi_init_path)
    bfi_data_writing = pd.read_csv(bfi_writing_path)
    
    # Define the columns to add from bfi_data_writing with renamed columns
    columns_to_add = ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"]
    renamed_columns = ["Writing_" + col for col in columns_to_add]

    # Add the selected columns to the initial dataframe with 'Writing_' prefix
    bfi_data_init[renamed_columns] = bfi_data_writing[columns_to_add]
    
    return bfi_data_init

def add_childhood_data(merged_df, childhood_path):
    """
    Merge the childhood data with the existing dataframe without duplicating 'Group' and 'Experimental_condition'.
    """
    childhood_data = pd.read_csv(childhood_path)
    
    # Exclude 'Group' and 'Experimental_condition' from the merge if they exist in childhood_data
    if 'Group' in childhood_data.columns and 'Experimental_condition' in childhood_data.columns:
        childhood_data = childhood_data.drop(columns=['Group', 'Experimental_condition'])
    
    merged_df = pd.merge(merged_df, childhood_data, on='Subject_ID', how='outer')
    
    return merged_df

def calculate_word_length(df, story_column):
    """
    Calculates word length for each story and adds it as a new column.
    """
    df['Word_Length'] = df[story_column].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)
    return df

def process_and_save_data(bfi_init_path, bfi_writing_path, childhood_path, output_file_path):
    bfi_data_combined = add_writing_columns(bfi_init_path, bfi_writing_path)
    bfi_data_final = add_childhood_data(bfi_data_combined, childhood_path)
    bfi_data_final = calculate_word_length(bfi_data_final, "Story")
    
    column_order = [col for col in bfi_data_final.columns if col not in ["Word_Length", "Story"]] + ["Word_Length", "Story"]
    bfi_data_final = bfi_data_final[column_order]
    
    bfi_data_final.to_csv(output_file_path, index=False)
    print(f"Data processing complete. Output file saved to: {output_file_path}")

if __name__ == "__main__":
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------- CONTROL group --------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
    folder_path = '../output/Control/'
    file_BFI_init = folder_path + 'CONTROL_BFI_data_init.csv'
    file_BFI_writing = folder_path + 'CONTROL_BFI_data_writing.csv'
    file_childhood = folder_path + 'CONTROL_Childhood.csv'
    output_file_path = folder_path + 'BFI_Story_data.csv'
    # Call the processing function with specified paths
    process_and_save_data(file_BFI_init, file_BFI_writing, file_childhood, output_file_path)
    

#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------- ANACREA group --------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
    folder_path = '../output/ANACREA/'
    file_BFI_init = folder_path + 'ANACREA_merged_BFI_data_init.csv'
    file_BFI_writing = folder_path + 'ANACREA_merged_BFI_data_writing.csv'
    file_childhood = folder_path + 'ANACREA_merged_Childhood.csv'
    output_file_path = folder_path + 'BFI_Story_data.csv'
    # Call the processing function with specified paths
    process_and_save_data(file_BFI_init, file_BFI_writing, file_childhood, output_file_path)
 
    
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------- CREAANA group --------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
    folder_path = '../output/CREAANA/'
    file_BFI_init = folder_path + 'CREAANA_merged_BFI_data_init.csv'
    file_BFI_writing = folder_path + 'CREAANA_merged_BFI_data_writing.csv'
    file_childhood = folder_path + 'CREAANA_merged_Childhood.csv'
    output_file_path = folder_path + 'BFI_Story_data.csv'
    # Call the processing function with specified paths
    process_and_save_data(file_BFI_init, file_BFI_writing, file_childhood, output_file_path)



