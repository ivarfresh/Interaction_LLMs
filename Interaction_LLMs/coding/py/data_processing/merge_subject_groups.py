import pandas as pd
import os

def merge_subjects(analytic_path, creative_path, output_path):
    """
    Merges analytic and creative subjects files for the same data file name, adjusting 'Subject_ID' to be continuous.
    """
    # Load the analytic and creative data
    analytic_data = pd.read_csv(analytic_path)
    creative_data = pd.read_csv(creative_path)
    
    # Adjust subject_id's in creative_data to be continuous with analytic_data
    max_subject_id_analytic = analytic_data['Subject_ID'].max()
    creative_data['Subject_ID'] = creative_data['Subject_ID'] + max_subject_id_analytic
    
    # Merge the data
    merged_data = pd.concat([analytic_data, creative_data], ignore_index=True)
    
    # Save the merged data to the output path
    merged_data.to_csv(output_path, index=False)
    
    # Print the path of the output file
    print(f"Combined data saved to: {output_path}")
    

def process_experiment_groups(folder_path, file_types):
    """
    Processes each file type for the experiment groups by merging 'analytic' and 'creative' subjects.
    The group name is dynamically extracted from the folder_path.
    """
    # Extract the group name from the folder_path
    group_name = os.path.basename(os.path.normpath(folder_path))
    # Assuming the group name is the entire folder name or adjust as necessary
    
    print(f'group_name; {group_name}')

    for file_type in file_types:
        analytic_path = os.path.join(folder_path, f"{group_name}_analytic_{file_type}.csv")
        creative_path = os.path.join(folder_path, f"{group_name}_creative_{file_type}.csv")
        output_path = os.path.join(folder_path, f"{group_name}_merged_{file_type}.csv")
        
        merge_subjects(analytic_path, creative_path, output_path)

if __name__ == "__main__":
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------- ANACREA group --------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
    folder_path = '../output/ANACREA/'  # Adjust as necessary
    file_types = ['BFI_data_init', 'BFI_data_writing', 'childhood']
    process_experiment_groups(folder_path, file_types)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------- CREAANA group --------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
    folder_path = '../output/CREAANA/'  # Adjust as necessary
    file_types = ['BFI_data_init', 'BFI_data_writing', 'childhood']
    process_experiment_groups(folder_path, file_types)


