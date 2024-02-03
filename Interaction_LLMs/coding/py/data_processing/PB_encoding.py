import pandas as pd
import os

def encode_BFI_values(folders, group):
    """
    Process 'BFI_Story_data.csv' files in the given folders and encode values as per requirements.

    Args:
    folders (list): List of folder paths containing 'BFI_Story_data.csv'.
    group (str): The experimental group of the dataset, used for setting the 'Experimental_condition'.
    """
    for folder in folders:
        # Construct file paths
        file_path = os.path.join(folder, 'BFI_Story_data.csv')
        new_file_path = os.path.join(folder, 'encoded_BFI_data.csv')

        # Check if the file exists in the folder
        if os.path.exists(file_path):
            # Load the CSV file
            data = pd.read_csv(file_path)

            # Drop the "Word_Length", "Story", and all "Writing_" columns
            columns_to_drop = ['Word_Length', 'Story'] + [col for col in data.columns if col.startswith('Writing_')]
            data = data.drop(columns=columns_to_drop)

            # Adjust encoding mechanism based on 'Group' column
            for col in data.columns[2:]:  # Assuming the 'Group' column is included and we start modifying from the 3rd column
                data[col] = data['Group'].apply(lambda x: 1 if x == "Creative" else 0)

            # Set 'Experimental_condition' based on group
            data['Experimental_condition'] = 0 if group == "CONTROL" else 1

            # Write to the new file
            data.to_csv(new_file_path, index=False)
        else:
            print(f"File not found in {folder}")

if __name__ == "__main__":
    folders_control = ['../output/Control/']
    folders_anacrea = ['../output/ANACREA/']
    folders_creaana = ['../output/CREAANA/']

    # Encode values for the CONTROL group
    encode_BFI_values(folders_control, "CONTROL")

    # Encode values for the ANACREA group
    encode_BFI_values(folders_anacrea, "ANACREA")

    # Encode values for the CREAANA group
    encode_BFI_values(folders_creaana, "CREAANA")
