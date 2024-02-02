"""
Code to generate our Means, ANOVA and Cohens'd results. 
Uncomment the desired parts to run. 
"""

import pandas as pd
from scipy.stats import f_oneway
import numpy as np

def process_data_file(file_path, group):
    data = pd.read_csv(file_path, delimiter=',')
    creative_data = data[data['Group'] == group]
    writing_data = creative_data.filter(regex='^Writing_')
    return writing_data

def process_and_align_data(control_file, experimental_file, group):
    control_data = process_data_file(control_file, group)
    experimental_data = process_data_file(experimental_file, group)

    # Ensure both dataframes have the same columns
    common_columns = control_data.columns.intersection(experimental_data.columns)
    control_data = control_data[common_columns]
    experimental_data = experimental_data[common_columns]

    return control_data, experimental_data

def compute_means(control_data, experimental_data):
    means = {}
    for column in control_data.columns:
        means[column] = {
            'Mean_Control_Group': control_data[column].mean(), 
            'Mean_Experimental_Group': experimental_data[column].mean()
        }
    return means

def compute_anova(control_data, experimental_data):
    anova_results = {}
    for column in control_data.columns:
        stat, pval = f_oneway(control_data[column], experimental_data[column])
        anova_results[column] = {'F-statistic': stat, 'p-value': pval}
    return anova_results

def cohens_d(control_group, experimental_group):
    diff = experimental_group.mean() - control_group.mean()
    pooled_std = np.sqrt((experimental_group.std() ** 2 + control_group.std() ** 2) / 2)
    d = diff / pooled_std
    return d

def compute_cohens_d(control_data, experimental_data):
    cohens_d_results = {}
    for column in control_data.columns:
        d = cohens_d(control_data[column], experimental_data[column])
        cohens_d_results[column] = d
    return cohens_d_results

def print_results(means, anova_results, cohens_d_results, exp_folder_name):
    print("Means:")
    for column, mean in means.items():
        print(f"{column}:\nMean Control Group = {mean['Mean_Control_Group']}\nMean {exp_folder_name} Group = {mean['Mean_Experimental_Group']}")
    print("\nANOVA Results:")
    for column, result in anova_results.items():
        print(f"{column}: F-statistic = {result['F-statistic']}, p-value = {result['p-value']}")
    print("\nCohen's d:")
    for column, d in cohens_d_results.items():
        print(f"{column}: Cohen's d = {d}")


def compute_intra_group_anova_cohensd(data, group):
    """
    Computes ANOVA and Cohen's d within a group for traits before and after writing.

    Parameters:
    - data (DataFrame): The dataset containing both before and after writing traits for a specific group.
    - group (str): The group to filter data by.

    Returns:
    - dict: A dictionary containing ANOVA results and Cohen's d values for each trait.
    """
    intra_group_results = {}
    filtered_data = data[data['Group'] == group]
    
    # Define the big five personality traits globally
    big_five = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']


    for trait in big_five:
        before_column = trait
        after_column = f'Writing_{trait}'
        
        if before_column in filtered_data.columns and after_column in filtered_data.columns:
            before_data = filtered_data[before_column]
            after_data = filtered_data[after_column]

            # Compute ANOVA
            stat, pval = f_oneway(before_data, after_data)
            # Compute Cohen's d
            d = cohens_d(before_data, after_data)
            # Compute means
            mean_before = before_data.mean()
            mean_after = after_data.mean()

            intra_group_results[trait] = {
                'F-statistic': stat,
                'p-value': pval,
                "Cohen's d": d,
                'Mean Before': mean_before,
                'Mean After': mean_after
            }

    return intra_group_results

def print_intra_group_results(intra_group_results, group):
    """
    Prints ANOVA results and Cohen's d values within a group for traits before and after writing.

    Parameters:
    - intra_group_results (dict): The results to print.
    - group (str): The group these results pertain to.
    """
    print(f"\nIntra-Group ANOVA and Cohen's d Results for {group} Group:")
    for trait, results in intra_group_results.items():
        cohen_d_value = results["Cohen's d"]
        print(f"{trait}: F-statistic = {results['F-statistic']:.4f}, p-value = {results['p-value']:.4f}, Cohen's d = {cohen_d_value:.4f}, , Mean Before = {results['Mean Before']:.4f}, Mean After = {results['Mean After']:.4f}")

# #-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# #------------------------------- Between Control and Experimental, before and after writing --------------------------------------------------------------------------------
# #----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Uncomment code to run
# Change: "exp_folder_name" and "group" to run for different folders and subject groups

# File paths
file_path = "../output/Control/BFI_Story_data.csv"
exp_folder_name = "CONTROL"
group = "Creative"

# Load data
data = pd.read_csv(file_path, delimiter=',')

# Perform intra-group ANOVA and Cohen's d computation
intra_group_results = compute_intra_group_anova_cohensd(data, group)

# Print intra-group ANOVA and Cohen's d results
print_intra_group_results(intra_group_results, group)

# #-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# #------------------------------ Between Control and Experimental, after writing --------------------------------------------------------------------------------
# #----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Uncomment code to run
# Change: "exp_folder_name" and "group" to run for different folders and subject groups


# # File paths
# control_file = "../output/Control/BFI_Story_data.csv"
# experimental_file = "../output/CREAANA/BFI_Story_data.csv"

# exp_folder_name = "2_CREAANA"
# group = "Analytic"

# # Process the files and ensure they have the same columns
# control_data, experimental_data = process_and_align_data(control_file, experimental_file, group)

# # Compute ANOVA, Cohen's d, and means
# anova_results = compute_anova(control_data, experimental_data)
# cohens_d_results = compute_cohens_d(control_data, experimental_data)
# means = compute_means(control_data, experimental_data)

# # Print results for between-group comparison
# print("Between-Group Comparison Results:")
# print_results(means, anova_results, cohens_d_results, exp_folder_name)



#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------ Control file: Creative ~ Analytic, both before writing --------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Uncomment code to run


# # Function to process a list of data files
# def process_data_files(file_paths, groups):
#     all_data = pd.DataFrame()
#     for file_path in file_paths:
#         data = pd.read_csv(file_path, delimiter=',')
#         filtered_data = data[data['Group'].isin(groups)]
#         all_data = pd.concat([all_data, filtered_data])
#     return all_data

# def compute_anova(data, groups, columns):
#     anova_results = {}
#     for column in columns:
#         group_data = [data[data['Group'] == group][column] for group in groups]
#         stat, pval = f_oneway(*group_data)
#         anova_results[column] = {'F-statistic': stat, 'p-value': pval}
#     return anova_results

# def print_anova_results(anova_results, groups, columns):
#     print("ANOVA Results:")
#     for column in columns:
#         result = anova_results[column]
#         print(f"Comparing '{groups[0]}' and '{groups[1]}' for '{column}':\n F-statistic = {result['F-statistic']}\n p-value = {result['p-value']}\n")

# # Columns and groups
# columns = ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"]
# groups = ["Creative", "Analytic"]

# # File paths
# control_file = "../output/Control/BFI_Story_data.csv"

# file_paths = [file]

# # Process the files
# data = process_data_files(file_paths, groups)

# # Compute ANOVA for each column
# anova_results = compute_anova(data, groups, columns)

# # Print ANOVA results
# print_anova_results(anova_results, groups, columns)


