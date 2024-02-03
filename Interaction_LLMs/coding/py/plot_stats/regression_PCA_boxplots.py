import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

def load_and_prepare_data(file_path, group_replace_map=None):
    """Loads and prepares BFI data."""
    data = pd.read_csv(file_path)
    if group_replace_map:
        data['Group'] = data['Group'].replace(group_replace_map)
    return data

def plot_trait_boxplots(data, trait_columns, group_column='Group', fig_size=(14, 7), custom_palette="Set2", title=None):
    """
    Plots boxplots for traits.
    """
    long_data = pd.melt(data, id_vars=[group_column], value_vars=trait_columns)
    plt.figure(figsize=fig_size)
    sns.boxplot(x='variable', y='value', hue=group_column, data=long_data, showfliers=True, palette=custom_palette)
    plt.xlabel('')
    plt.ylabel('')
    if title:  
        plt.title(title)  
    plt.show()

def perform_pca_and_plot(data, score_columns, group_column='Group', fig_size=(8, 8), custom_palette="Set2", title=None):
    """
    Performs PCA on provided scores and plots the result.
    """
    score_vectors = StandardScaler().fit_transform(data[score_columns])
    pca_result = PCA(n_components=2).fit_transform(score_vectors)
    pca_df = pd.DataFrame({'PC1': pca_result[:, 0], 'PC2': pca_result[:, 1], 'Group': data[group_column]})
    
    plt.figure(figsize=fig_size)
    sns.scatterplot(x='PC1', y='PC2', hue='Group', data=pca_df, palette=custom_palette, s=150, alpha=0.7)
    
    plt.xlabel('')
    plt.ylabel('')
    if title:  
        plt.title(title)  
    
    plt.show()


def logistic_regression_analysis(data, group_column='Group', score_columns=None):
    """Performs logistic regression analysis and prints CV accuracy."""
    if score_columns is None:
        score_columns = data.columns.difference([group_column, 'Subject_ID', 'Experimental_condition'])
    X = StandardScaler().fit_transform(data[score_columns])
    y = LabelEncoder().fit_transform(data[group_column])
    logreg_pipeline = make_pipeline(LogisticRegression(random_state=42))
    cv_scores = cross_val_score(logreg_pipeline, X, y, cv=10, scoring='accuracy')
    print(f'Average Accuracy (10-fold CV): {np.mean(cv_scores):.2%}')
    print(f'Standard Deviation of Accuracy: {np.std(cv_scores):.2%}')

def main():
    # Constants
    big_five = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']
    traits_after = [f'Writing_{trait}' for trait in big_five]
    group_replace_map = {'Analytic': 'Analytical'}

    # Paths for control group analysis
    # Replace with desired paths
    control_bfi_path = '../output/Data/0_CONTROL/BFI_Story_data.csv'
    control_liwc_path = '../output/Data/0_CONTROL/LIWC_results.csv'
    
    # Paths for experimental group analysis
    # Replace with desired paths
    experimental_bfi_path = "../output/Data/2_CREAANA/BFI_Story_data.csv"
    experimental_liwc_path = '../output/Data/2_CREAANA/LIWC_results.csv'
    
    # Experiment 1.A: Analyze BFI data for the control group
    print("Experiment 1.A: Control Group - BFI Traits")
    bfi_control = load_and_prepare_data(control_bfi_path, group_replace_map)
    plot_trait_boxplots(bfi_control, big_five, title='Control Group - BFI Traits')

    # Experiment 1.B: Analyze LIWC data for the control group
    print("\nExperiment 1.B: Control Group - LIWC PCA")
    liwc_control = load_and_prepare_data(control_liwc_path)
    perform_pca_and_plot(liwc_control, score_columns=liwc_control.columns[3:], title='Control Group - LIWC PCA')
    logistic_regression_analysis(liwc_control, 'Group', liwc_control.columns[3:])

    # Experiment 2.A: Analyze BFI data after interaction for the experimental group (CREAANA)
    print("\nExperiment 2.A: Experimental Group After Interaction - BFI Traits")
    bfi_exp_interaction = load_and_prepare_data(experimental_bfi_path, group_replace_map)
    plot_trait_boxplots(bfi_exp_interaction, traits_after, title='Experimental Group After Interaction - BFI Traits')

    # Experiment 2.B: Analyze LIWC data after interaction for the experimental group
    print("\nExperiment 2.B: Experimental Group - LIWC PCA After Interaction")
    liwc_interaction = load_and_prepare_data(experimental_liwc_path)
    perform_pca_and_plot(liwc_interaction, score_columns=liwc_interaction.columns[3:], title='Experimental Group - LIWC PCA After Interaction')
    logistic_regression_analysis(liwc_interaction, 'Group', liwc_interaction.columns[3:])

if __name__ == "__main__":
    main()



