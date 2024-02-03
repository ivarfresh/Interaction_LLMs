"""
This code creates the violin plots based on the Top5 SpearmanR correlations for each BFI trait.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Replace with your own Top5 SpearmanR correlation
data_before = {
    "Extraversion": {"posemo": 0.696, "anger": -0.656, "incl": 0.636, "discrep": -0.620, "tentat": -0.586},
    "Agreeableness": {"incl": 0.687, "posemo": 0.672, "discrep": -0.658, "anger": -0.611, "tentat": -0.577},
    "Conscientiousness": {"posemo": 0.676, "anger": -0.666, "incl": 0.657, "discrep": -0.621, "ppron": -0.560},
    "Neuroticism": {"discrep": -0.468, "insight": -0.414, "incl": 0.365, "relig": 0.349, "posemo": 0.342},
    "Openness": {"discrep": -0.727, "posemo": 0.679, "incl": 0.659, "anger": -0.650, "pronoun": -0.637}
}
data_after = {
    "Extraversion": {"posemo": -0.2319, "anger": 0.2727, "incl": -0.0685, "discrep": 0.3633, "tentat": 0.2280},
    "Agreeableness": {"incl": -0.1749, "posemo": -0.2044, "discrep": 0.3083, "anger": 0.2439, "tentat": 0.1383},
    "Conscientiousness": {"posemo": -0.2263, "anger": 0.2892, "incl": -0.1855, "discrep": 0.3236, "ppron": 0.4264},
    "Neuroticism": {"discrep": 0.1402, "insight": 0.0513, "incl": -0.0057, "relig": 0.0199, "posemo": -0.0168},
    "Openness": {"discrep": 0.3211, "posemo": -0.2594, "incl": -0.1260, "anger": 0.2850, "pronoun": 0.2754}
}

# Converting data to DataFrames
df_before = pd.DataFrame(data_before)
df_after = pd.DataFrame(data_after)

# Merging and transforming data for plotting
df_plot = pd.concat([df_before.stack(), df_after.stack()], axis=1)
df_plot.columns = ["Before", "After"]
df_plot.reset_index(inplace=True)
df_plot.columns = ["Term", "Trait", "Before", "After"]
df_violin_combined = pd.melt(df_plot, id_vars=['Term', 'Trait'], value_vars=['Before', 'After'])
df_violin_combined.columns = ['Term', 'Trait', 'Condition', 'Correlation']

# Updating the condition names
df_violin_combined['Condition'] = df_violin_combined['Condition'].replace({'Before': 'No Interaction', 'After': 'Interaction'})

# Creating the violin plot
plt.figure(figsize=(12, 6))
sns.violinplot(x='Trait', y='Correlation', hue='Condition', data=df_violin_combined, split=True, palette=["#1f77b4", "red"], zorder=1)
plt.ylabel("")  # Removing the y-axis label
plt.xlabel("")  # Removing the x-axis label
plt.xticks(rotation=15, fontsize=20)  # Adjusting x-axis ticks
plt.yticks(fontsize=20)               # Adjusting y-axis ticks

# Increase the height of x ticks
plt.ylim(-2.0, 2.0) 

# Adding light gray lines at all x-ticks with a lower zorder
for tick in plt.gca().get_yticks():
    plt.axhline(y=tick, color='gray', linestyle='dashdot', linewidth=0.4, zorder=0)

sns.despine(left = True)  # Removing the top and right spines

# Adjust the legend position
plt.legend(fontsize=20, loc='upper left', bbox_to_anchor=(0.29, 1.25))

plt.show()




