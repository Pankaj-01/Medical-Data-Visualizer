import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv(r"/workspace/boilerplate-mean-variance-standard-deviation-calculator/Medical Data Visualizer/medical_examination.csv")

# Add 'overweight' column
df['overweight'] = np.where((df['weight'] / np.square(df['height'] / 100)) > 25, 1, 0)

# Normalize data by making 0 always good and 1 always bad.
df['cholesterol'] = np.where(df['cholesterol'] == 1, 0, 1)
df['gluc'] = np.where(df['gluc'] == 1, 0, 1)

# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, vmin=0, vmax=0.25, fmt='.1f', linewidth=1, annot=True, square=True, mask=mask, cbar_kws={'shrink': .82})

    # Display the plot inline (comment out if not needed)
    plt.show()

    # Save the figure (optional)
    fig.savefig('heatmap.png')

    return fig

draw_heat_map()
