import os

import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np

def create_heatmap(cka_matrix, epoch, save_path):
    # Create figure with higher DPI and larger size for better quality
    plt.figure(figsize=(12, 10), dpi=300)
    
    # Create a custom colormap: Dark blue to light yellow
    colors = [
        '#0d0887',  # Deep blue
        '#5a189a',  # Purple
        '#9c179e',  # Magenta
        '#ed7953',  # Orange
        '#fdca26',  # Yellow
        '#ffffcc'   # Light yellow (near white)
    ]
    n_bins = 256  # Number of color gradients
    custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
    
    # Set up the plot style
    sns.set_style("whitegrid")
    
    # Create the heatmap
    ax = sns.heatmap(
        cka_matrix,
        cmap=custom_cmap,
        vmin=0.1,
        vmax=1.0,  # Full CKA similarity range
        annot=False,
        square=True,
        cbar_kws={
            'label': 'CKA Similarity',
            'orientation': 'vertical',
            'fraction': 0.046,
            'pad': 0.04,
            'aspect': 30,
            'ticks': np.arange(0.1, 1.1, 0.1)  # Step size of 0.1
        }
    )
    
    # Customize the plot
    plt.title(f"CKA Similarity Between Experts (Epoch {epoch})", 
              pad=20, 
              fontsize=16, 
              fontweight='bold')
    
    # Add axis labels with improved readability
    plt.xlabel("Expert Index", fontsize=12, labelpad=10)
    plt.ylabel("Expert Index", fontsize=12, labelpad=10)
    
    # Adjust layout for better appearance
    plt.tight_layout()
    
    # Add borders to the heatmap
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color('black')
    
    # Save the figure
    plt.savefig(
        f"{save_path}/Heatmap_{epoch}.png",
        bbox_inches='tight',
        dpi=300,
        facecolor='white',
        edgecolor='none'
    )
    plt.close()
