# Include "system("sandbox_plots.py")" in main.cpp

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

def plot_distribution(filename, plot_log=True):
    input_path = os.path.join('out', filename)

    # Read numbers from the file
    with open(input_path, 'r') as f:
        numbers = [float(line.strip()) for line in f]

    # If filename contains "alpha" or "beta", set x-axis to log scale
    if plot_log and ('alpha' in filename or 'beta' in filename):
        plt.xscale('log')

        # Create logarithmic bins
        bin_edges = np.logspace(np.log10(min(numbers)), np.log10(max(numbers)), num=50)
        plt.hist(numbers, bins=bin_edges, edgecolor='black', alpha=0.7)

    else:
        # Plotting the histogram
        plt.hist(numbers, bins='auto', edgecolor='black', alpha=0.7)

    plt.xlabel('Sampled Parameter Value')
    plt.ylabel('Frequency')

    # Save the plot as a JPEG file
    if plot_log and ('alpha' in filename or 'beta' in filename):
        output_path = os.path.join('out', 'plot_{}_log.jpg'.format(filename.split('_')[0]))
    else:
        output_path = os.path.join('out', 'plot_{}.jpg'.format(filename.split('_')[0]))
    plt.savefig(output_path, format='jpeg')
    plt.close()

if __name__ == '__main__':
    plot_distribution('alpha_samples.txt')
    plot_distribution('beta_samples.txt')
    plot_distribution('epsilon_samples.txt')
    plot_distribution('lambda_samples.txt')
    plot_distribution('alpha_samples.txt', plot_log=False)
    plot_distribution('beta_samples.txt', plot_log=False)