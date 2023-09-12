# Include "system("sandbox_plots.py")" in main.cpp

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def plot_distribution(filename):
    input_path = os.path.join('out', filename)

    # Read numbers from the file
    with open(input_path, 'r') as f:
        numbers = [float(line.strip()) for line in f]

    # If filename contains "alpha" or "beta", set x-axis to log scale
    if 'alphaw' in filename or 'betaw' in filename:
        plt.xscale('log')

    # Plotting the histogram
    plt.hist(numbers, bins='auto', edgecolor='black', alpha=0.7)

    plt.title('Distribution of Numbers')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Save the plot as a JPEG file
    output_path = os.path.join('out', 'plot_{}.jpg'.format(filename.split('_')[0]))
    plt.savefig(output_path, format='jpeg')
    plt.close()

if __name__ == '__main__':
    plot_distribution('alpha_samples.txt')
    plot_distribution('beta_samples.txt')
    plot_distribution('epsilon_samples.txt')
    plot_distribution('lambda_samples.txt')