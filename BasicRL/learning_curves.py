# Include "system("learning_curves.py")" in main.cpp

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
 
"""
def plot():
    df = pd.read_csv("out/results.csv")
    sns.set(style='whitegrid')
    sns.lineplot(data=df, x='Episode', y='Average Discounted Return')
    plt.fill_between(df['Episode'], df['Average Discounted Return'] - df['Standard Error'], df['Average Discounted Return'] + df['Standard Error'], color='tab:blue', alpha=0.3)
    plt.xlabel('Episode')
    plt.ylabel('Average Discounted Return')
    plt.savefig("out/plot.jpg", format='jpeg')
    plt.show()
"""

def plot():
    # Make a list of CSV files with results
    csv_files = [file for file in os.listdir('out/') if file.endswith('.csv')]  
    
    for file in csv_files:
    # Extract the filename without extension
        filename = os.path.splitext(file)[0]
        
        # Read the CSV file
        csv_path = os.path.join('out/', file)
        df = pd.read_csv(csv_path)
        
        # Plotting
        sns.set(style='whitegrid')
        sns.lineplot(data=df, x='Episode', y='Average Discounted Return')
        plt.fill_between(df['Episode'], df['Average Discounted Return'] - df['Standard Error'], df['Average Discounted Return'] + df['Standard Error'], color='tab:blue', alpha=0.3)
        plt.xlabel('Episode')
        plt.ylabel('Average Discounted Return')
        
        # Save the plot as a JPEG file
        output_path = os.path.join('out/', 'plot_{}.jpg'.format(filename))
        plt.savefig(output_path, format='jpeg')
        plt.close()  # Close the plot to free up memory
        
        print('Plot saved for file:', file)
    

if __name__ == "__main__":
    plot()
