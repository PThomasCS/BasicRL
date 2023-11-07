# Include "system("rank_params.py")" in main.cpp

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np

def rank(weight_area=0.5, weight_last_avg_return=0.5):
    # # Get the absolute path to the directory where the script is located
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    #
    # # Define the relative path to the 'out/' directory from the script's directory
    # out_directory_path = os.path.join(script_dir, 'out/')
    #
    # # Print working directory
    # print(f"Script directory: {script_dir}")
    print(f"Working directory: {os.getcwd()}")
    # print(f"Out directory: {out_directory_path}")

    # Make a list of all "results_summary" csv files
    csv_files = [file for file in os.listdir('out/') if file.endswith('.csv') and file.startswith('results_summary')]

    # Empty lists to store filename, its score, area and last average return
    rankings = [] 
    areas = []
    last_avg_returns = []
    
    # Go through all files to collect all areas and last average returns for normalization
    for file in csv_files:
        csv_path = os.path.join('out/', file)
        df = pd.read_csv(csv_path)
        last_avg_return = df['Average Discounted Return'].iloc[-1]
        area = np.trapz(df['Average Discounted Return'], df['Episode'])
        areas.append(area)
        last_avg_returns.append(last_avg_return)

    # Normalize data
    min_area, max_area = min(areas), max(areas)
    min_last_avg_return, max_last_avg_return = min(last_avg_returns), max(last_avg_returns)
    
    normalized_areas = [(x - min_area) / (max_area - min_area) for x in areas]
    normalized_last_avg_returns = [(x - min_last_avg_return) / (max_last_avg_return - min_last_avg_return) for x in last_avg_returns]
    
    # Go through each file to compute score
    for i, file in enumerate(csv_files):
        filename = os.path.splitext(file)[0]
        score = weight_area * normalized_areas[i] + weight_last_avg_return * normalized_last_avg_returns[i]
        rankings.append((filename, score, areas[i], last_avg_returns[i]))
        
    # Sort rankings based on score
    sorted_rankings = sorted(rankings, key=lambda x: x[1], reverse=True)
    
    # Save to .txt file
    with open('out/best/rankings.txt', 'w') as f:
        for rank in sorted_rankings:
            f.write(f"{rank[0]}.csv, Score: {rank[1]}, Area: {rank[2]}, Last Return: {rank[3]}\n")
    
    # Plot top 3 files
    for i in range(min(3, len(sorted_rankings))):
        file = sorted_rankings[i][0] + ".csv"
        csv_path = os.path.join('out/', file)
        df = pd.read_csv(csv_path)
    
        sns.set(style='whitegrid')
        sns.lineplot(data=df, x='Episode', y='Average Discounted Return')
        plt.fill_between(df['Episode'], df['Average Discounted Return'] - df['Standard Error'], df['Average Discounted Return'] + df['Standard Error'], color='tab:blue', alpha=0.3)
        plt.xlabel('Episode')
        plt.ylabel('Average Discounted Return')
    
        print(f"Plotting file: {file}")  # Print the filename being plotted
    
        # Ensure a unique filename for each plot
        output_path = os.path.join('out/best', f'plot_rank_{i + 1}_{file.split(".")[0]}.jpg')
        plt.savefig(output_path, format='jpeg')
        plt.close()
        
        # Comment

if __name__ == "__main__":
    rank()