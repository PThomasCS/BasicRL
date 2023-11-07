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

def plot(num_samples):
    # Make a list of CSV files with results
    print(f"Working directory: {os.getcwd()}")
    csv_files = [file for file in os.listdir('out/') if file.endswith('.csv') and file.startswith('results_summary')]
    
    for file in csv_files:
    # Extract the filename without extension
        filename = os.path.splitext(file)[0]

        # Check if the plot for the current file already exists
        output_path = os.path.join('out/', 'plot_{}.jpg'.format(filename))
        if os.path.exists(output_path):
            print('Plot already exists for file:', file)
            continue
        
        # Read the CSV file
        csv_path = os.path.join('out/', file)
        df = pd.read_csv(csv_path)

        # Create a new dataframe contating averages of every n points (n = num_samples)
        avg_data = []

        for i in range(0, len(df), num_samples):
            data = df.iloc[i:i+num_samples]
            episode_num = data['Episode'].iloc[-1]
            avg_return = data['Average Discounted Return'].mean()    
            avg_std_error = data['Standard Error'].mean()  # Mean standard error
            avg_data.append({'Episode': episode_num, 'Average Discounted Return': avg_return, 'Standard Error': avg_std_error})

        df_avg = pd.DataFrame(avg_data)
        
        # Plotting
        sns.set(style='whitegrid')
        sns.lineplot(data=df_avg, x='Episode', y='Average Discounted Return')
        plt.fill_between(df_avg['Episode'], df_avg['Average Discounted Return'] - df_avg['Standard Error'], df_avg['Average Discounted Return'] + df_avg['Standard Error'], color='tab:blue', alpha=0.3)
        plt.xlabel('Episode')
        plt.ylabel('Average Discounted Return')
        
        # Save the plot as a JPEG file
        output_path = os.path.join('out/', 'plot_{}.jpg'.format(filename))
        plt.savefig(output_path, format='jpeg')
        plt.close()  # Close the plot to free up memory
        
        # print('Plot saved for file:', file)
    

if __name__ == "__main__":
    plot(num_samples=5)
