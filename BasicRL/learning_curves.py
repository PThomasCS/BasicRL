# Include "system("learning_curves.py")" in main.cpp

import matplotlib.pyplot as plt
import pandas as pd
 

def plot_graph():
    df = pd.read_csv("out/results.csv")
    plt.errorbar(df['Episode'], df['Average Discounted Return'], yerr=df['Standard Error'], ecolor=['grey'])
    plt.xlabel('Episode')
    plt.ylabel('Average Discounted Return')
    plt.savefig("out/plot.jpg", format='jpeg')
    plt.show()


# TO-DO: add new graphs (e.g., cumulative reward, etc.)

# Plot learning curve showing cumulative actions taken so far on the x-axis and number of episodes on the y-axis
# If the agent is learning, the graph should have an increasing slope

def plot():
    x = []

    # x-axis data showing cumulative actions taken
    with open('cmake-build-debug/actions.txt', 'r') as f:
        for line in f:
            data = line.strip()  # Strip whitespace
            x.append(float(data))  # Convert to float

    # y-axis data showing number of episodes
    y = list(range(1, len(x) + 1))

    plt.plot(x, y)
    plt.ticklabel_format(style='plain')
    plt.xlabel('# Actions Taken')
    plt.ylabel('# Episodes')
    plt.show()


def plot_mc():
    y = []

    # y-axis data showing the number of steps needed by the agent to reach the goal state
    with open('cmake-build-debug/returns.txt', 'r') as f:
        for line in f:
            data = line.strip()  # Strip whitespace
            y.append(float(data) * (-1))  # Convert to float

    # x-axis data showing number of episodes
    x = list(range(1, len(y) + 1))

    plt.plot(x, y)
    plt.ticklabel_format(style='plain')
    plt.xlabel('# Episodes')
    plt.ylabel('# Steps to reach goal')
    plt.show()


if __name__ == "__main__":
    plot_graph()
    # plot_mc()
