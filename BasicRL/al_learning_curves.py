import matplotlib.pyplot as plt


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
    plt.xlabel('# Actions Taken')
    plt.ylabel('# Episodes')
    plt.show()


if __name__ == "__main__":
    plot()
