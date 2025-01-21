import pandas as pd
import os

import matplotlib.pyplot as plt

def main():
    dfs = []
    # Load from entropy_data (iterate all files)
    for f in os.listdir("entropy_data_1000"):
        print(f'Loading: {os.path.join("entropy_data_1000", f)}')
        df = pd.read_csv(os.path.join("entropy_data_1000", f))
        dfs.append(df)
    # Plot all entropy in the same plot
    for df in dfs:
        plt.plot(df["Entropy"], color='blue', alpha=0.1)
        #print(df["Entropy"])
    # Plot entropy average
    entropy_avg = sum([df["Entropy"] for df in dfs]) / len(dfs)
    plt.plot(entropy_avg, color='blue')
    plt.show()
    # Plot all accuracy in the same plot
    for df in dfs:
        plt.plot(df["NormalAccuracy"], color='red', alpha=0.1)
        # Plot accuracy with entropy
        plt.plot(df["EntropyAccuracy"], color='green', alpha=0.1)
    # Plot accuracy average
    acc_avg = sum([df["NormalAccuracy"] for df in dfs]) / len(dfs)
    plt.plot(acc_avg, color='red')
    acc_avg = sum([df["EntropyAccuracy"] for df in dfs]) / len(dfs)
    plt.plot(acc_avg, color='green')
    plt.grid()
    # Set ylim to 0.92, 1.0
    plt.ylim(0.92, 1.0)
    plt.show()
    # Cummulative statistics for accuracy with entropy (searching for maximal and minimal accuracy stdev)
    print("Accuracy with entropy")
    stdevs = [df["EntropyAccuracy"].iloc[150:].std() for df in dfs]
    maxs = [df["EntropyAccuracy"].max() for df in dfs]
    # Get min after a certain epoch
    mins = [df["EntropyAccuracy"].iloc[100:].min() for df in dfs]
    print(f'Average stdev: {sum(stdevs) / len(stdevs)}')
    print(f"Average maximal accuracy: {sum(maxs)/len(maxs)}")
    print(f"Average accuracy: {sum(mins) / len(mins)}")
    print(f"Minimal accuracy: {min(mins)}")
    # Cummulative statistics for accuracy without entropy (searching for maximal and minimal accuracy stdev)
    print("Accuracy without entropy")
    stdevs = [df["NormalAccuracy"].iloc[150:].std() for df in dfs]
    maxs = [df["NormalAccuracy"].max() for df in dfs]
    mins = [df["NormalAccuracy"].iloc[100:].min() for df in dfs]
    print(f'Average stdev: {sum(stdevs) / len(stdevs)}')
    print(f"Average maximal accuracy: {sum(maxs)/len(maxs)}")
    print(f"Average accuracy: {sum(mins) / len(mins)}")
    print(f"Minimal accuracy: {min(mins)}")








if __name__ == "__main__":
    main()
