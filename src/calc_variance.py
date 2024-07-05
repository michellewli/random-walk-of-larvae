import pandas as pd
import numpy as np

'''
# Read the CSV file into a DataFrame
filename = input("CSV file name: ")
df = pd.read_csv(filename)

# List to store the standard deviations of the ratios
std_devs = []

# Initialize variables to track current 'track' value and ratios
current_track = df.iloc[0]['track']
ratios = []

# Iterate through the DataFrame row by row
for index, row in df.iterrows():
    if row['track'] != current_track:
        # Calculate the standard deviation for the previous track group
        if len(ratios) > 1:  # Ensure at least two data points for sample std deviation
            std_dev = np.std(ratios, ddof=1)  # Sample standard deviation
            std_devs.append(std_dev)
            print(f"Track {current_track} - Standard Deviation: {std_dev}")

        # Reset for the new track group
        current_track = row['track']
        ratios = []

    # Append ratio of runL to runT if runT is not zero
    if row['runT'] != 0:
        ratios.append(row['runL'] / row['runT'])

# Calculate and store the standard deviation for the last group
if len(ratios) > 1:  # Ensure at least two data points for sample std deviation
    std_dev = np.std(ratios, ddof=1)  # Sample standard deviation
    std_devs.append(std_dev)
    print(f"Track {current_track} - Standard Deviation: {std_dev}")

# Calculate the mean and standard deviation of the standard deviations
if std_devs:
    mean_std_dev = np.mean(std_devs)
    std_dev_std_dev = np.std(std_devs, ddof=1) if len(std_devs) > 1 else 0.0

    print(f"List of standard deviations: {std_devs}")
    print(f"Mean of standard deviations: {mean_std_dev}")
    print(f"Standard deviation of standard deviations: {std_dev_std_dev}")
else:
    print("No valid standard deviations calculated.")
'''


import pandas as pd
import numpy as np

# Read the CSV file into a DataFrame
filename = input("CSV file name: ")
df = pd.read_csv(filename)

# Lists to store the means and standard deviations of the ratios for each track
means = []
std_devs = []

# Initialize variables to track current 'track' value and ratios
current_track = df.iloc[0]['track']
ratios = []

# Iterate through the DataFrame row by row
for index, row in df.iterrows():
    if row['track'] != current_track:
        # Calculate the mean and standard deviation for the previous track group
        if len(ratios) > 1:  # Ensure at least two data points for sample std deviation
            track_mean = np.mean(ratios)
            track_std_dev = np.std(ratios, ddof=1)  # Sample standard deviation
            means.append(track_mean)
            std_devs.append(track_std_dev)
            print(f"Track {current_track} - Mean: {track_mean}, Standard Deviation: {track_std_dev}")

        # Reset for the new track group
        current_track = row['track']
        ratios = []

    # Append ratio of runL to runT if runT is not zero
    if row['runT'] != 0:
        ratios.append(row['runL'] / row['runT'])

# Calculate and store the mean and standard deviation for the last group
if len(ratios) > 1:  # Ensure at least two data points for sample std deviation
    track_mean = np.mean(ratios)
    track_std_dev = np.std(ratios, ddof=1)  # Sample standard deviation
    means.append(track_mean)
    std_devs.append(track_std_dev)
    print(f"Track {current_track} - Mean: {track_mean}, Standard Deviation: {track_std_dev}")

# Calculate the average mean and average standard deviation of all tracks
if means and std_devs:
    average_mean = np.mean(means)
    average_std_dev = np.mean(std_devs)

    print(f"List of means: {means}")
    print(f"List of standard deviations: {std_devs}")
    print(f"Average of means: {average_mean}")
    print(f"Average of standard deviations: {average_std_dev}")
else:
    print("No valid means or standard deviations calculated.")