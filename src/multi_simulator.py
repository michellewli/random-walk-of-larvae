import matplotlib.pyplot as plt
import numpy as np
import random
import csv
from scipy.stats import truncnorm
from datetime import datetime
import mpld3
import os
from typing import List, Tuple

# Helper function to generate truncated normal values
def get_truncated_normal(mean: float, std_dev: float, lower_bound: float = 0) -> float:
    a = (lower_bound - mean) / std_dev  # Lower bound in standard normal terms
    return truncnorm(a, float('inf'), loc=mean, scale=std_dev).rvs()

class LarvaWalker:
    def __init__(self, N: int, T: int, time_step: float, bias: float = 0):
        self.N = N
        self.T = T
        self.time_step = time_step
        self.bias = bias
        self.turning_probability = (N / T) / time_step
        self.x, self.y = 0.0, 0.0
        self.angle = random.uniform(0, 2 * np.pi)  # initial angle in radians, 0 means facing right
        self.x_positions = [self.x]  # starting x position
        self.y_positions = [self.y]  # starting y position
        self.turn_points_x = []  # x coordinates of turn points
        self.turn_points_y = []  # y coordinates of turn points
        self.num_turns = 0
        self.num_runs = 0
        self.speeds = []
        self.angles = [self.angle]  # initialize with the starting angle in radians
        self.times = [0]  # start time at 0
        self.timestamps = [0]  # track timestamps including turn times
        self.turn_times = []  # track individual turn times
        self.drift_rates = []  # track drift rates

    def simulate(self) -> Tuple[List[float], List[float], List[float], List[float]]:
        timestamp = 0
        lambda_ = 10
        drift_rate = np.random.exponential(scale=1/lambda_)
        self.drift_rates.append(drift_rate)
        direction = True  # automatically will choose to drift right, maybe can incorporate handedness to variable

        while timestamp <= self.T:
            timestamp += self.time_step
            turn_or_not = np.random.uniform(0.0, 1.0)  # random float to compare with probability of turning

            v0 = get_truncated_normal(mean=2.9095, std_dev=0.7094)  # speed of larva in px/s
            self.speeds.append(v0)

            if turn_or_not < self.turning_probability:  # will turn either left or right
                # First, move to the current position based on the previous angle
                self.x += v0 * np.cos(self.angle) * self.time_step
                self.y += v0 * np.sin(self.angle) * self.time_step

                # Record turn point
                self.turn_points_x.append(self.x)
                self.turn_points_y.append(self.y)

                # Determine direction of turn
                prob_left_right = 0.5 if self.bias == 0 else self.bias  # probability of left or right is 1/2 or bias %
                left_or_right = random.random()  # random number to compare with probability of going left or right

                # Angle at which larva will turn wrt the direction it's already facing
                reference_angle = np.radians(get_truncated_normal(mean=66.501, std_dev=36.874))  # angle in radians
                if left_or_right < prob_left_right:  # if random number is <0.5, go left
                    self.angle += reference_angle  # turn left by reference angle
                    direction = False
                else:  # if random number is >=0.5, go right
                    self.angle -= reference_angle  # turn right by reference angle
                    direction = True

                self.angles.append(self.angle % (2 * np.pi))  # Add the new angle after turning in radians
                self.num_turns += 1

                # pause time for turn
                mean = 4.3
                turn_time = np.random.exponential(scale=mean)
                self.turn_times.append(turn_time)
                timestamp += turn_time

                self.timestamps.append(timestamp)  # Add the current timestamp including turn time
                # Append new position to the lists
                self.x_positions.append(self.x)
                self.y_positions.append(self.y)

                # pick a drift rate (dtheta/dt)
                drift_rate = np.random.exponential(scale=1/lambda_)  # resets after each turn
                self.drift_rates.append(drift_rate)

            else:  # will go straight (with slight drift)
                self.num_runs += 1

                drift_angle = drift_rate * self.time_step  # Calculate drift angle based on drift rate and time step

                if direction:
                    self.angle += drift_angle
                else:
                    self.angle -= drift_angle
                self.x += v0 * np.cos(self.angle) * self.time_step
                self.y += v0 * np.sin(self.angle) * self.time_step
                self.x_positions.append(self.x)
                self.y_positions.append(self.y)
                self.angles.append(self.angle % (2 * np.pi))  # Update the angle list with the current angle
                self.timestamps.append(timestamp)  # Update the timestamps list with the current time

        return self.x_positions, self.y_positions, self.turn_points_x, self.turn_points_y

def main():
    N = int(input("Number of turns (N): "))  # number of turns
    T = int(input("Total time for experiment (in seconds): "))  # total time in seconds
    time_step = float(input("Time step (in seconds): "))  # in seconds
    num_walkers = int(input("How many larvae? "))  # number of larvae
    bias = float(input("Left or right handed (decimal [0.0, 1.0]: "))  # <0.5=left, >0.5=right, 0.5="normal"

    walkers = []
    all_speeds = []
    all_angles = []
    all_drift_rates = []
    for _ in range(num_walkers):
        walker = LarvaWalker(N, T, time_step, bias)
        walker.simulate()
        walkers.append(walker)
        all_speeds.extend(walker.speeds)
        all_angles.extend(walker.angles[1:])  # Exclude the initial angle from the list of angles for histogram
        all_drift_rates.extend(walker.drift_rates)

    colors = plt.get_cmap('tab20', num_walkers)  # Use a colormap to generate distinct colors

    plt.figure(figsize=(10, 6))

    # Prepare CSV file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('data', exist_ok=True)
    csv_filename = f'data/larva_data_{timestamp}.csv'

    with open(csv_filename, mode='w', newline='') as csv_file:
        fieldnames = ['Column1', 'set', 'expt', 'track', 'time0', 'reoYN', 'runQ', 'runL', 'runT', 'runX', 'reo#HS',
                      'reoQ1', 'reoQ2', 'reoHS1', 'runQ0', 'runX0', 'runY0', 'runX1', 'runY1']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for i, walker in enumerate(walkers):
            prev_angle = walker.angles[0]
            prev_x = walker.x_positions[0]
            prev_y = walker.y_positions[0]
            prev_timestamp = walker.timestamps[0]
            for j in range(1, len(walker.x_positions)):
                if j >= len(walker.angles) or j >= len(walker.timestamps) or j >= len(walker.speeds):
                    break
                current_angle = walker.angles[j]
                runQ = current_angle - prev_angle
                runL = walker.speeds[j] * walker.time_step
                runT = walker.timestamps[j] - prev_timestamp - (walker.turn_times[j - 1] if j - 1 < len(walker.turn_times) else 0)  # Total time w/o turn time
                runX0 = prev_x
                runY0 = prev_y
                runX1 = walker.x_positions[j]
                runY1 = walker.y_positions[j]
                row = {
                    'Column1': '',
                    'set': 1,
                    'expt': 1,
                    'track': i + 1,
                    'time0': walker.timestamps[j],  # Use the timestamp including turn time for time0
                    'reoYN': 1,
                    'runQ': runQ,
                    'runL': runL,
                    'runT': runT,
                    'runX': runX1,
                    'reo#HS': np.random.choice([0, 1, 2, 3, 4, 5], p=[0.05, 0.7, 0.1, 0.05, 0.05, 0.05]),
                    'reoQ1': prev_angle,
                    'reoQ2': current_angle,
                    'reoHS1': np.random.choice([0, 1, 2, 3, 4, 5], p=[0.05, 0.7, 0.1, 0.05, 0.05, 0.05]),
                    'runQ0': prev_angle,
                    'runX0': runX0,
                    'runY0': runY0,
                    'runX1': runX1,
                    'runY1': runY1
                }
                writer.writerow(row)
                prev_angle = current_angle
                prev_x = walker.x_positions[j]
                prev_y = walker.y_positions[j]
                prev_timestamp = walker.timestamps[j]

            # Plot trajectory
            plt.plot(walker.x_positions, walker.y_positions, label=f'Larva {i + 1}', color=colors(i))
            plt.scatter(walker.turn_points_x, walker.turn_points_y, s=10, color=colors(i))

    plt.title('Larvae Random Walk with Turning Points')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=1, fontsize='small')  # Place legend inside the plot
    plt.grid(True)

    # Save interactive plot as HTML using mpld3
    os.makedirs('simulations', exist_ok=True)
    interactive_filename = f'simulations/larva_path_{timestamp}.html'
    mpld3.save_html(plt.gcf(), interactive_filename)

    # Collect all turn_time values
    all_turn_times = []
    for walker in walkers:
        all_turn_times.extend(walker.turn_times)

    # Plot histogram of turn_time values
    fig, axs = plt.subplots(figsize=(8, 6))
    axs.hist(all_turn_times, bins=30, color='orange', edgecolor='black', alpha=0.7)
    axs.set_title('Histogram of Turn Times')
    axs.set_xlabel('Turn Time (seconds)')
    axs.set_ylabel('Frequency')
    axs.grid(True)
    plt.show()

    # Save histogram as HTML using mpld3
    os.makedirs('histograms', exist_ok=True)
    hist_interactive_filename = f'histograms/turn_time_histogram_{timestamp}.html'
    mpld3.save_html(fig, hist_interactive_filename)

    # Plot histograms of all speeds, angles, and drift rates
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].hist(all_speeds, bins=30, color='blue', edgecolor='black', alpha=0.7)
    axs[0].set_title('Histogram of Speeds')
    axs[0].set_xlabel('Speed (px/s)')
    axs[0].set_ylabel('Frequency')

    axs[1].hist(all_angles, bins=30, color='green', edgecolor='black', alpha=0.7)
    axs[1].set_title('Histogram of Angles')
    axs[1].set_xlabel('Angle (radians)')
    axs[1].set_ylabel('Frequency')

    axs[2].hist(all_drift_rates, bins=30, color='red', edgecolor='black', alpha=0.7)
    axs[2].set_title('Histogram of Drift Rates')
    axs[2].set_xlabel('Drift Rate (radians/s)')
    axs[2].set_ylabel('Frequency')

    plt.show()

    hist_interactive_filename = f'histograms/larva_histograms_{timestamp}.html'
    mpld3.save_html(fig, hist_interactive_filename)


if __name__ == "__main__":
    main()
