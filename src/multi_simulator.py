import matplotlib.pyplot as plt
import numpy as np
import random
import csv
from scipy.stats import truncnorm
from datetime import datetime
import mpld3
import os


# Helper function to generate truncated normal values
def get_truncated_normal(mean: float, std_dev: float, lower_bound: float = 0) -> float:
    a = (lower_bound - mean) / std_dev  # Lower bound in standard normal terms
    return truncnorm(a, float('inf'), loc=mean, scale=std_dev).rvs()


# Helper function to normalize angle between -π and π
def normalize_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


class Larva:
    def __init__(self, N: int, T: int, time_step: float, turn_bias: float = 0, drift_bias: float = 0):
        self.N = N
        self.T = T
        self.time_step = time_step
        self.turn_bias = turn_bias
        self.drift_bias = drift_bias
        self.turning_probability = (N / T) / time_step
        self.x, self.y = (2550 / 2), (1950 / 2)
        self.angle = normalize_angle(random.uniform(0, 2 * np.pi))  # initial angle in radians
        self.x_positions = [self.x]  # starting x position
        self.y_positions = [self.y]  # starting y position
        self.plot_x_positions = [self.x]  # starting x position
        self.plot_y_positions = [self.y]  # starting y position
        self.turn_points_x = [self.x]  # x coordinates of turn points
        self.turn_points_y = [self.y]  # y coordinates of turn points
        self.num_turns = 0
        self.num_runs = 0
        self.speeds = []
        self.angles = [self.angle]  # initialize with the starting angle in radians
        self.drift_angles = []  # angle that the larva faces after finishing a sequence of drifts, Q1
        self.plot_angles = [self.angle]
        self.times = [0]  # start time at 0
        self.timestamps = [0]  # track timestamps including turn times
        self.plot_timestamps = [0]
        self.turn_times = []  # track individual turn times
        self.runL_distances = []  # track distances between turns
        self.last_move = "D"  # Initialize last move as drift

    def simulate(self):
        timestamp = 0
        lambda_ = 10
        drift_rate = np.random.exponential(scale=1 / lambda_)
        drift_left_or_right = random.random()  # picks a random number from [0.0, 1.0)

        v0 = get_truncated_normal(mean=2.847191967574795, std_dev=0.3518021717781707)  # speed of larva in px/s
        stdevi = get_truncated_normal(mean=0.3518021717781707, std_dev=0.1622577507017134)

        # Variable to accumulate distance during drift
        drift_distance_accumulator = 0.0
        drift_start_x = self.x
        drift_start_y = self.y
        moves = []

        while timestamp <= self.T:
            timestamp += self.time_step
            turn_or_not = np.random.uniform(0.0, 1.0)  # random float to compare with probability of turning

            # Prevent consecutive turns
            if self.last_move == "T":
                turn_or_not = 1.0  # Force drift if last move was a turn

            if turn_or_not < self.turning_probability:  # will turn either left or right
                moves.append("T")
                self.angle = normalize_angle(self.angle)  # Normalize the angle
                self.drift_angles.append(self.angle)  # Add the angle faced before turning in radians

                # Calculate drift distance before turn
                if drift_distance_accumulator > 0:
                    self.runL_distances.append(drift_distance_accumulator)
                drift_distance_accumulator = 0.0  # Reset accumulator for the next segment

                # First, move to the current position based on the previous angle
                self.x += v0 * np.cos(self.angle) * self.time_step
                self.y += v0 * np.sin(self.angle) * self.time_step

                # Record turn point
                self.turn_points_x.append(self.x)
                self.turn_points_y.append(self.y)

                # Determine direction of turn
                prob_left_right = self.turn_bias  # probability of left or right is 1/2 or bias %
                left_or_right = random.random()  # random number to compare with probability of going left or right

                # Angle at which larva will turn wrt the direction it's already facing
                reference_angle = np.radians(get_truncated_normal(mean=66.501, std_dev=36.874))  # angle in radians
                if left_or_right < prob_left_right:  # if random number is <0.5, go left
                    self.angle += reference_angle  # turn left by reference angle
                else:  # if random number is >=0.5, go right
                    self.angle -= reference_angle  # turn right by reference angle

                self.angle = normalize_angle(self.angle)  # Normalize the angle
                self.angles.append(self.angle)  # Add the new angle after turning in radians
                self.num_turns += 1

                # time it takes to pause and then make the turn
                mean = 4.3
                turn_time = np.random.exponential(scale=mean)
                self.turn_times.append(turn_time)
                timestamp += turn_time

                self.plot_timestamps.append(timestamp)  # Add the current timestamp including turn time
                self.timestamps.append(timestamp)
                # Append new position to the lists
                self.plot_x_positions.append(self.x)
                self.plot_y_positions.append(self.y)
                self.x_positions.append(self.x)
                self.y_positions.append(self.y)

                drift_left_or_right = random.random()  # picks a random number for drift direction

                v = get_truncated_normal(mean=v0, std_dev=stdevi)  # speed of larva in px/s
                self.speeds.append(v)

                # Start new drift segment
                drift_start_x = self.x
                drift_start_y = self.y

                self.last_move = "T"  # Update last move to turn

            else:  # will go straight (with slight drift)
                moves.append("D")
                self.num_runs += 1

                drift_angle = drift_rate * self.time_step  # Calculate drift angle based on drift rate and time step

                prob_drift_left_or_right = self.drift_bias  # probability of drifting left or right is the drift bias

                if drift_left_or_right < prob_drift_left_or_right:
                    self.angle += drift_angle
                else:
                    self.angle -= drift_angle

                self.angle = normalize_angle(self.angle)  # Normalize the angle
                self.x += v0 * np.cos(self.angle) * self.time_step
                self.y += v0 * np.sin(self.angle) * self.time_step
                self.plot_x_positions.append(self.x)
                self.plot_y_positions.append(self.y)
                self.plot_angles.append(self.angle)  # Update the angle list with the current angle
                self.plot_timestamps.append(timestamp)  # Update the timestamps list with the current time

                # Accumulate distance during drift
                drift_distance_accumulator += np.sqrt((self.x - drift_start_x) ** 2 +
                                                      (self.y - drift_start_y) ** 2)
                drift_start_x = self.x
                drift_start_y = self.y

                self.last_move = "D"  # Update last move to drift

        # At the end of the simulation, add the last drift distance if applicable
        if drift_distance_accumulator > 0 and self.num_turns > 0:
            self.runL_distances.append(drift_distance_accumulator)

def main():
    N = int(input("Number of turns (N): "))  # number of turns
    T = int(input("Total time for experiment (in seconds): "))  # total time in seconds
    time_step = float(input("Time step (in seconds): "))  # in seconds
    num_larvae = int(input("How many larvae? "))  # number of larvae
    turn_bias = float(input("Left or right handed turns (decimal [0.0, 1.0]): "))  # <0.5=left, >0.5=right, 0.5="normal"
    drift_bias = float(input("Left or right handed drifts (decimal [0.0, 1.0]): "))  # same numerical representation ^

    larvae = []
    for _ in range(num_larvae):
        larva = Larva(N, T, time_step, turn_bias, drift_bias)
        larva.simulate()
        larvae.append(larva)

    colors = plt.get_cmap('tab20', num_larvae)  # Use a colormap to generate distinct colors

    plt.figure(figsize=(10, 6))

    # Prepare CSV file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('../data', exist_ok=True)
    txt_filename = f'../data/larva_data_{timestamp}.txt'

    with open(txt_filename, mode='w', newline='') as txt_file:
        fieldnames = ['Column1', 'set', 'expt', 'track', 'time0', 'reoYN', 'runQ', 'runL', 'runT', 'runX', 'reo#HS',
                      'reoQ1', 'reoQ2', 'reoHS1', 'runQ0', 'runX0', 'runY0', 'runX1', 'runY1']
        writer = csv.DictWriter(txt_file, fieldnames=fieldnames)
        writer.writeheader()

        for i, larva in enumerate(larvae):
            prev_angle = larva.angles[0]
            prev_x = larva.turn_points_x[0]
            prev_y = larva.turn_points_y[0]
            prev_timestamp = larva.timestamps[0]
            for j in range(1, len(larva.turn_points_x)):
                if j >= len(larva.angles) or j >= len(larva.timestamps) or j >= len(larva.speeds) or j - 1 >= len(larva.drift_angles):
                    break
                current_angle = larva.angles[j]
                Q1 = larva.drift_angles[j - 1]
                runQ = normalize_angle(current_angle - prev_angle)
                runL = larva.runL_distances[j - 1] if j - 1 < len(
                    larva.runL_distances) else 0.0  # Use distance between turns as runL
                runT = larva.timestamps[j] - prev_timestamp - (
                    larva.turn_times[j - 1] if j - 1 < len(larva.turn_times) else 0)
                runX0 = prev_x
                runY0 = prev_y
                runX1 = larva.turn_points_x[j]
                runY1 = larva.turn_points_y[j]
                row = {
                    'Column1': '',
                    'set': 1,
                    'expt': 1,
                    'track': i + 1,
                    'time0': larva.timestamps[j - 1],
                    'reoYN': 1,
                    'runQ': runQ,
                    'runL': runL,
                    'runT': runT,
                    'runX': runX1,
                    'reo#HS': np.random.choice([0, 1, 2, 3, 4, 5], p=[0.05, 0.7, 0.1, 0.05, 0.05, 0.05]),
                    'reoQ1': Q1,
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
                prev_x = larva.turn_points_x[j]
                prev_y = larva.turn_points_y[j]
                prev_timestamp = larva.timestamps[j]

            # Plot trajectory
            plt.plot(larva.plot_x_positions, larva.plot_y_positions, label=f'Larva {i + 1}', color=colors(i))
            plt.scatter(larva.turn_points_x, larva.turn_points_y, s=10, color=colors(i))

    plt.title('Larvae Random Walk with Turning Points')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize='small')  # Place legend outside the plot
    plt.grid(True)
    plt.show()

    # Save interactive plot as HTML using mpld3
    os.makedirs('../simulations', exist_ok=True)
    interactive_filename = f'../simulations/larva_path_{timestamp}.html'
    mpld3.save_html(plt.gcf(), interactive_filename)


if __name__ == "__main__":
    main()
