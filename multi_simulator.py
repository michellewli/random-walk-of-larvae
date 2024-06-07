import matplotlib.pyplot as plt
import numpy as np
import random
import csv
from scipy.stats import truncnorm
from datetime import datetime
import mpld3

# Helper function to generate truncated normal values
def get_truncated_normal(mean, std_dev, lower_bound=0):
    a = (lower_bound - mean) / std_dev  # Lower bound in standard normal terms
    return truncnorm(a, float('inf'), loc=mean, scale=std_dev).rvs()

class LarvaWalker:
    def __init__(self, N, T, time_step=1):
        self.N = N
        self.T = T
        self.time_step = time_step
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

    def simulate(self):
        elapsed_time = 0
        for _ in np.arange(0, int(self.T), self.time_step):
            elapsed_time += self.time_step
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
                prob_left_right = 0.5  # probability of left or right is 1/2 or 5/10
                left_or_right = random.random()  # random number to compare with probability of going left or right

                # Angle at which larva will turn wrt the direction it's already facing
                reference_angle = np.radians(get_truncated_normal(mean=66.501, std_dev=36.874))  # angle in radians
                if left_or_right < prob_left_right:  # if random number is <0.5, go left
                    self.angle += reference_angle  # turn left by reference angle
                else:  # if random number is >=0.5, go right
                    self.angle -= reference_angle  # turn right by reference angle

                self.angles.append(self.angle % (2 * np.pi))  # Add the new angle after turning in radians
                self.num_turns += 1

                self.times.append(elapsed_time)  # Add the current elapsed time
                # Append new position to the lists
                self.x_positions.append(self.x)
                self.y_positions.append(self.y)

            else:  # will go straight
                self.num_runs += 1

                # Update position without turning
                self.x += v0 * np.cos(self.angle) * self.time_step
                self.y += v0 * np.sin(self.angle) * self.time_step

        return self.x_positions, self.y_positions, self.turn_points_x, self.turn_points_y

def main():
    N = 100  # number of turns
    T = 600  # total time in seconds
    time_step = 1  # in seconds
    num_walkers = int(input("How many larvae? "))  # number of larvae

    walkers = []
    all_speeds = []
    all_angles = []
    for _ in range(num_walkers):
        walker = LarvaWalker(N, T, time_step)
        walker.simulate()
        walkers.append(walker)
        all_speeds.extend(walker.speeds)
        all_angles.extend(walker.angles[1:])  # Exclude the initial angle from the list of angles for histogram

    colors = plt.get_cmap('tab20', num_walkers)  # Use a colormap to generate distinct colors

    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare CSV file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f'data/larva_data_{timestamp}.csv'

    with open(csv_filename, mode='w', newline='') as csv_file:
        fieldnames = ['Column1', 'set', 'expt', 'track', 'time0', 'reoYN', 'runQ', 'runL', 'runT', 'runX', 'reo#HS', 'reoQ1', 'reoQ2', 'reoHS1', 'runQ0', 'runX0', 'runY0', 'runX1', 'runY1']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for i, walker in enumerate(walkers):
            prev_angle = walker.angles[0]
            prev_x = walker.x_positions[0]
            prev_y = walker.y_positions[0]
            for j in range(1, len(walker.x_positions)):
                current_angle = walker.angles[j]
                runQ = current_angle - prev_angle
                runL = walker.speeds[j-1] * walker.time_step
                runT = walker.times[j] - walker.times[j - 1]
                runX0 = prev_x
                runY0 = prev_y
                runX1 = walker.x_positions[j]
                runY1 = walker.y_positions[j]
                row = {
                    'Column1': '',
                    'set': 1,
                    'expt': 1,
                    'track': i + 1,
                    'time0': walker.times[j],  # Use the elapsed time for time0
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

            # Plot trajectory
            ax.plot(walker.x_positions, walker.y_positions, label=f'Larva {i+1}', color=colors(i))
            ax.scatter(walker.turn_points_x, walker.turn_points_y, s=10, color=colors(i))

    ax.set_title('Larvae Random Walk with Turning Points')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.legend(loc='best', bbox_to_anchor=(1.05, 1), ncol=1, fontsize='small')  # Place legend outside the plot
    ax.grid(True)

    # Save interactive plot as an HTML file
    interactive_filename = f'images/larva_path_{timestamp}.html'
    mpld3.save_html(fig, interactive_filename)

    plt.show()

    # Plot histograms of all speeds and angles
    fig_hist, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    ax1.hist(all_speeds, bins=30, color='blue', edgecolor='black', alpha=0.7)
    ax1.set_title('Histogram of Speeds')
    ax1.set_xlabel('Speed (px/s)')
    ax1.set_ylabel('Frequency')

    ax2.hist(all_angles, bins=30, color='green', edgecolor='black', alpha=0.7)
    ax2.set_title('Histogram of Angles')
    ax2.set_xlabel('Angle (radians)')
    ax2.set_ylabel('Frequency')

    # Save interactive histograms as an HTML file
    hist_interactive_filename = f'histograms/larva_histograms_{timestamp}.html'
    mpld3.save_html(fig_hist, hist_interactive_filename)

    plt.show()

if __name__ == "__main__":
    main()
