import matplotlib.pyplot as plt
import numpy as np
import random
import csv
from scipy.stats import truncnorm
from scipy.interpolate import interp1d
from datetime import datetime
import os
import mpld3

# Helper function to generate truncated normal values
def get_truncated_normal(mean, std_dev, lower_bound=0):
    a = (lower_bound - mean) / std_dev  # Lower bound in standard normal terms
    return truncnorm(a, float('inf'), loc=mean, scale=std_dev).rvs()

class LarvaWalker:
    def __init__(self, N, T, time_step, handedness=0):
        self.N = N
        self.T = T
        self.time_step = time_step
        self.handedness = handedness
        self.turning_probability = ((N / T) / time_step) + self.handedness
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
    N = int(input("Number of turns (N): "))  # number of turns
    T = int(input("Total time for experiment: "))  # total time in seconds
    time_step = float(input("Time step: "))  # in seconds
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

    plt.figure(figsize=(10, 6))

    # Prepare CSV file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('data', exist_ok=True)
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

            # Interpolate trajectory for smooth plotting
            x = np.array(walker.x_positions)
            y = np.array(walker.y_positions)
            t = np.linspace(0, 1, len(x))
            t_new = np.linspace(0, 1, 300)  # Increase the number of points for smoothness

            x_interpolator = interp1d(t, x, kind='cubic')
            y_interpolator = interp1d(t, y, kind='cubic')

            x_smooth = x_interpolator(t_new)
            y_smooth = y_interpolator(t_new)

            plt.plot(x_smooth, y_smooth, label=f'Larva {i+1}', color=colors(i))
            plt.scatter(walker.turn_points_x, walker.turn_points_y, s=10, color=colors(i))

    plt.title('Larvae Random Walk with Turning Points')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=1, fontsize='small')  # Place legend inside the plot
    plt.grid(True)

    # Save interactive plot as HTML using mpld3
    os.makedirs('histograms', exist_ok=True)
    html_filename = f'histograms/larva_walk_{timestamp}.html'
    mpld3.save_html(plt.gcf(), html_filename)
    print(f'Interactive plot saved as {html_filename}')

    plt.show()

if __name__ == '__main__':
    main()
