import matplotlib.pyplot as plt
import numpy as np
import random
import csv
from scipy.stats import truncnorm, norm
from datetime import datetime

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
        self.angle = random.uniform(0, 180)  # initial angle, 0 means facing right
        self.x_positions = [self.x]  # starting x position
        self.y_positions = [self.y]  # starting y position
        self.turn_points_x = []  # x coordinates of turn points
        self.turn_points_y = []  # y coordinates of turn points
        self.num_turns = 0
        self.num_runs = 0
        self.movement_sequence = []
        self.speeds = []
        self.angles = []

    def simulate(self):
        for _ in np.arange(0, int(self.T), self.time_step):
            turn_or_not = np.random.uniform(0.0, 1.0)  # random float to compare with probability of turning

            v0 = get_truncated_normal(mean=2.9095, std_dev=0.7094)  # speed of larva in px/s
            self.speeds.append(v0)

            if turn_or_not < self.turning_probability:  # will turn either left or right
                # First, move to the current position based on the previous angle
                self.x += v0 * np.cos(np.radians(self.angle)) * self.time_step
                self.y += v0 * np.sin(np.radians(self.angle)) * self.time_step

                # Record turn point
                self.turn_points_x.append(self.x)
                self.turn_points_y.append(self.y)

                # Determine direction of turn
                prob_left_right = 0.5  # probability of left or right is 1/2 or 5/10
                left_or_right = random.random()  # random number to compare with probability of going left or right

                # Angle at which larva will turn wrt the direction it's already facing
                reference_angle = get_truncated_normal(mean=66.501, std_dev=36.874)  # angle in degrees
                self.angles.append(reference_angle)

                if left_or_right < prob_left_right:  # if random number is <0.5, go left
                    self.angle += reference_angle  # turn left by reference angle
                else:  # if random number is >=0.5, go right
                    self.angle -= reference_angle  # turn right by reference angle

                self.num_turns += 1

                # Record movement type
                self.movement_sequence.append('T')
            else:  # will go straight
                self.num_runs += 1
                # Record movement type
                self.movement_sequence.append('S')

                # Update position without turning
                self.x += v0 * np.cos(np.radians(self.angle)) * self.time_step
                self.y += v0 * np.sin(np.radians(self.angle)) * self.time_step

            # Append new position to the lists
            self.x_positions.append(self.x)
            self.y_positions.append(self.y)

        return self.x_positions, self.y_positions, self.turn_points_x, self.turn_points_y

    def runs_test(self):
        sequence = self.movement_sequence
        n1 = sequence.count('T')
        n2 = sequence.count('S')
        n = n1 + n2

        if n1 == 0 or n2 == 0:
            return None, None, None, None  # Not enough data to perform the test

        # Calculate number of runs
        runs = 1
        for i in range(1, len(sequence)):
            if sequence[i] != sequence[i - 1]:
                runs += 1

        # Expected number of runs
        expected_runs = (2 * n1 * n2) / n + 1
        # Variance of the number of runs
        variance_runs = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n ** 2 * (n - 1))

        if variance_runs == 0:
            return runs, expected_runs, None, None

        # Z-value for the runs test
        z = (runs - expected_runs) / np.sqrt(variance_runs)
        # Two-tailed p-value
        p_value = 2 * (1 - norm.cdf(abs(z)))

        return runs, expected_runs, z, p_value

def main():
    N = 100  # number of turns
    T = 600  # total time in seconds
    time_step = 1  # in seconds
    num_walkers = int(input("How many larvae? "))  # number of larvae

    walkers = []
    for _ in range(num_walkers):
        walker = LarvaWalker(N, T, time_step)
        walker.simulate()
        walkers.append(walker)

    colors = plt.get_cmap('tab20', num_walkers)  # Use a colormap to generate distinct colors

    plt.figure(figsize=(10, 6))

    # Prepare CSV file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f'data/larva_data_{timestamp}.csv'

    with open(csv_filename, mode='w', newline='') as csv_file:
        fieldnames = ['Larva', 'Number of Turns', 'Number of Straight Runs', 'Speed', 'Angle', 'Runs', 'Expected Runs', 'Z-value', 'P-value']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        max_length = max(len(walker.speeds) for walker in walkers)

        for i, walker in enumerate(walkers):
            runs, expected_runs, z, p_value = walker.runs_test()
            for j in range(max_length):
                row = {
                    'Larva': f'Larva {i+1}' if j == 0 else '',
                    'Number of Turns': walker.num_turns if j == 0 else '',
                    'Number of Straight Runs': walker.num_runs if j == 0 else '',
                    'Speed': walker.speeds[j] if j < len(walker.speeds) else '',
                    'Angle': walker.angles[j] if j < len(walker.angles) else '',
                    'Runs': runs if j == 0 else '',
                    'Expected Runs': expected_runs if j == 0 else '',
                    'Z-value': z if j == 0 else '',
                    'P-value': p_value if j == 0 else ''
                }
                writer.writerow(row)

            # Plot trajectory
            plt.plot(walker.x_positions, walker.y_positions, label=f'Larva {i+1}', color=colors(i))
            plt.scatter(walker.turn_points_x, walker.turn_points_y, s=10, color=colors(i))

    plt.title('Larvae Random Walk with Turning Points')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1), ncol=1, fontsize='small')  # Place legend outside the plot
    plt.grid(True)

    # Save plot as an image with a timestamp
    filename = f'images/larva_path_{timestamp}.png'
    plt.savefig(filename, bbox_inches='tight')  # Save figure with tight bounding box

    plt.show()

if __name__ == "__main__":
    main()
