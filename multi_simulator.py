import matplotlib.pyplot as plt
import numpy as np
import random
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
        self.x, self.y = 0, 0
        self.angle = 0  # initial angle, 0 means facing right
        self.x_positions = [self.x]  # starting x position
        self.y_positions = [self.y]  # starting y position
        self.turn_points_x = []  # x coordinates of turn points
        self.turn_points_y = []  # y coordinates of turn points
        self.num_turns = 0
        self.num_runs = 0
        self.movement_sequence = []
        self.runtime = []
        self.total_time = 0

    def simulate(self):
        for _ in np.arange(0, int(self.T), self.time_step):
            turn_or_not = np.random.uniform(0.0, 1.0)  # random float to compare with probability of turning

            v0 = get_truncated_normal(mean=2.9095, std_dev=0.7094)  # speed of larva in px/s
            deltat = get_truncated_normal(mean=18.704, std_dev=23.316)  # change in time between turns
            self.runtime.append(deltat)
            self.total_time += deltat

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

    for i, walker in enumerate(walkers):
        runs, expected_runs, z, p_value = walker.runs_test()
        print(f"Walker {i+1}:")
        print(f"  Number of turns: {walker.num_turns}")
        print(f"  Number of straight runs: {walker.num_runs}")
        if runs is not None:
            print(f"  Probability Runs: {runs}")
            print(f"  Expected Runs: {expected_runs}")
            print(f"  Z-value: {z}")
            print(f"  P-value: {p_value}")
            if p_value < 0.05:
                print(f"  The sequence is not random (p-value: {p_value} < 0.05).")
            else:
                print(f"  The sequence is random (p-value: {p_value} >= 0.05).")
        else:
            print("  Not enough data to perform the Runs Test")
        print(f"  Total time taken: {walker.total_time} seconds")

        # Plot trajectory
        plt.plot(walker.x_positions, walker.y_positions, label=f'Larva {i+1}', color=colors(i))
        plt.scatter(walker.turn_points_x, walker.turn_points_y, s=10, color=colors(i))

    plt.scatter(0, 0, color='green', label='Start Position')  # Plot starting position
    plt.scatter(walkers[-1].x_positions[-1], walkers[-1].y_positions[-1], color='blue', label='End Position')  # Last point

    plt.title('Larvae Random Walk with Turning Points')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1), ncol=1, fontsize='small')  # Place legend outside the plot
    plt.grid(True)

    # Save plot as an image with a timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'larva_path_{timestamp}.png'
    plt.savefig(filename, bbox_inches='tight')  # Save figure with tight bounding box

    plt.show()

if __name__ == "__main__":
    main()
