import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import norm

# Input parameters
N = 100  # number of turns
T = 600  # total time in seconds
time_step = 1  # in seconds
total_time = 0

turning_probability = (N / T) / time_step

# Initialize position and direction
x, y = 0, 0
angle = 0  # initial angle, 0 means facing right

# Lists to store the trajectory
x_positions = [x]  # starting x position
y_positions = [y]  # starting y position

turn_points_x = []  # x coordinates of turn points
turn_points_y = []  # y coordinates of turn points

num_turns = 0
num_runs = 0

# Sequence to store the type of each movement (turn: 'T', straight: 'S')
movement_sequence = []

# Simulation loop
for _ in np.arange(0, int(T), time_step):
    turn_or_not = np.random.uniform(0.0, 1.0)  # random float to compare with probability of turning

    v0 = np.random.normal(loc=2.9095, scale=0.7094, size=1)[0]  # speed of larva in px/s
    deltat = np.random.normal(loc=18.704, scale=23.316, size=1)[0]  # change in time between turns
    total_time += deltat

    if turn_or_not < turning_probability:  # will turn either left or right
        # First, move to the current position based on the previous angle
        x += v0 * np.cos(np.radians(angle)) * time_step
        y += v0 * np.sin(np.radians(angle)) * time_step
        
        # Record turn point
        turn_points_x.append(x)
        turn_points_y.append(y)

        # Determine direction of turn
        prob_left_right = 0.5  # probability of left or right is 1/2 or 5/10
        left_or_right = random.random()  # random number to compare with probability of going left or right

        # Angle at which larva will turn wrt the direction it's already facing
        reference_angle = np.random.normal(loc=66.501, scale=36.874, size=1)[0]  # angle in degrees

        if left_or_right < prob_left_right:  # if random number is <0.5, go left
            angle += reference_angle  # turn left by reference angle
        else:  # if random number is >=0.5, go right
            angle -= reference_angle  # turn right by reference angle
        
        num_turns += 1

        # Record movement type
        movement_sequence.append('T')
    else:  # will go straight
        num_runs += 1
        # Record movement type
        movement_sequence.append('S')
        
        # Update position without turning
        x += v0 * np.cos(np.radians(angle)) * time_step
        y += v0 * np.sin(np.radians(angle)) * time_step

    # Append new position to the lists
    x_positions.append(x)
    y_positions.append(y)

print("Number of turns:", num_turns)
print("Number of straight runs:", num_runs)

# Runs Test
def runs_test(sequence):
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

runs, expected_runs, z, p_value = runs_test(movement_sequence)
alpha = 0.05  # Significance level

if runs is not None:
    print(f"Probability Runs: {runs}")
    print(f"Expected Runs: {expected_runs}")
    print(f"Z-value: {z}")
    print(f"P-value: {p_value}")

    if p_value < alpha:
        print(f"The sequence is not random (p-value: {p_value} < alpha: {alpha}).")
    else:
        print(f"The sequence is random (p-value: {p_value} >= alpha: {alpha}).")
else:
    print("Not enough data to perform the Runs Test")

# Print the amount of time the runs took
print(f"Total time taken: {total_time} seconds")


# Plot trajectory
plt.scatter(0, 0, color='green', label='Start Position')  # Plot starting position
plt.plot(x_positions, y_positions, label='Trajectory')
plt.scatter(turn_points_x, turn_points_y, color='red', label='Turning Points')  # Plot turn points as red dots
plt.scatter(x_positions[-1], y_positions[-1], color='blue', label='End Position')  # Plot last point in trajectory
plt.title('Larva Random Walk with Turning Points')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.legend()
plt.grid(True)
plt.show()

