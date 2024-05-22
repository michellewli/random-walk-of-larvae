import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import norm

# Input parameters
N = 100
T = 600  # from Ch16, at least 100 turns for ~10 min
time_interval = 1

turning_probability = (N / T)

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
for _ in np.arange(0, int(T), time_interval):
    turn_or_not = np.random.uniform(0.0, 1.0)  # random float to compare with probability of turning

    if turn_or_not < turning_probability:  # will turn either left or right
        prob_left_right = 0.5  # probability of left or right is 1/2 or 5/10
        left_or_right = random.random()  # random number to compare with probability of going left or right
        if left_or_right < prob_left_right:  # if random number is <0.5, go left
            angle += np.pi / 2  # turn left by 90 degrees
        else:  # if random number is >=0.5, go right
            angle -= np.pi / 2  # turn right by 90 degrees
        num_turns += 1

        # Record turn point
        turn_points_x.append(x)
        turn_points_y.append(y)

        # Record movement type
        movement_sequence.append('T')
    else:  # will go straight
        num_runs += 1
        # Record movement type
        movement_sequence.append('S')

    # Update position
    x += np.cos(angle)  # move in x direction
    y += np.sin(angle)  # move in y direction

    # Append new position to the lists
    x_positions.append(x)
    y_positions.append(y)

# Plot trajectory
plt.plot(x_positions, y_positions, label='Trajectory')
plt.scatter(turn_points_x, turn_points_y, color='red', label='Turn Points')  # Plot turn points as red dots
plt.title('Larva Trajectory with Turn Points')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.xlim(-100, 100)  # Set x-axis limits
plt.ylim(-100, 100)  # Set y-axis limits
plt.legend()
plt.grid(True)
plt.show(block=True)

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
    print(f"Runs: {runs}")
    print(f"Expected Runs: {expected_runs}")
    print(f"Z-value: {z}")
    print(f"P-value: {p_value}")

    if p_value < alpha:
        print(f"The sequence is not random (p-value: {p_value} < alpha: {alpha}).")
    else:
        print(f"The sequence is random (p-value: {p_value} >= alpha: {alpha}).")
else:
    print("Not enough data to perform the Runs Test")


