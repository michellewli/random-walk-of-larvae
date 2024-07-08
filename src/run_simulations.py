import copy_sim
import os


def run_simulation(N, T, time_step, num_larvae, turn_bias, drift_bias):
    # Modify the main function in multi_simulator to accept parameters
    copy_sim.main(N, T, time_step, num_larvae, turn_bias, drift_bias)


def main():
    N = 100
    T = 600
    time_step = 1
    num_larvae = 1000
    drift_bias = 0.5

    # Iterate over turn_bias values from 0 to 1 with 0.05 increments, produces 21 simulations
    for i in range(21):
        turn_bias = i * 0.05
        run_simulation(N, T, time_step, num_larvae, turn_bias, drift_bias)


if __name__ == "__main__":
    main()
