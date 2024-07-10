import copy_sim
import os

def run_simulation(N, T, time_step, num_larvae, turn_bias, drift_bias):
    copy_sim.main(N, T, time_step, num_larvae, turn_bias, drift_bias)

def main():
    N = 100
    T = 600
    time_step = 1
    num_larvae = 1000
    drift_bias = 0.5

    # Iterate over turn_bias values from 0.0 to 1.0 with 0.05 increments, produces 21 simulations
    turn_bias_values = [round(i * 0.05, 2) for i in range(21)]  # Generate list of float values
    for turn_bias in turn_bias_values:
        print(f"num: {turn_bias}")
        run_simulation(N, T, time_step, num_larvae, turn_bias, drift_bias)

if __name__ == "__main__":
    main()
