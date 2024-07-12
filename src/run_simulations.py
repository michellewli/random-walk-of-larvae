import copy_sim

def run_simulation(N, T, time_step, num_larvae, turn_bias, drift_bias):
    copy_sim.main(N, T, time_step, num_larvae, turn_bias, drift_bias)

def main():
    N = 100
    T = 600
    time_step = 1
    num_larvae = 1000
    turn_bias = 0.5

    drift_bias_values = [round(i * 0.05, 2) for i in range(21)]  # Generate list of float values
    for i, drift_bias in enumerate(drift_bias_values):
        print(f"Running simulation {i + 1}/21 with drift_bias: {drift_bias}")
        run_simulation(N, T, time_step, num_larvae, turn_bias, drift_bias)
        print(f"Finished simulation {i + 1}/21 with drift_bias: {drift_bias}")

if __name__ == "__main__":
    main()
