import numpy as np
from AOA import AOA
from AOA_QL import AOA_QL
import benchmarks
import matplotlib.pyplot as plt

# Define the benchmark functions and their properties
BENCHMARKS = {
    "F1": {"func": benchmarks.F1, "LB": -100, "UB": 100, "Dim": 30},
    "F2": {"func": benchmarks.F2, "LB": -10, "UB": 10, "Dim": 30},
    "F3": {"func": benchmarks.F3, "LB": -100, "UB": 100, "Dim": 30},
    "F4": {"func": benchmarks.F4, "LB": -100, "UB": 100, "Dim": 30},
    "F5": {"func": benchmarks.F5, "LB": -30, "UB": 30, "Dim": 30},
    "F6": {"func": benchmarks.F6, "LB": -100, "UB": 100, "Dim": 30},
    "F7": {"func": benchmarks.F7, "LB": -1.28, "UB": 1.28, "Dim": 30},
    "F8": {"func": benchmarks.F8, "LB": -5.12, "UB": 5.12, "Dim": 30},
    "F9": {"func": benchmarks.F9, "LB": -32, "UB": 32, "Dim": 30},
    "F10": {"func": benchmarks.F10, "LB": -600, "UB": 600, "Dim": 30},
    "F11": {"func": benchmarks.F11, "LB": -10, "UB": 10, "Dim": 30},
    "F12": {"func": benchmarks.F12, "LB": -50, "UB": 50, "Dim": 30},
}

Solution_no = 20
M_Iter = 1000

for name, props in BENCHMARKS.items():
    print(f"--- Running Benchmark {name} ---")

    # Run the original AOA algorithm
    Best_FF_AOA, Best_P_AOA, Conv_curve_AOA = AOA(Solution_no, M_Iter, props["LB"], props["UB"], props["Dim"], props["func"])

    # Run the AOA-QL algorithm
    Best_FF_AOA_QL, Best_P_AOA_QL, Conv_curve_AOA_QL = AOA_QL(Solution_no, M_Iter, props["LB"], props["UB"], props["Dim"], props["func"])

    # Print the results
    print(f"AOA Best Fitness for {name}: {Best_FF_AOA}")
    print(f"AOA_QL Best Fitness for {name}: {Best_FF_AOA_QL}")

    # Plot the convergence curves
    plt.figure(figsize=(10, 6))
    plt.plot(Conv_curve_AOA, label='AOA')
    plt.plot(Conv_curve_AOA_QL, label='AOA-QL')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.title(f'Convergence Curve for {name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'images/convergence_curve_{name}.png')
    plt.close()

print("All benchmarks completed.")
