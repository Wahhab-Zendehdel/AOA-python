import numpy as np
from AOA import AOA
from AOA_QL import AOA_QL
import matplotlib.pyplot as plt

# Define the objective function
def F1(x):
    return np.sum(x**2)

Solution_no = 20
LB = -100
UB = 100
Dim = 10
M_Iter = 1000

# Run the original AOA algorithm
Best_FF_AOA, Best_P_AOA, Conv_curve_AOA = AOA(Solution_no, M_Iter, LB, UB, Dim, F1)

# Run the AOA-QL algorithm
Best_FF_AOA_QL, Best_P_AOA_QL, Conv_curve_AOA_QL = AOA_QL(Solution_no, M_Iter, LB, UB, Dim, F1)

# Print the results
print(f"AOA Best Fitness: {Best_FF_AOA}")
print(f"AOA_QL Best Fitness: {Best_FF_AOA_QL}")

# Plot the convergence curves
plt.figure(figsize=(10, 6))
plt.plot(Conv_curve_AOA, label='AOA')
plt.plot(Conv_curve_AOA_QL, label='AOA-QL')
plt.xlabel('Iteration')
plt.ylabel('Best Fitness')
plt.title('Convergence Curve')
plt.legend()
plt.grid(True)
plt.savefig('images/convergence_curve.png')
plt.show()
