import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the objective function
def F1(x):
    return np.sum(x**2)  # Sum of squares of each element in x

# Function to initialize the population
def initialization(N, Dim, UB, LB):
    if np.isscalar(UB):  # If UB is a scalar
        X = np.random.rand(N, Dim) * (UB - LB) + LB  # Generate N random solutions within the bounds
    else:
        X = np.zeros((N, Dim))  # Initialize the population matrix
        for i in range(Dim):  # For each dimension
            Ub_i = UB[i]  # Upper bound for this dimension
            Lb_i = LB[i]  # Lower bound for this dimension
            X[:, i] = np.random.rand(N) * (Ub_i - Lb_i) + Lb_i  # Generate N random solutions within the bounds for this dimension

    return X

# Main AOA function
def AOA(N, M_Iter, LB, UB, Dim, F_obj):
    print('AOA Working')
    Best_P = np.zeros(Dim)  # Initialize the best solution
    Best_FF = np.inf  # Initialize the best fitness value
    Conv_curve = np.zeros(M_Iter)  # Initialize the convergence curve
    X = initialization(N, Dim, UB, LB)  # Initialize the population
    Xnew = np.copy(X)  # Copy of the population
    Ffun = np.zeros(X.shape[0])  # Initialize the fitness values
    Ffun_new = np.zeros(Xnew.shape[0])  # Initialize the new fitness values
    MOP_Max = 1  # Maximum probability ratio
    MOP_Min = 0.2  # Minimum probability ratio
    C_Iter = 1  # Current iteration
    Alpha = 5  # Alpha parameter
    Mu = 0.499  # Mu parameter
    for i in range(X.shape[0]):  # For each solution in the population
        Ffun[i] = F_obj(X[i, :])  # Calculate the fitness value
        if Ffun[i] < Best_FF:  # If this solution is better than the current best
            Best_FF = Ffun[i]  # Update the best fitness value
            Best_P = np.copy(X[i, :])  # Update the best solution
    while C_Iter < M_Iter + 1:  # Main loop
        MOP = 1 - ((C_Iter) ** (1 / Alpha) / (M_Iter) ** (1 / Alpha))  # Calculate the probability ratio
        MOA = MOP_Min + C_Iter * ((MOP_Max - MOP_Min) / M_Iter)  # Calculate the accelerated function
        for i in range(X.shape[0]):  # For each solution in the population
            for j in range(X.shape[1]):  # For each dimension of the solution
                r1 = np.random.rand()  # Generate a random number
                if np.isscalar(LB):  # If LB is a scalar
                    if r1 < MOA:  # If the random number is less than MOA
                        r2 = np.random.rand()  # Generate another random number
                        if r2 > 0.5:  # If the second random number is greater than 0.5
                            Xnew[i, j] = Best_P[j] / (MOP + np.finfo(float).eps) * ((UB - LB) * Mu + LB)  # Update the solution
                        else:
                            Xnew[i, j] = Best_P[j] * MOP * ((UB - LB) * Mu + LB)  # Update the solution
                    else:
                        r3 = np.random.rand()  # Generate another random number
                        if r3 > 0.5:  # If the third random number is greater than 0.5
                            Xnew[i, j] = Best_P[j] - MOP * ((UB - LB) * Mu + LB)  # Update the solution
                        else:
                            Xnew[i, j] = Best_P[j] + MOP * ((UB - LB) * Mu + LB)  # Update the solution
                if np.isscalar(LB) is False:  # If LB is not a scalar
                    r1 = np.random.rand()  # Generate a random number
                    if r1 < MOA:  # If the random number is less than MOA
                        r2 = np.random.rand()  # Generate another random number
                        if r2 > 0.5:  # If the second random number is greater than 0.5
                            Xnew[i, j] = Best_P[j] / (MOP + np.finfo(float).eps) * ((UB[j] - LB[j]) * Mu + LB[j])  # Update the solution
                        else:
                            Xnew[i, j] = Best_P[j] * MOP * ((UB[j] - LB[j]) * Mu + LB[j])  # Update the solution
                    else:
                        r3 = np.random.rand()  # Generate another random number
                        if r3 > 0.5:  # If the third random number is greater than 0.5
                            Xnew[i, j] = Best_P[j] - MOP * ((UB[j] - LB[j]) * Mu + LB[j])  # Update the solution
                        else:
                            Xnew[i, j] = Best_P[j] + MOP * ((UB[j] - LB[j]) * Mu + LB[j])  # Update the solution

            Flag_UB = Xnew[i, :] > UB  # Check if the solution exceeds the upper bounds
            Flag_LB = Xnew[i, :] < LB  # Check if the solution exceeds the lower bounds
            Xnew[i, :] = (Xnew[i, :] * (~(Flag_UB + Flag_LB))) + UB * Flag_UB + LB * Flag_LB  # Adjust the solution to be within the bounds
            Ffun_new[i] = F_obj(Xnew[i, :])  # Calculate the new fitness value
            if Ffun_new[i] < Ffun[i]:  # If the new solution is better than the current solution
                X[i, :] = np.copy(Xnew[i, :])  # Update the solution
                Ffun[i] = Ffun_new[i]  # Update the fitness value
            if Ffun[i] < Best_FF:  # If this solution is better than the current best
                Best_FF = Ffun[i]  # Update the best fitness value
                Best_P = np.copy(X[i, :])  # Update the best solution
        Conv_curve[C_Iter - 1] = Best_FF  # Update the convergence curve
        if C_Iter % 50 == 0:  # Every 50 iterations
            print(f'At iteration {C_Iter}, the best solution fitness is {Best_FF}')  # Print the best fitness value
        C_Iter += 1  # Increment the current iteration
    return Best_FF, Best_P, Conv_curve  # Return the best fitness value, the best solution, and the convergence curve

Solution_no = 20  # Number of search solutions
LB = -100  # Lower bound
UB = 100  # Upper bound
Dim = 10  # Number of dimensions
M_Iter = 1000  # Maximum number of iterations
Best_FF, Best_P, Conv_curve = AOA(Solution_no, M_Iter, LB, UB, Dim, F1)  # Run the AOA algorithm

print(f'The best-obtained solution by Math Optimizer is: {Best_P}')
print(f'The best optimal value of the objective function found by Math Optimizer is: {Best_FF}')
