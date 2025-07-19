import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Q-learning parameters
Q_TABLE_SIZE = (10, 4)  # (states, actions)
ALPHA_QL = 0.1  # Learning rate
GAMMA_QL = 0.9  # Discount factor
EPSILON_QL = 0.1  # Epsilon for epsilon-greedy policy

# Initialize Q-table
Q_table = np.zeros(Q_TABLE_SIZE)

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

# Function to get the state of an agent
def get_state(agent_fitness, best_fitness, diversity, iteration, max_iterations):
    # Fitness level
    fitness_ratio = agent_fitness / best_fitness if best_fitness != 0 else 1
    if fitness_ratio < 1.1:
        fitness_level = 0  # Elite
    elif fitness_ratio < 1.5:
        fitness_level = 1  # Good
    elif fitness_ratio < 2.0:
        fitness_level = 2  # Average
    else:
        fitness_level = 3  # Poor

    # Diversity level
    if diversity < 0.1:
        diversity_level = 0  # Low
    elif diversity < 0.5:
        diversity_level = 1  # Medium
    else:
        diversity_level = 2  # High

    # Search phase
    if iteration < max_iterations / 3:
        phase = 0  # Early
    elif iteration < 2 * max_iterations / 3:
        phase = 1  # Middle
    else:
        phase = 2  # Late

    # Combine to a single state index
    state = fitness_level + diversity_level * 4 + phase * 8
    return min(state, Q_TABLE_SIZE[0] - 1)

# Function to select an action
def choose_action(state):
    if np.random.rand() < EPSILON_QL:
        return np.random.randint(Q_TABLE_SIZE[1])  # Exploration
    else:
        return np.argmax(Q_table[state, :])  # Exploitation

# Function to get the reward
def get_reward(old_fitness, new_fitness, is_best):
    if new_fitness < old_fitness:
        if is_best:
            return 10  # Bonus for new global best
        return 1  # Improvement
    return -0.1  # Stagnation

# Main AOA_QL function
def AOA_QL(N, M_Iter, LB, UB, Dim, F_obj):
    print('AOA-QL Working')
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
        diversity = np.mean(np.abs(X - Best_P))
        for i in range(X.shape[0]):  # For each solution in the population
            state = get_state(Ffun[i], Best_FF, diversity, C_Iter, M_Iter)
            action = choose_action(state)

            # Apply action
            if action == 0:  # Aggressive Exploration
                MOP = 0.9
                MOA = 0.9
            elif action == 1:  # Balanced Exploration
                MOP = 0.5
                MOA = 0.7
            elif action == 2:  # Aggressive Exploitation
                MOP = 0.1
                MOA = 0.3
            else:  # Fine-Tuning Exploitation
                MOP = 0.01
                MOA = 0.1

            for j in range(X.shape[1]):  # For each dimension of the solution
                r1 = np.random.rand()
                if np.isscalar(LB):
                    if r1 < MOA:
                        r2 = np.random.rand()
                        if r2 > 0.5:
                            Xnew[i, j] = Best_P[j] / (MOP + np.finfo(float).eps) * ((UB - LB) * Mu + LB)
                        else:
                            Xnew[i, j] = Best_P[j] * MOP * ((UB - LB) * Mu + LB)
                    else:
                        r3 = np.random.rand()
                        if r3 > 0.5:
                            Xnew[i, j] = Best_P[j] - MOP * ((UB - LB) * Mu + LB)
                        else:
                            Xnew[i, j] = Best_P[j] + MOP * ((UB - LB) * Mu + LB)
                else:
                    if r1 < MOA:
                        r2 = np.random.rand()
                        if r2 > 0.5:
                            Xnew[i, j] = Best_P[j] / (MOP + np.finfo(float).eps) * ((UB[j] - LB[j]) * Mu + LB[j])
                        else:
                            Xnew[i, j] = Best_P[j] * MOP * ((UB[j] - LB[j]) * Mu + LB[j])
                    else:
                        r3 = np.random.rand()
                        if r3 > 0.5:
                            Xnew[i, j] = Best_P[j] - MOP * ((UB[j] - LB[j]) * Mu + LB[j])
                        else:
                            Xnew[i, j] = Best_P[j] + MOP * ((UB[j] - LB[j]) * Mu + LB[j])

            Flag_UB = Xnew[i, :] > UB
            Flag_LB = Xnew[i, :] < LB
            Xnew[i, :] = (Xnew[i, :] * (~(Flag_UB + Flag_LB))) + UB * Flag_UB + LB * Flag_LB
            Ffun_new[i] = F_obj(Xnew[i, :])

            is_best = Ffun_new[i] < Best_FF
            reward = get_reward(Ffun[i], Ffun_new[i], is_best)
            new_state = get_state(Ffun_new[i], Best_FF, diversity, C_Iter, M_Iter)

            # Q-table update
            old_value = Q_table[state, action]
            next_max = np.max(Q_table[new_state, :])
            new_value = (1 - ALPHA_QL) * old_value + ALPHA_QL * (reward + GAMMA_QL * next_max)
            Q_table[state, action] = new_value

            if Ffun_new[i] < Ffun[i]:
                X[i, :] = np.copy(Xnew[i, :])
                Ffun[i] = Ffun_new[i]
            if Ffun[i] < Best_FF:
                Best_FF = Ffun[i]
                Best_P = np.copy(X[i, :])

        Conv_curve[C_Iter - 1] = Best_FF
        if C_Iter % 50 == 0:
            print(f'At iteration {C_Iter}, the best solution fitness is {Best_FF}')
        C_Iter += 1
    return Best_FF, Best_P, Conv_curve

# Solution_no = 20  # Number of search solutions
# LB = -100  # Lower bound
# UB = 100  # Upper bound
# Dim = 10  # Number of dimensions
# M_Iter = 1000  # Maximum number of iterations
# Best_FF, Best_P, Conv_curve = AOA_QL(Solution_no, M_Iter, LB, UB, Dim, F1)  # Run the AOA algorithm

# print(f'The best-obtained solution by Math Optimizer is: {Best_P}')
# print(f'The best optimal value of the objective function found by Math Optimizer is: {Best_FF}')
