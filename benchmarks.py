import numpy as np

def F1(x):
    return np.sum(x**2)

def F2(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def F3(x):
    return np.sum([np.sum(x[:i+1])**2 for i in range(len(x))])

def F4(x):
    return np.max(np.abs(x))

def F5(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

def F6(x):
    return np.sum((np.floor(x + 0.5))**2)

def F7(x):
    return np.sum(np.arange(1, len(x) + 1) * (x**4)) + np.random.rand()

def F8(x):
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)

def F9(x):
    n = len(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e

def F10(x):
    return np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))) + 1

def F11(x):
    y = 1 + (x - 1) / 4
    return np.sin(np.pi * y[0])**2 + np.sum((y[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * y[:-1] + 1)**2)) + (y[-1] - 1)**2 * (1 + np.sin(2 * np.pi * y[-1])**2)

def u(x, a, k, m):
    if x > a:
        return k * (x - a)**m
    elif x < -a:
        return k * (-x - a)**m
    else:
        return 0

def F12(x):
    n = len(x)
    term1 = np.sin(3 * np.pi * x[0])**2
    term2 = np.sum((x[:-1] - 1)**2 * (1 + np.sin(3 * np.pi * x[1:])**2))
    term3 = (x[-1] - 1)**2 * (1 + np.sin(2 * np.pi * x[-1])**2)
    term4 = np.sum([u(xi, 5, 100, 4) for xi in x])
    return (term1 + term2 + term3) / 10 + term4
