import numpy as np
from scipy.optimize import fsolve, root

# 1. Проверка уравнения для Лабы 2.1
eq = lambda x: np.cos(x) + 0.25 * x - 0.5
root_21 = fsolve(eq, 1.5)
print("NumPy/SciPy корень (Лаба 2.1):", root_21[0])

# 2. Проверка системы для Лабы 2.2
def system(vars):
    x1, x2 = vars
    f1 = x1 - np.cos(x2) - 1.0
    f2 = x2 - np.log10(x1 + 1.0) - 2.0
    return [f1, f2]

root_22 = root(system, [1.5, 2.2])
print("NumPy/SciPy корни (Лаба 2.2):", root_22.x)