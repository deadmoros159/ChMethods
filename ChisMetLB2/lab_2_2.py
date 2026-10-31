# lab_2_2.py
import math
import matplotlib.pyplot as plt
from numerical_utils import solve_gauss, norm_vector_inf


a = 2.0

def system_f(X):
    x1, x2 = X
    f1 = x1 - math.cos(x2) - 1.0
    f2 = x2 - math.log10(x1 + 1.0) - a
    return [f1, f2]

def phi_system(X):
    """Преобразованная система для метода простой итерации: X = Psi(X)"""
    x1, x2 = X
    nx1 = 1.0 + math.cos(x2)
    nx2 = a + math.log10(x1 + 1.0)
    return [nx1, nx2]

def jacobian(X):
    """Матрица Якоби для метода Ньютона."""
    x1, x2 = X
    ln10 = math.log(10.0)
    j11 = 1.0
    j12 = math.sin(x2)
    j21 = -1.0 / ((x1 + 1.0) * ln10)
    j22 = 1.0
    return [[j11, j12], [j21, j22]]

def simple_iteration_system(X0, eps=1e-6, max_iter=100):
    """Метод простой итерации для систем."""
    X = X0[:]
    history = [X[:]]
    for _ in range(max_iter):
        X_next = phi_system(X)
        history.append(X_next[:])
        if norm_vector_inf([X_next[k] - X[k] for k in range(len(X))]) < eps:
            return X_next, history
        X = X_next
    return X, history

def newton_system(X0, eps=1e-6, max_iter=100):
    """Метод Ньютона для систем нелинейных уравнений."""
    X = X0[:]
    history = [X[:]]
    for i in range(max_iter):
        F = system_f(X)
        if norm_vector_inf(F) < eps:
            return X, history
        J = jacobian(X)
        # Система J * dX = -F
        minus_F = [-f_val for f_val in F]
        dX = solve_gauss(J, minus_F)
        
        X_next = [X[k] + dX[k] for k in range(len(X))]
        history.append(X_next[:])
        
        if norm_vector_inf(dX) < eps:
            return X_next, history
        X = X_next
    return X, history

def main():
    print("=== Лабораторная работа 2.2 (Вариант 5) ===")
    print(f"Система уравнений (при a = {a}):")
    print("  x1 - cos(x2) = 1")
    print(f"  x2 - lg(x1 + 1) = {a}")

    try:
        eps = float(input("Введите точность вычислений (например, 1e-6): "))
    except ValueError:
        eps = 1e-6
        print("Установлено значение по умолчанию: 1e-6")

    # Графическое представление
    x1_vals = [x / 50.0 for x in range(0, 150)] # от 0 до 3
    x2_pi = []
    x2_n = []
    for x1 in x1_vals:
        # Из первого уравнения: cos(x2) = x1 - 1 => x2 = arccos(x1 - 1) если -1 <= x1-1 <= 1 (т.е. 0 <= x1 <= 2)
        val = x1 - 1.0
        if -1.0 <= val <= 1.0:
            x2_pi.append(math.acos(val))
            x2_n.append(a + math.log10(x1 + 1.0))
        else:
            x2_pi.append(float('nan'))
            x2_n.append(float('nan'))

    plt.figure(figsize=(10, 5))
    plt.plot(x1_vals, x2_pi, label="Из 1-го ур.: x2(x1)", color='red')
    plt.plot(x1_vals, x2_n, label="Из 2-го ур.: x2(x1)", color='blue')
    plt.title("Графическое определение начального приближения (положительные корни)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.legend()
    plt.savefig("lab_2_2_graph.png")
    print("График пересечения кривых сохранен в lab_2_2_graph.png.")

    # Начальное приближение по графику в положительной области
    X0 = [1.5, 2.2]

    # Решение методом простой итерации
    try:
        root_si, hist_si = simple_iteration_system(X0, eps)
        print(f"\n[Метод простой итерации]:")
        print(f"  Решение: x1 = {root_si[0]:.8f}, x2 = {root_si[1]:.8f}")
        print(f"  Количество итераций: {len(hist_si) - 1}")
    except Exception as e:
        print(f"\n[Метод простой итерации не сошелся]: {e}")
        hist_si = [X0]

    # Решение методом Ньютона
    root_n, hist_n = newton_system(X0, eps)
    print(f"\n[Метод Ньютона]:")
    print(f"  Решение: x1 = {root_n[0]:.8f}, x2 = {root_n[1]:.8f}")
    print(f"  Количество итераций: {len(hist_n) - 1}")

    # Погрешностей
    true_val = root_n
    errors_n = [max(abs(val[0] - true_val[0]), abs(val[1] - true_val[1])) for val in hist_n]

    plt.figure(figsize=(10, 5))
    plt.semilogy(range(len(errors_n)), errors_n, marker='x', color='purple', label="Метод Ньютона (Система)")
    if 'hist_si' in locals() and len(hist_si) > 1:
        errors_si = [max(abs(val[0] - true_val[0]), abs(val[1] - true_val[1])) for val in hist_si]
        plt.semilogy(range(len(errors_si)), errors_si, marker='o', color='orange', label="Простая итерация (Система)")
    plt.title("Зависимость погрешности от количества итераций (Система)")
    plt.xlabel("Номер итерации")
    plt.ylabel("Погрешность")
    plt.grid(True)
    plt.legend()
    plt.savefig("lab_2_2_error_analysis.png")
    print("График погрешностей систем сохранен в lab_2_2_error_analysis.png.")

if __name__ == "__main__":
    main()