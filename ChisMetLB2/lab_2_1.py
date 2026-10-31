# lab_2_1.py
import math
import matplotlib.pyplot as plt

def f(x):
    """Исходная функция: cos(x) + 0.25*x - 0.5 = 0"""
    return math.cos(x) + 0.25 * x - 0.5

def df(x):
    """Производная функции для метода Ньютона: -sin(x) + 0.25"""
    return -math.sin(x) + 0.25

tau = 0.5

def phi(x):
    """
    Преобразованная функция для метода простой итерации: x_{k+1} = phi(x_k)
    Используем формулу с шаговым множителем (релаксацией) со знаком '+' 
    для попадания в область сходимости (|1 + tau * f'(x)| < 1).
    """
    return x + tau * f(x)

def dphi(x):
    """Производная вспомогательной функции phi(x)"""
    return 1.0 + tau * df(x)

def simple_iteration_method(x0, eps=1e-6, max_iter=100):
    """Метод простой итерации для нелинейного уравнения."""
    x = x0
    history = [x]
    for i in range(max_iter):
        x_next = phi(x)
        history.append(x_next)
        if abs(x_next - x) < eps:
            return x_next, history
        x = x_next
    return x, history

def newton_method(x0, eps=1e-6, max_iter=100):
    """Метод Ньютона для нелинейного уравнения."""
    x = x0
    history = [x]
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-12:
            raise ValueError("Производная близка к нулю.")
        x_next = x - fx / dfx
        history.append(x_next)
        if abs(x_next - x) < eps:
            return x_next, history
        x = x_next
    return x, history

def main():
    print("=== Лабораторная работа 2.1 (Вариант 5) ===")
    print("Уравнение: cos(x) + 0.25*x - 0.5 = 0")

    # Ввод точности с клавиатуры
    try:
        eps = float(input("Введите точность вычислений (например, 1e-6): "))
    except ValueError:
        eps = 1e-6
        print("Установлено значение по умолчанию: 1e-6")

    # 1. Графическое определение корня
    xs = [x / 100.0 for x in range(0, 300)]
    ys = [f(x) for x in xs]

    plt.figure(figsize=(10, 5))
    plt.plot(xs, ys, label="f(x) = cos(x) + 0.25x - 0.5", color='blue')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title("График функции для определения начального приближения")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.savefig("lab_2_1_graph.png")
    print("График функции сохранен в файл lab_2_1_graph.png.")
    
    # Начальное приближение, определенное по графику
    x0 = 1.5

    q = abs(dphi(x0))
    print(f"Модуль производной phi'(x0) = {q:.4f} " + ("< 1 (условие сходимости выполнено)" if q < 1 else ""))

    # 2. Решение методом простой итерации
    root_si, hist_si = simple_iteration_method(x0, eps)
    print(f"\n[Метод простой итерации]:")
    print(f"  Корень: {root_si:.8f}")
    print(f"  Количество итераций: {len(hist_si) - 1}")

    # 3. Решение методом Ньютона
    root_n, hist_n = newton_method(x0, eps)
    print(f"\n[Метод Ньютона]:")
    print(f"  Корень: {root_n:.8f}")
    print(f"  Количество итераций: {len(hist_n) - 1}")

    # 4. Анализ зависимости погрешности от количества итераций
    true_val = root_n
    errors_si = [abs(val - true_val) for val in hist_si]
    errors_n = [abs(val - true_val) for val in hist_n]

    plt.figure(figsize=(10, 5))
    plt.semilogy(range(len(errors_si)), errors_si, marker='o', label="Простая итерация")
    plt.semilogy(range(len(errors_n)), errors_n, marker='x', label="Метод Ньютона")
    plt.title("Зависимость погрешности от количества итераций")
    plt.xlabel("Номер итерации")
    plt.ylabel("Погрешность (логарифмическая шкала)")
    plt.grid(True)
    plt.legend()
    plt.savefig("lab_2_1_error_analysis.png")
    print("График погрешностей сохранен в файл lab_2_1_error_analysis.png.")

if __name__ == "__main__":
    main()