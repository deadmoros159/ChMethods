"""Лабораторная 3: метод вращений Якоби для собственных значений."""
from numerical_methods import (
    NumericalMethodError, format_matrix, format_vector, identity, jacobi_rotations,
    matmul, max_difference, to_matrix,
)


def run(config: dict) -> str:
    cfg = config.get("lab3", config)
    if not cfg:
        raise NumericalMethodError("В input JSON отсутствует раздел lab3.")
    a = to_matrix(cfg.get("A"), "lab3.A")
    if "epsilon" not in cfg:
        raise NumericalMethodError("Укажите lab3.epsilon во входном JSON.")
    eps = float(cfg["epsilon"])
    max_iter = int(cfg.get("max_iterations", 100000))
    values, vectors, rotations, offdiag = jacobi_rotations(a, eps, max_iter)
    # Проверяем исходную (до вращений) матрицу A: A V = V Λ.
    av = matmul(a, vectors)
    v_lambda = [[vectors[i][j] * values[j] for j in range(len(values))] for i in range(len(values))]
    out = [
        "=" * 78,
        "ЛАБОРАТОРНАЯ РАБОТА № 3. МЕТОД ВРАЩЕНИЙ ЯКОБИ",
        "=" * 78,
        f"Заданная точность ε = {eps:g}",
        f"Число вращений = {rotations}",
        f"Максимальный внедиагональный элемент после завершения = {offdiag:.3e}",
        "Исходная матрица A:\n" + format_matrix(a),
        "Найденные собственные значения: " + format_vector(values),
        "Собственные векторы расположены по столбцам матрицы V:\n" + format_matrix(vectors),
        "Матрица Λ = diag(λ):\n" + format_matrix([[values[i] if i == j else 0.0 for j in range(len(values))] for i in range(len(values))]),
        "Проверка A · V:\n" + format_matrix(av),
        "Проверка V · Λ:\n" + format_matrix(v_lambda),
        f"max|A·V - V·Λ| = {max_difference(av, v_lambda):.3e}",
        f"max|VᵀV-E| = {max_difference(matmul([list(row) for row in zip(*vectors)], vectors), identity(len(values))):.3e} (ортонормальность собственных векторов)",
    ]
    return "\n".join(out)


if __name__ == "__main__":
    from cli_utils import run_cli

    run_cli(run, "Лабораторная работа № 3")
