"""Лабораторная 4: QR-разложение, QR-итерации и проверка комплексной пары."""
from numerical_methods import (
    NumericalMethodError, format_eigenvalues, format_matrix, householder_qr,
    matmul, max_difference, qr_eigenvalues, to_matrix,
)


def run(config: dict) -> str:
    cfg = config.get("lab4", config)
    if not cfg:
        raise NumericalMethodError("В input JSON отсутствует раздел lab4.")
    a = to_matrix(cfg.get("A"), "lab4.A")
    if len(a) != len(a[0]):
        raise NumericalMethodError("В этой лабораторной задаче A должна быть квадратной: нужны собственные значения.")
    if "epsilon" not in cfg:
        raise NumericalMethodError("Укажите lab4.epsilon во входном JSON.")
    eps = float(cfg["epsilon"])
    max_iter = int(cfg.get("max_iterations", 20000))
    q, r = householder_qr(a)
    qr = matmul(q, r)
    eigenvalues, iterations, schur_form, converged = qr_eigenvalues(a, eps, max_iter)
    out = [
        "=" * 78,
        "ЛАБОРАТОРНАЯ РАБОТА № 4. QR-РАЗЛОЖЕНИЕ И СОБСТВЕННЫЕ ЗНАЧЕНИЯ",
        "=" * 78,
        f"Заданная точность ε = {eps:g}",
        "Исходная матрица A:\n" + format_matrix(a),
        "Матрица Q:\n" + format_matrix(q),
        "Матрица R:\n" + format_matrix(r),
        "Произведение Q · R:\n" + format_matrix(qr),
        f"Проверка QR: max|A-Q·R| = {max_difference(a, qr):.3e}",
        f"Проверка ортогональности: max|QᵀQ-E| = {max_difference(matmul([list(row) for row in zip(*q)], q), [[1.0 if i == j else 0.0 for j in range(len(q))] for i in range(len(q))]):.3e}",
        "\nСобственные значения исходной A, полученные собственным QR-алгоритмом (без NumPy): " + format_eigenvalues(eigenvalues),
        f"Число QR-итераций = {iterations}",
        "Статус квазитреугольной формы: " + ("достигнута" if converged else "лимит итераций достигнут; результаты могут быть приближёнными"),
        "Приближённая квазитреугольная матрица:\n" + format_matrix(schur_form),
        "\nПример матрицы с комплексно-сопряжёнными собственными значениями",
    ]
    if "complex_example" not in cfg:
        raise NumericalMethodError("Укажите lab4.complex_example во входном JSON для обязательной проверки комплексной пары.")
    example_matrix = to_matrix(cfg["complex_example"], "lab4.complex_example")
    if len(example_matrix) != len(example_matrix[0]):
        raise NumericalMethodError("lab4.complex_example должна быть квадратной матрицей.")
    out.append("Матрица-пример C:\n" + format_matrix(example_matrix))
    try:
        import numpy as np
        numpy_values = np.linalg.eigvals(np.array(example_matrix, dtype=float))
        out.append("Собственные значения C (NumPy, разрешённое исключение): " + format_eigenvalues([complex(v) for v in numpy_values]))
        pair_exists = any(v.imag > 1e-10 for v in numpy_values) and any(v.imag < -1e-10 for v in numpy_values)
        conjugate_check = all(any(abs(complex(w) - complex(v).conjugate()) < 1e-8 for w in numpy_values) for v in numpy_values)
        out.append(f"Проверка комплексно-сопряжённой пары: {'найдена' if pair_exists else 'для выбранного примера пара не обнаружена'}")
        out.append(f"Проверка сопряжённости: {'выполнена' if conjugate_check else 'не выполнена'}")
    except ImportError:
        out.append("NumPy не установлен. Для этой дополнительной проверки установите зависимость командой: python -m pip install -r requirements.txt")
    return "\n".join(out)


if __name__ == "__main__":
    from cli_utils import run_cli

    run_cli(run, "Лабораторная работа № 4")
