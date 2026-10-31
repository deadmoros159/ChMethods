from numerical_methods import (
    NumericalMethodError, determinant_from_lu, format_matrix, format_vector,
    identity, inverse_from_lu, lu_decomposition, matmul, max_difference,
    matvec, residual_inf, solve_lu, thomas_sweep, to_matrix, to_vector,
)


def run(config: dict) -> str:
    sections = config.get("lab1", config)
    out = ["=" * 78, "ЛАБОРАТОРНАЯ РАБОТА № 1. ЛУ-РАЗЛОЖЕНИЕ И МЕТОД ПРОГОНКИ", "=" * 78]
    lu_cfg = sections.get("lu")
    if not lu_cfg:
        raise NumericalMethodError("В input JSON отсутствует раздел lab1.lu.")
    a = to_matrix(lu_cfg.get("A"), "lab1.lu.A")
    b = to_vector(lu_cfg.get("b"), "lab1.lu.b")
    if len(a) != len(b):
        raise NumericalMethodError("Для LU: длина b должна совпадать с размером A.")
    l, u, p, sign = lu_decomposition(a)
    x = solve_lu(l, u, p, b)
    inv = inverse_from_lu(l, u, p)
    determinant = determinant_from_lu(u, sign)
    lu_product = matmul(l, u)
    pa = matmul(p, a)
    a_inv = matmul(a, inv)
    out += [
        "\n1.1. LU-разложение с частичным выбором главного элемента",
        "Исходная матрица A:\n" + format_matrix(a),
        "Вектор b: " + format_vector(b),
        "Матрица перестановок P:\n" + format_matrix(p),
        "Матрица L:\n" + format_matrix(l),
        "Матрица U:\n" + format_matrix(u),
        "Произведение L · U:\n" + format_matrix(lu_product),
        "Решение Ax=b: x = " + format_vector(x),
        f"Невязка ||Ax-b||∞ = {residual_inf(a, x, b):.3e}",
        "Обратная матрица A⁻¹:\n" + format_matrix(inv),
        f"det(A) = {determinant:.12g}",
        "Проверка A · A⁻¹ ≈ E:\n" + format_matrix(a_inv),
        f"max|A·A⁻¹-E| = {max_difference(a_inv, identity(len(a))):.3e}",
        "Проверка L · U ≈ P · A:",
        f"max|L·U-P·A| = {max_difference(lu_product, pa):.3e}",
    ]

    sweep = sections.get("sweep")
    if not sweep:
        raise NumericalMethodError("В input JSON отсутствует раздел lab1.sweep.")
    lower = sweep.get("lower", [])
    diagonal = sweep.get("diagonal")
    upper = sweep.get("upper", [])
    rhs = sweep.get("b")
    solution, alpha, beta = thomas_sweep(lower, diagonal, upper, rhs)
    out += [
        "\n1.2. Метод прогонки (хранятся только три диагонали и правая часть)",
        "Нижняя диагональ a: " + format_vector(lower),
        "Главная диагональ b: " + format_vector(diagonal),
        "Верхняя диагональ c: " + format_vector(upper),
        "Правая часть d: " + format_vector(rhs),
        "Прогоночные коэффициенты α: " + format_vector(alpha),
        "Прогоночные коэффициенты β: " + format_vector(beta),
        "Решение системы: x = " + format_vector(solution),
    ]
    return "\n".join(out)


if __name__ == "__main__":
    from cli_utils import run_cli

    run_cli(run, "Лабораторная работа № 1")
