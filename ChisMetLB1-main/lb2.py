from time import perf_counter

from numerical_methods import (
    NumericalMethodError, format_vector, iterative_solve, jacobi_convergence_factor,
    lu_decomposition, matvec, residual_inf, seidel_sassenfeld_factor, solve_lu,
    to_matrix, to_vector,
)


def run(config: dict) -> str:
    cfg = config.get("lab2", config)
    if not cfg:
        raise NumericalMethodError("В input JSON отсутствует раздел lab2.")
    a = to_matrix(cfg.get("A"), "lab2.A")
    b = to_vector(cfg.get("b"), "lab2.b")
    if "epsilon" not in cfg:
        raise NumericalMethodError("Укажите lab2.epsilon во входном JSON.")
    eps = float(cfg["epsilon"])
    max_iter = int(cfg.get("max_iterations", 10000))
    x0 = cfg.get("x0")
    if len(a) != len(b):
        raise NumericalMethodError("Для итерационных методов длина b должна совпадать с размером A.")

    l, u, p, _ = lu_decomposition(a)
    reference = solve_lu(l, u, p, b)
    start = perf_counter()
    xj, nj, dj, rj = iterative_solve(a, b, eps, "jacobi", max_iter, x0)
    jacobi_seconds = perf_counter() - start
    start = perf_counter()
    xs, ns, ds, rs = iterative_solve(a, b, eps, "seidel", max_iter, x0)
    seidel_seconds = perf_counter() - start
    qj = jacobi_convergence_factor(a)
    qs = seidel_sassenfeld_factor(a)
    out = [
        "=" * 78,
        "ЛАБОРАТОРНАЯ РАБОТА № 2. МЕТОД ПРОСТЫХ ИТЕРАЦИЙ И МЕТОД ЗЕЙДЕЛЯ",
        "=" * 78,
        f"Заданная точность ε = {eps:g}",
        f"Максимальное число итераций = {max_iter}",
        "Эталонное решение (вычислено через LU-разложение): x* = " + format_vector(reference),
        "\nМетод простых итераций (Якоби)",
        "Решение: " + format_vector(xj),
        f"Количество итераций: {nj}",
        f"Время вычисления на этом запуске: {jacobi_seconds:.6e} с",
        f"Последнее изменение ||x(k)-x(k-1)||∞ = {dj:.3e}",
        f"Невязка ||Ax-b||∞ = {rj:.3e}",
        f"Отклонение от эталона ||x-x*||∞ = {max(abs(t-r) for t, r in zip(xj, reference)):.3e}",
        f"Оценка достаточного условия сходимости Якоби q = {qj:.6g}: " + ("выполнено (q<1)" if qj < 1 else "не гарантируется этим достаточным условием"),
        "\nМетод Зейделя",
        "Решение: " + format_vector(xs),
        f"Количество итераций: {ns}",
        f"Время вычисления на этом запуске: {seidel_seconds:.6e} с",
        f"Последнее изменение ||x(k)-x(k-1)||∞ = {ds:.3e}",
        f"Невязка ||Ax-b||∞ = {rs:.3e}",
        f"Отклонение от эталона ||x-x*||∞ = {max(abs(t-r) for t, r in zip(xs, reference)):.3e}",
        f"Оценка достаточного условия сходимости Зейделя (условие Зейделя/Зассенфельда) β = {qs:.6g}: " + ("выполнено (β<1)" if qs < 1 else "не гарантируется этим достаточным условием"),
        "\nСравнение скорости сходимости",
    ]
    if nj < ns:
        out.append(f"Якоби потребовал меньше итераций: {nj} против {ns}.")
    elif ns < nj:
        out.append(f"Зейдель потребовал меньше итераций: {ns} против {nj}.")
    else:
        out.append(f"Оба метода потребовали одинаковое число итераций: {nj}.")
    faster = "Якоби" if jacobi_seconds < seidel_seconds else "Зейдель"
    out.append(f"По измеренному времени в этом запуске быстрее: {faster} (время зависит от оборудования и фоновой нагрузки).")
    out.append("Примечание: число итераций зависит от A, b, начального приближения и ε; это не универсальная оценка для всех систем.")
    return "\n".join(out)


if __name__ == "__main__":
    from cli_utils import run_cli

    run_cli(run, "Лабораторная работа № 2")
