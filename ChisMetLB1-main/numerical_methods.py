from __future__ import annotations

import math
from typing import Sequence

Vector = list[float]
Matrix = list[list[float]]


class NumericalMethodError(ValueError):
    """Ошибка входных данных или невозможность применить метод."""


def to_matrix(value: Sequence[Sequence[float]], name: str = "A") -> Matrix:
    if not isinstance(value, (list, tuple)) or not value:
        raise NumericalMethodError(f"{name}: матрица должна быть непустым двумерным массивом.")
    rows: Matrix = []
    width = None
    for i, row in enumerate(value):
        if not isinstance(row, (list, tuple)) or not row:
            raise NumericalMethodError(f"{name}: строка {i + 1} пустая или некорректная.")
        if width is None:
            width = len(row)
        if len(row) != width:
            raise NumericalMethodError(f"{name}: строки матрицы имеют разную длину.")
        converted = []
        for item in row:
            try:
                number = float(item)
            except (TypeError, ValueError) as exc:
                raise NumericalMethodError(f"{name}: все элементы должны быть числами.") from exc
            if not math.isfinite(number):
                raise NumericalMethodError(f"{name}: NaN и бесконечности не допускаются.")
            converted.append(number)
        rows.append(converted)
    return rows


def to_vector(value: Sequence[float], name: str = "b") -> Vector:
    if not isinstance(value, (list, tuple)) or not value:
        raise NumericalMethodError(f"{name}: вектор должен быть непустым массивом.")
    result = []
    for item in value:
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise NumericalMethodError(f"{name}: все элементы должны быть числами.") from exc
        if not math.isfinite(number):
            raise NumericalMethodError(f"{name}: NaN и бесконечности не допускаются.")
        result.append(number)
    return result


def require_square(a: Matrix, name: str = "A") -> None:
    if any(len(row) != len(a) for row in a):
        raise NumericalMethodError(f"{name} должна быть квадратной матрицей.")


def identity(n: int) -> Matrix:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def transpose(a: Matrix) -> Matrix:
    return [list(row) for row in zip(*a)]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    if not a or not b or len(a[0]) != len(b):
        raise NumericalMethodError("Несогласованные размеры матриц для умножения.")
    bt = transpose(b)
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def matvec(a: Matrix, x: Vector) -> Vector:
    if not a or len(a[0]) != len(x):
        raise NumericalMethodError("Несогласованные размеры матрицы и вектора.")
    return [sum(ai * xi for ai, xi in zip(row, x)) for row in a]


def max_abs_matrix(a: Matrix) -> float:
    return max((abs(value) for row in a for value in row), default=0.0)


def max_abs_vector(v: Vector) -> float:
    return max((abs(value) for value in v), default=0.0)


def max_difference(a: Matrix, b: Matrix) -> float:
    if len(a) != len(b) or any(len(ra) != len(rb) for ra, rb in zip(a, b)):
        raise NumericalMethodError("Невозможно сравнить матрицы разных размеров.")
    return max((abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb)), default=0.0)


def vector_difference(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise NumericalMethodError("Невозможно сравнить векторы разной длины.")
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def residual_inf(a: Matrix, x: Vector, b: Vector) -> float:
    return vector_difference(matvec(a, x), list(b))


def lu_decomposition(a_in: Sequence[Sequence[float]], pivot_tol: float = 1e-14) -> tuple[Matrix, Matrix, Matrix, int]:
    """PA=LU с частичным выбором главного элемента по столбцу."""
    a = to_matrix(a_in, "A")
    require_square(a)
    n = len(a)
    u = [row[:] for row in a]
    l = identity(n)
    p = identity(n)
    sign = 1
    threshold = pivot_tol * max(1.0, max_abs_matrix(a))

    for k in range(n):
        pivot_row = max(range(k, n), key=lambda i: abs(u[i][k]))
        if abs(u[pivot_row][k]) <= threshold:
            raise NumericalMethodError(
                f"Матрица вырождена или плохо обусловлена: опорный элемент столбца {k + 1} близок к нулю."
            )
        if pivot_row != k:
            u[k], u[pivot_row] = u[pivot_row], u[k]
            p[k], p[pivot_row] = p[pivot_row], p[k]
            # Переставляем в L только уже рассчитанную часть строк.
            for j in range(k):
                l[k][j], l[pivot_row][j] = l[pivot_row][j], l[k][j]
            sign *= -1
        for i in range(k + 1, n):
            factor = u[i][k] / u[k][k]
            l[i][k] = factor
            u[i][k] = 0.0
            for j in range(k + 1, n):
                u[i][j] -= factor * u[k][j]
    return l, u, p, sign


def solve_lu(l: Matrix, u: Matrix, p: Matrix, b_in: Sequence[float]) -> Vector:
    b = to_vector(b_in, "b")
    n = len(l)
    if len(b) != n or len(u) != n or len(p) != n:
        raise NumericalMethodError("Размеры L, U, P и b не согласованы.")
    pb = matvec(p, b)
    y = [0.0] * n
    for i in range(n):
        y[i] = pb[i] - sum(l[i][j] * y[j] for j in range(i))
        if abs(l[i][i]) < 1e-15:
            raise NumericalMethodError("Нулевой диагональный элемент L.")
        y[i] /= l[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        denominator = u[i][i]
        if abs(denominator) < 1e-15:
            raise NumericalMethodError("Нулевой диагональный элемент U.")
        x[i] = (y[i] - sum(u[i][j] * x[j] for j in range(i + 1, n))) / denominator
    return x


def inverse_from_lu(l: Matrix, u: Matrix, p: Matrix) -> Matrix:
    n = len(l)
    cols = [solve_lu(l, u, p, [1.0 if i == j else 0.0 for i in range(n)]) for j in range(n)]
    return transpose(cols)


def determinant_from_lu(u: Matrix, permutation_sign: int) -> float:
    value = float(permutation_sign)
    for i in range(len(u)):
        value *= u[i][i]
    return value


def thomas_sweep(lower_in: Sequence[float], diagonal_in: Sequence[float],
                 upper_in: Sequence[float], rhs_in: Sequence[float],
                 tol: float = 1e-14) -> tuple[Vector, Vector, Vector]:
    """Метод прогонки: хранит только три диагонали и правую часть.

    lower, upper имеют длину n-1; diagonal и rhs — длину n.
    Возвращает x, alpha, beta, где x_i=alpha_i*x_(i+1)+beta_i.
    """
    lower = to_vector(list(lower_in), "lower") if len(lower_in) else []
    diagonal = to_vector(diagonal_in, "diagonal")
    upper = to_vector(list(upper_in), "upper") if len(upper_in) else []
    rhs = to_vector(rhs_in, "rhs")
    n = len(diagonal)
    if len(rhs) != n or len(lower) != n - 1 or len(upper) != n - 1:
        raise NumericalMethodError("Для прогонки нужны len(diagonal)=len(rhs)=n и len(lower)=len(upper)=n-1.")
    threshold = tol * max(1.0, max_abs_vector(diagonal), max_abs_vector(lower), max_abs_vector(upper))
    alpha = [0.0] * n
    beta = [0.0] * n
    denominator = diagonal[0]
    if abs(denominator) <= threshold:
        raise NumericalMethodError("Метод прогонки неприменим: первый знаменатель равен нулю или близок к нему.")
    if n > 1:
        alpha[0] = -upper[0] / denominator
    beta[0] = rhs[0] / denominator
    for i in range(1, n):
        denominator = diagonal[i] + lower[i - 1] * alpha[i - 1]
        if abs(denominator) <= threshold:
            raise NumericalMethodError(f"Метод прогонки неприменим: знаменатель на шаге {i + 1} близок к нулю.")
        if i < n - 1:
            alpha[i] = -upper[i] / denominator
        beta[i] = (rhs[i] - lower[i - 1] * beta[i - 1]) / denominator
    x = [0.0] * n
    x[-1] = beta[-1]
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]
    return x, alpha, beta


def jacobi_convergence_factor(a: Matrix) -> float:
    n = len(a)
    factors = []
    for i in range(n):
        if abs(a[i][i]) < 1e-15:
            return math.inf
        factors.append(sum(abs(a[i][j]) for j in range(n) if j != i) / abs(a[i][i]))
    return max(factors, default=0.0)


def seidel_sassenfeld_factor(a: Matrix) -> float:
    n = len(a)
    beta = [0.0] * n
    for i in range(n):
        if abs(a[i][i]) < 1e-15:
            return math.inf
        left = sum(abs(a[i][j]) * beta[j] for j in range(i))
        right = sum(abs(a[i][j]) for j in range(i + 1, n))
        beta[i] = (left + right) / abs(a[i][i])
    return max(beta, default=0.0)


def iterative_solve(a_in: Sequence[Sequence[float]], b_in: Sequence[float], eps: float,
                    method: str, max_iterations: int = 10000,
                    x0_in: Sequence[float] | None = None) -> tuple[Vector, int, float, float]:
    a = to_matrix(a_in, "A")
    require_square(a)
    b = to_vector(b_in, "b")
    n = len(a)
    if len(b) != n:
        raise NumericalMethodError("Длина b должна совпадать с размером A.")
    if eps <= 0 or not math.isfinite(eps):
        raise NumericalMethodError("Точность epsilon должна быть положительным конечным числом.")
    if max_iterations < 1:
        raise NumericalMethodError("max_iterations должен быть не меньше 1.")
    if any(abs(a[i][i]) < 1e-15 for i in range(n)):
        raise NumericalMethodError("Итерационные методы требуют ненулевых элементов на главной диагонали.")
    x = to_vector(x0_in, "x0") if x0_in is not None else [0.0] * n
    if len(x) != n:
        raise NumericalMethodError("Длина начального приближения x0 должна совпадать с размером A.")
    normalized_method = method.lower()
    if normalized_method not in {"jacobi", "seidel"}:
        raise NumericalMethodError("Метод должен быть 'jacobi' или 'seidel'.")

    for iteration in range(1, max_iterations + 1):
        old = x[:]
        new = old[:] if normalized_method == "jacobi" else x
        for i in range(n):
            left_sum = sum(a[i][j] * (old[j] if normalized_method == "jacobi" else new[j]) for j in range(i))
            right_sum = sum(a[i][j] * old[j] for j in range(i + 1, n))
            new[i] = (b[i] - left_sum - right_sum) / a[i][i]
        x = new[:]
        delta = vector_difference(x, old)
        res = residual_inf(a, x, b)
        if delta <= eps:
            return x, iteration, delta, res
    raise NumericalMethodError(
        f"Метод {method} не достиг epsilon={eps:g} за {max_iterations} итераций; "
        f"последняя невязка={residual_inf(a, x, b):.3e}. Попробуйте изменить матрицу, точность или лимит."
    )


def jacobi_rotations(a_in: Sequence[Sequence[float]], eps: float,
                     max_iterations: int = 100000) -> tuple[Vector, Matrix, int, float]:
    """Вращения Якоби для действительной симметричной матрицы."""
    original = to_matrix(a_in, "A")
    require_square(original)
    n = len(original)
    if eps <= 0 or not math.isfinite(eps):
        raise NumericalMethodError("Точность epsilon должна быть положительным конечным числом.")
    if max_iterations < 1:
        raise NumericalMethodError("max_iterations должен быть не меньше 1.")
    symmetry_tol = max(1e-12, eps * 10) * max(1.0, max_abs_matrix(original))
    for i in range(n):
        for j in range(i + 1, n):
            if abs(original[i][j] - original[j][i]) > symmetry_tol:
                raise NumericalMethodError("Метод вращений Якоби применим к симметричным матрицам; A несимметрична.")
    a = [row[:] for row in original]
    # Убираем возможный шум последнего разряда при проверке симметрии.
    for i in range(n):
        for j in range(i):
            a[i][j] = a[j][i] = (a[i][j] + a[j][i]) / 2.0
    v = identity(n)
    rotations = 0
    threshold = eps * max(1.0, max_abs_matrix(a))
    while rotations < max_iterations:
        p, q, largest = 0, 0, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(a[i][j]) > largest:
                    p, q, largest = i, j, abs(a[i][j])
        if largest <= threshold:
            break
        app, aqq, apq = a[p][p], a[q][q], a[p][q]
        tau = (aqq - app) / (2.0 * apq)
        if tau == 0:
            t = 1.0
        else:
            t = math.copysign(1.0, tau) / (abs(tau) + math.sqrt(1.0 + tau * tau))
        c = 1.0 / math.sqrt(1.0 + t * t)
        s = t * c
        a[p][p] = app - t * apq
        a[q][q] = aqq + t * apq
        a[p][q] = a[q][p] = 0.0
        for k in range(n):
            if k not in (p, q):
                akp, akq = a[k][p], a[k][q]
                a[k][p] = a[p][k] = c * akp - s * akq
                a[k][q] = a[q][k] = s * akp + c * akq
            vkp, vkq = v[k][p], v[k][q]
            v[k][p] = c * vkp - s * vkq
            v[k][q] = s * vkp + c * vkq
        rotations += 1
    else:
        raise NumericalMethodError(f"Метод Якоби не сошёлся за {max_iterations} вращений.")

    # Сортируем собственные значения по возрастанию с соответствующими столбцами V.
    order = sorted(range(n), key=lambda j: a[j][j])
    values = [a[j][j] for j in order]
    vectors = [[v[i][j] for j in order] for i in range(n)]
    max_offdiag = max((abs(a[i][j]) for i in range(n) for j in range(n) if i != j), default=0.0)
    return values, vectors, rotations, max_offdiag


def householder_qr(a_in: Sequence[Sequence[float]], tol: float = 1e-14) -> tuple[Matrix, Matrix]:
    """QR-разложение действительной матрицы методом отражений Хаусхолдера."""
    a = to_matrix(a_in, "A")
    m, n = len(a), len(a[0])
    r = [row[:] for row in a]
    q = identity(m)
    for k in range(min(m, n)):
        x = [r[i][k] for i in range(k, m)]
        norm_x = math.sqrt(sum(value * value for value in x))
        if norm_x <= tol:
            continue
        alpha = -math.copysign(norm_x, x[0] if x[0] != 0 else 1.0)
        v = x[:]
        v[0] -= alpha
        norm_v = math.sqrt(sum(value * value for value in v))
        if norm_v <= tol:
            continue
        v = [value / norm_v for value in v]
        # R <- H R, где H = I - 2 vv^T.
        for j in range(k, n):
            projection = 2.0 * sum(v[i - k] * r[i][j] for i in range(k, m))
            for i in range(k, m):
                r[i][j] -= projection * v[i - k]
        # Q <- Q H.
        for i in range(m):
            projection = 2.0 * sum(q[i][j] * v[j - k] for j in range(k, m))
            for j in range(k, m):
                q[i][j] -= projection * v[j - k]
    # Убираем численный шум ниже главной диагонали.
    threshold = tol * max(1.0, max_abs_matrix(a)) * 10
    for i in range(m):
        for j in range(min(i, n)):
            if abs(r[i][j]) < threshold:
                r[i][j] = 0.0
    return q, r


def hessenberg_reduction(a_in: Matrix, tol: float = 1e-14) -> Matrix:
    """Сведение квадратной матрицы к верхней форме Хессенберга подобными преобразованиями."""
    h = [row[:] for row in a_in]
    n = len(h)
    scale = max(1.0, max_abs_matrix(h))
    for k in range(n - 2):
        x = [h[i][k] for i in range(k + 1, n)]
        norm_x = math.sqrt(sum(value * value for value in x))
        if norm_x <= tol * scale:
            continue
        alpha = -math.copysign(norm_x, x[0] if x[0] != 0 else 1.0)
        v = x[:]
        v[0] -= alpha
        norm_v = math.sqrt(sum(value * value for value in v))
        if norm_v <= tol * scale:
            continue
        v = [value / norm_v for value in v]
        # H <- P H
        for j in range(k, n):
            proj = 2.0 * sum(v[i - k - 1] * h[i][j] for i in range(k + 1, n))
            for i in range(k + 1, n):
                h[i][j] -= proj * v[i - k - 1]
        # H <- H P
        for i in range(n):
            proj = 2.0 * sum(h[i][j] * v[j - k - 1] for j in range(k + 1, n))
            for j in range(k + 1, n):
                h[i][j] -= proj * v[j - k - 1]
        for i in range(k + 2, n):
            if abs(h[i][k]) < tol * scale * 10:
                h[i][k] = 0.0
    return h


def _wilkinson_shift(t: Matrix, m: int) -> float:
    if m < 2:
        return t[m - 1][m - 1]
    a, b = t[m - 2][m - 2], t[m - 2][m - 1]
    c, d = t[m - 1][m - 2], t[m - 1][m - 1]
    midpoint = (a + d) / 2.0
    discriminant = ((a - d) / 2.0) ** 2 + b * c
    if discriminant < 0:
        # Для комплексной пары вещественный сдвиг равен действительной части пары.
        return midpoint
    root = math.sqrt(discriminant)
    lambda1, lambda2 = midpoint + root, midpoint - root
    return lambda1 if abs(lambda1 - d) < abs(lambda2 - d) else lambda2


def qr_eigenvalues(a_in: Sequence[Sequence[float]], eps: float = 1e-10,
                   max_iterations: int = 20000) -> tuple[list[complex], int, Matrix, bool]:
    """Собственные значения через вещественные QR-итерации со сдвигом.

    В результате матрица приближается к квазитреугольной форме; блоки 1x1 и 2x2
    преобразуются в собственные значения аналитически. NumPy не используется.
    """
    original = to_matrix(a_in, "A")
    require_square(original)
    if eps <= 0 or not math.isfinite(eps):
        raise NumericalMethodError("Точность epsilon должна быть положительным конечным числом.")
    if max_iterations < 1:
        raise NumericalMethodError("max_iterations должен быть не меньше 1.")
    n = len(original)
    if n == 1:
        return [complex(original[0][0], 0.0)], 0, original, True
    t = hessenberg_reduction(original)
    scale = max(1.0, max_abs_matrix(t))
    threshold = eps * scale
    converged = False
    iterations = 0
    def is_quasi_triangular(matrix: Matrix) -> bool:
        # Значимые элементы ниже первой поддиагонали не должны оставаться.
        below = sum(abs(matrix[i][j]) for i in range(n) for j in range(i - 1))
        if below > threshold:
            return False
        for row in range(1, n - 1):
            if abs(matrix[row][row - 1]) > threshold and abs(matrix[row + 1][row]) > threshold:
                return False
        return True

    for iterations in range(1, max_iterations + 1):
        if is_quasi_triangular(t):
            converged = True
            break
        mu = _wilkinson_shift(t, n)
        shifted = [row[:] for row in t]
        for i in range(n):
            shifted[i][i] -= mu
        q, r = householder_qr(shifted, tol=max(1e-15, eps * 1e-3))
        t = matmul(r, q)
        for i in range(n):
            t[i][i] += mu
        for i in range(n):
            for j in range(i - 1):
                if abs(t[i][j]) <= threshold:
                    t[i][j] = 0.0
    else:
        iterations = max_iterations
        converged = is_quasi_triangular(t)

    # Извлекаем собственные значения из 1x1 и 2x2 блоков.
    vals: list[complex] = []
    i = n - 1
    block_tol = max(threshold * 10, 1e-12 * scale)
    while i >= 0:
        if i == 0 or abs(t[i][i - 1]) <= block_tol:
            vals.append(complex(t[i][i], 0.0))
            i -= 1
        else:
            aa, bb = t[i - 1][i - 1], t[i - 1][i]
            cc, dd = t[i][i - 1], t[i][i]
            trace = aa + dd
            determinant = aa * dd - bb * cc
            discriminant = complex(trace * trace - 4.0 * determinant, 0.0)
            root = discriminant ** 0.5
            vals.extend([(trace + root) / 2.0, (trace - root) / 2.0])
            i -= 2
    vals.sort(key=lambda z: (round(z.real, 12), round(z.imag, 12)))
    return vals, iterations, t, converged


def format_matrix(a: Matrix, precision: int = 8) -> str:
    if not a:
        return "[]"
    rendered = [[f"{v:.{precision}g}" for v in row] for row in a]
    widths = [max(len(rendered[i][j]) for i in range(len(rendered))) for j in range(len(rendered[0]))]
    return "\n".join("[ " + "  ".join(value.rjust(widths[j]) for j, value in enumerate(row)) + " ]" for row in rendered)


def format_vector(v: Sequence[float], precision: int = 10) -> str:
    return "[" + ", ".join(f"{value:.{precision}g}" for value in v) + "]"


def format_eigenvalues(values: Sequence[complex], precision: int = 10) -> str:
    def one(z: complex) -> str:
        if abs(z.imag) < 10 ** (-precision + 2):
            return f"{z.real:.{precision}g}"
        sign = "+" if z.imag >= 0 else "-"
        return f"{z.real:.{precision}g} {sign} {abs(z.imag):.{precision}g}i"
    return "[" + ", ".join(one(z) for z in values) + "]"
