# numerical_utils.py
import math

def norm_vector(v):
    """Euclidean norm of a vector."""
    return math.sqrt(sum(x ** 2 for x in v))

def norm_vector_inf(v):
    """Infinity norm (maximum absolute value) of a vector."""
    return max(abs(x) for x in v)

def solve_gauss(A, b):
    """
    Solves linear system A * x = b using Gaussian elimination with partial pivoting.
    Pure Python implementation.
    """
    n = len(b)
    # Make a copy to avoid mutating original matrices
    M = [row[:] for row in A]
    y = b[:]

    for i in range(n):
        # Find pivot
        max_el = abs(M[i][i])
        max_row = i
        for k in range(i + 1, n):
            if abs(M[k][i]) > max_el:
                max_el = abs(M[k][i])
                max_row = k

        # Swap rows
        M[i], M[max_row] = M[max_row], M[i]
        y[i], y[max_row] = y[max_row], y[i]

        if abs(M[i][i]) < 1e-12:
            raise ValueError("Matrix is singular or nearly singular.")

        # Eliminate column
        for k in range(i + 1, n):
            c = -M[k][i] / M[i][i]
            for j in range(i, n):
                if i == j:
                    M[k][j] = 0
                else:
                    M[k][j] += c * M[i][j]
            y[k] += c * y[i]

    # Back substitution
    x = [0.0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        sum_ax = sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - sum_ax) / M[i][i]

    return x