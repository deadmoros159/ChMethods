"""Автоматические тесты численных методов. Запуск: python3 -m unittest -v"""
import json
import subprocess
import sys
import unittest
from pathlib import Path

import lb1
import lb2
import lb3
import lb4
from numerical_methods import (
    householder_qr,
    jacobi_rotations,
    lu_decomposition,
    matmul,
    max_difference,
    qr_eigenvalues,
    solve_lu,
    thomas_sweep,
)

BASE_DIR = Path(__file__).resolve().parent


class NumericalMethodsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data1 = json.loads((BASE_DIR / "data_lb1.json").read_text(encoding="utf-8"))
        cls.data2 = json.loads((BASE_DIR / "data_lb2.json").read_text(encoding="utf-8"))
        cls.data3 = json.loads((BASE_DIR / "data_lb3.json").read_text(encoding="utf-8"))
        cls.data4 = json.loads((BASE_DIR / "data_lb4.json").read_text(encoding="utf-8"))

    def test_lu_reconstruction_and_solution(self):
        cfg = self.data1["lu"]
        l, u, p, _ = lu_decomposition(cfg["A"])
        self.assertLess(max_difference(matmul(l, u), matmul(p, cfg["A"])), 1e-10)
        x = solve_lu(l, u, p, cfg["b"])
        residual = [
            sum(aij * xj for aij, xj in zip(row, x)) - rhs
            for row, rhs in zip(cfg["A"], cfg["b"])
        ]
        self.assertLess(max(abs(value) for value in residual), 1e-10)

    def test_sweep(self):
        cfg = self.data1["sweep"]
        x, alpha, beta = thomas_sweep(
            cfg["lower"], cfg["diagonal"], cfg["upper"], cfg["b"]
        )
        self.assertEqual(len(alpha), 3)
        self.assertEqual(len(beta), 3)
        self.assertLess(max(abs(value - 1.0) for value in x), 1e-10)

    def test_iteration_methods(self):
        report = lb2.run(self.data2)
        self.assertIn("Количество итераций", report)
        self.assertIn("Сравнение скорости сходимости", report)

    def test_jacobi_rotations(self):
        cfg = self.data3
        values, vectors, _, _ = jacobi_rotations(cfg["A"], cfg["epsilon"])
        av = matmul(cfg["A"], vectors)
        vl = [
            [vectors[i][j] * values[j] for j in range(len(values))]
            for i in range(len(values))
        ]
        self.assertLess(max_difference(av, vl), 1e-8)

    def test_qr_reconstruction(self):
        a = self.data4["A"]
        q, r = householder_qr(a)
        self.assertLess(max_difference(a, matmul(q, r)), 1e-10)

    def test_complex_pair(self):
        values, _, _, converged = qr_eigenvalues([[0, -1], [1, 0]], 1e-10)
        self.assertTrue(converged)
        self.assertEqual(len(values), 2)
        self.assertTrue(any(abs(value.imag - 1.0) < 1e-8 for value in values))
        self.assertTrue(any(abs(value.imag + 1.0) < 1e-8 for value in values))

    def test_arbitrary_dimension(self):
        # Размерность не зашита: проверяем матрицу 4x4 вместо примера 3x3.
        a = [[8, -1, 0, 0], [-1, 8, -1, 0], [0, -1, 8, -1], [0, 0, -1, 8]]
        b = [7, 6, 6, 7]
        l, u, p, _ = lu_decomposition(a)
        x = solve_lu(l, u, p, b)
        self.assertEqual(len(x), 4)
        residual = [sum(a[i][j] * x[j] for j in range(4)) - b[i] for i in range(4)]
        self.assertLess(max(abs(value) for value in residual), 1e-10)
        q, r = householder_qr(a)
        self.assertLess(max_difference(a, matmul(q, r)), 1e-10)

    def test_all_reports(self):
        self.assertIn("выбором главного элемента", lb1.run(self.data1))
        self.assertIn("Зейделя", lb2.run(self.data2))
        self.assertIn("A · V", lb3.run(self.data3))
        self.assertIn("NumPy", lb4.run(self.data4))

    def test_four_command_line_launches(self):
        for script, data in [
            ("lb1.py", "data_lb1.json"),
            ("lb2.py", "data_lb2.json"),
            ("lb3.py", "data_lb3.json"),
            ("lb4.py", "data_lb4.json"),
        ]:
            with self.subTest(script=script):
                result = subprocess.run(
                    [sys.executable, script, data],
                    cwd=BASE_DIR,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, msg=result.stderr)
                self.assertIn("ЛАБОРАТОРНАЯ РАБОТА", result.stdout)


if __name__ == "__main__":
    unittest.main()
