# Basis elements -- Chebyshev polynomials
# QR -- Householder transformation
# pryamoi metod
# N = 5
import time

import numpy as np
import math

f = lambda x: (x ** 4 + 14 * x ** 3 + 49 * x ** 2 + 32 * x - 12) * np.exp(x)

class ChebyshevPolyManager:
    _polynomials = [lambda x: 1, lambda x: x]
    _coeffs = [[1], [0, 1]]

    _temp_derivative = []

    def give_polynomial(self, n: int):
        if n < len(self._polynomials):
            # Fastpass, if poly is already computed
            return self._polynomials[n]
        else:
            # If not, we need to compute all previous polynomials
            while len(self._polynomials) - 1 < n:
                self._compute_next_polynomial()
            return self._polynomials[-1]

    def _compute_next_polynomial(self):
        coeff = [0] + [2 * i for i in self._coeffs[-1]]

        for i, c in enumerate(self._coeffs[-2]):
            coeff[i] -= c

        self._coeffs.append(coeff)
        n = len(self._coeffs) - 1
        self._polynomials.append(lambda x: self._poly(n, x))

    def _poly(self, n: int, x: float) -> float:
        s = 0
        for i, c in enumerate(self._coeffs[n]):
            s += c * pow(x, i)
        return s

    def _current_derivative(self, x: float) -> float:
        s = 0
        for i, c in enumerate(self._temp_derivative):
            s += c * pow(x, i)
        return s

    def get_derivative(self, n: int, order: int, x: float):
        if order > n:
            return 0
        else:
            self.give_polynomial(n)
            self._temp_derivative = [0] * (n - order + 1)

            for i in range(order, n + 1):
                mult = math.factorial(i) / math.factorial(i - order)
                self._temp_derivative[i - order] = mult * self._coeffs[n][i]

            return self._current_derivative(x)


def init_matrix():
    # Левая ячейка
    # Ур-я коллокации
    for i in range(0, N + 1):
        for j in range(0, N + 1):
            global_matrix[i, j] = poly.get_derivative(j, 4, y_[i])
    x = np.array([y_[i - 1] * ((points[1] - points[0]) / 2) + (points[0] + points[1]) / 2 for i in range(1, N + 1 + 1)])
    global_vector[0: N + 1] = ((points[1] - points[0]) / 2) ** 4 * f(x)

    # Условия согласования
    # В текущей ячейке
    global_matrix[N + 1, 0:N + 1] = [poly.give_polynomial(i)(1) + poly.get_derivative(i, 1, 1) for i in range(N + 1)]
    global_matrix[N + 2, 0:N + 1] = [poly.get_derivative(i, 2, 1) + poly.get_derivative(i, 3, 1) for i in range(N + 1)]
    # В соседней ячейке
    global_matrix[N + 1, N + 1:(N + 1) * 2] = [-poly.give_polynomial(i)(-1) - poly.get_derivative(i, 1, -1) for i in range(N + 1)]
    global_matrix[N + 2, N + 1:(N + 1) * 2] = [-poly.get_derivative(i, 2, -1) - poly.get_derivative(i, 3, -1) for i in range(N + 1)]


    # Граничные условия
    global_matrix[N + 3, 0:N + 1] = [poly.give_polynomial(i)(-1) for i in range(N + 1)]
    global_matrix[N + 4, 0:N + 1] = [poly.get_derivative(i, 1, -1) for i in range(N + 1)]


    # Внутренние ячейки
    for i in range(1, K - 1):
        curr_start_idx = i * (N + 5)
        curr_start_row_idx = i * (N + 1)

        # Ур-я коллокации
        for k in range(0, N + 1):
            for j in range(0, N + 1):
                global_matrix[curr_start_idx + k, curr_start_row_idx + j] = poly.get_derivative(j, 4, y_[k])
        x = np.array([y_[j - 1] * ((points[1] - points[0]) / 2) + ((points[i] + points[i + 1]) / 2) for j in range(1, N + 1 + 1)])
        global_vector[curr_start_idx: curr_start_idx + N + 1] = ((points[1] - points[0]) / 2) ** 4 * f(x) # TODO

        # Условия согласования
        global_matrix[curr_start_idx + N + 1, curr_start_row_idx: curr_start_row_idx + N + 1] = \
            [poly.give_polynomial(j)(-1) - poly.get_derivative(j, 1, -1) for j in range(N + 1)]
        global_matrix[curr_start_idx + N + 1 + 1, curr_start_row_idx: curr_start_row_idx + N + 1] = \
            [poly.get_derivative(j, 2, -1) - poly.get_derivative(j, 3, -1) for j in range(N + 1)]

        global_matrix[curr_start_idx + N + 1, curr_start_row_idx - 1 * (N + 1): curr_start_row_idx] = \
            [-poly.give_polynomial(j)(1) + poly.get_derivative(j, 1, 1) for j in range(N + 1)]
        global_matrix[curr_start_idx + N + 1 + 1, curr_start_row_idx - 1 * (N + 1): curr_start_row_idx] = \
            [-poly.get_derivative(j, 2, 1) + poly.get_derivative(j, 3, 1) for j in range(N + 1)]


        global_matrix[curr_start_idx + N + 1 + 2, curr_start_row_idx: curr_start_row_idx + N + 1] = \
            [poly.give_polynomial(j)(1) + poly.get_derivative(j, 1, 1) for j in range(N + 1)]
        global_matrix[curr_start_idx + N + 1 + 3, curr_start_row_idx: curr_start_row_idx + N + 1] = \
            [poly.get_derivative(j, 2, 1) + poly.get_derivative(j, 3, 1) for j in range(N + 1)]

        global_matrix[curr_start_idx + N + 1 + 2, curr_start_row_idx + 1 * (N + 1): curr_start_row_idx + 2 * (N + 1):] = \
            [-poly.give_polynomial(j)(-1) - poly.get_derivative(j, 1, -1) for j in range(N + 1)]
        global_matrix[curr_start_idx + N + 1 + 3, curr_start_row_idx + 1 * (N + 1): curr_start_row_idx + 2 * (N + 1):] = \
            [-poly.get_derivative(j, 2, -1) - poly.get_derivative(j, 3, -1) for j in range(N + 1)]

    # Крайняя правая ячейка
    curr_start_idx = (K - 1) * (N + 5)
    curr_start_row_idx = (K - 1) * (N + 1)

    for i in range(curr_start_idx, curr_start_idx + N + 1):
        global_matrix[i, curr_start_row_idx: curr_start_row_idx + N + 1] = [poly.get_derivative(j, 4, y_[i - curr_start_idx]) for j in range(N + 1)]
    x = np.array([y_[i - 1] * ((points[1] - points[0]) / 2) + ((points[K - 1] + points[K]) / 2) for i in range(1, N + 1 + 1)])
    global_vector[curr_start_idx: curr_start_idx + N + 1] = ((points[1] - points[0]) / 2) ** 4 * f(x)

    # Условия согласования
    # В текущей ячейке
    global_matrix[curr_start_idx + N + 1, curr_start_row_idx : curr_start_row_idx + N + 1] = [poly.give_polynomial(i)(-1) - poly.get_derivative(i, 1, -1) for i in range(N + 1)]
    global_matrix[curr_start_idx + N + 2, curr_start_row_idx : curr_start_row_idx + N + 1] = [poly.get_derivative(i, 2, -1) - poly.get_derivative(i, 3, -1) for i in range(N + 1)]
    # В соседней ячейке
    global_matrix[curr_start_idx + N + 1, curr_start_row_idx - (N + 1) : curr_start_row_idx] = [-poly.give_polynomial(i)(1) + poly.get_derivative(i, 1, 1) for i in range(N + 1)]
    global_matrix[curr_start_idx + N + 2, curr_start_row_idx - (N + 1) : curr_start_row_idx] = [-poly.get_derivative(i, 2, 1) + poly.get_derivative(i, 3, 1) for i in range(N + 1)]


    # Граничные условия
    global_matrix[curr_start_idx + N + 3, curr_start_row_idx: curr_start_row_idx + N + 1] = [poly.give_polynomial(i)(1) for i in range(N + 1)]
    global_matrix[curr_start_idx + N + 4, curr_start_row_idx: curr_start_row_idx + N + 1] = [poly.get_derivative(i, 1, 1) for i in range(N + 1)]

K = 5
N = 4

y_ = np.array([-np.cos((2 * i - 1) / (2 * (N + 1)) * np.pi) for i in range(1, N + 2)])

poly = ChebyshevPolyManager()
points = np.arange(0, 1 + 1e-10, 1 / K)

global_matrix = np.zeros((K * (N + 5), K * (N + 1)))
global_vector = np.zeros((K * (N + 5)))
init_matrix()


def givens_rotations(a: np.ndarray, b: np.ndarray):
    # theta = [cos \theta, sin \theta]
    def calculate_theta() -> tuple:
        from numpy import sqrt

        t = sqrt(a[i, i] ** 2 + a[j, i] ** 2)
        if abs(t) < 1e-6:
            return 1, 0
        else:
            return a[i, i] / t, -a[j, i] / t

    def main_element_selection(a: np.ndarray, b: np.ndarray) -> list:
        def swap(i: int, j: int) -> None:
            a[[i, j], :] = a[[j, i], :]
            b[[i, j]] = b[[j, i]]
            swaps.append((i, j))

        swaps = []

        for i in range(a.shape[1]):
            #   ind, value
            m = (i, a[i, i])
            for j in range(i + 1, a.shape[0]):
                if abs(m[1]) < abs(a[j, i]):
                    m = (j, a[j, i])

            if i != m[0]: swap(i, m[0])

        return swaps

    def rotation_matrix_dot(a: np.array, i: tuple, j: tuple, theta: tuple) -> None:
        # rows a[i], a[j] is reference to element from matrix, so i need to copy them
        a_i = a[i, :].copy()
        a_j = a[j, :].copy()

        a[i, :] = a_i[:] * theta[0] - a_j[:] * theta[1]
        a[j, :] = a_i[:] * theta[1] + a_j[:] * theta[0]

    def rotation_vec_dot(a: np.array, i: tuple, j: tuple, theta: tuple) -> None:
        a_i = a[i].copy()
        a_j = a[j].copy()

        a[i] = a_i * theta[0] - a_j * theta[1]
        a[j] = a_i * theta[1] + a_j * theta[0]

    swaps = main_element_selection(a, b)
    gm = np.eye(a.shape[0])

    for i in range(a.shape[1]):
        for j in range(i + 1, a.shape[0]):
            if abs(a[j, i]) >= 1e-6:
                theta = calculate_theta()
                rotation_vec_dot(b, i, j, theta)
                rotation_matrix_dot(a, i, j, theta)

    return a

def gauss(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    for i in range(a.shape[1] - 1, 0, -1):
        for j in range(i - 1, -1, -1):
            c = -1 * a[j, i] / a[i, i]
            a[j, i] += a[i, i] * c
            b[j] += b[i] * c
    x = []
    for i in range(a.shape[1]):
        x.append(b[i] / a[i, i])
    return np.array(x)

SLAE_solving_time = []

start = time.perf_counter()
ans = givens_rotations(global_matrix, global_vector)
result = gauss(ans,  global_vector)
end = time.perf_counter()
SLAE_solving_time.append('%.2e' % (end - start))