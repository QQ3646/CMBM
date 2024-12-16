# Basis elements -- Chebyshev polynomials
# QR -- Householder transformation
# pryamoi metod
# N = 5
import os
import time

import numpy as np
import math

from matplotlib import pyplot as plt

f = lambda x: (x ** 4 + 14 * x ** 3 + 49 * x ** 2 + 32 * x - 12) * np.exp(x)

u_exact = lambda x: x ** 2 * (1 - x) ** 2 * np.exp(x)


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
        global_vector[curr_start_idx: curr_start_idx + N + 1] = ((points[1] - points[0]) / 2) ** 4 * f(x)

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

def u(x, Matrix, GlobalNodes):
    h = (GlobalNodes[1] - GlobalNodes[0]) / 2

    if x == 1.0:
        cell_id = len(GlobalNodes) - 2
    else:
        cell_id = int(x // (2 * h))

    x_c = (GlobalNodes[cell_id] + GlobalNodes[cell_id + 1]) / 2
    y = (x - x_c) / h
    ans = sum([Matrix[cell_id, i] * poly.give_polynomial(i)(y) for i in range(N + 1)])
    return ans

def plot_solution(X_cor, u_, u_ex):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16
    plt.clf()

    plt.grid(True)
    # рисуем точное решение
    plt.plot(X_cor, u_ex, label='Точное решение')
    # рисуем численное решение
    plt.plot(X_cor, u_, linestyle='--', label='Приближение')
    plt.legend(loc=4)

    path = "./pic/"
    if not os.path.exists(path):
        os.mkdir(path)

    plt.savefig(f"./pic/{K}.jpeg")

N = 4
err_rinf = []
err_ainf = []
times = []

with open("output.csv", "w") as fl:
    fl.write("K;$||E_a||_\\infty$;R;$||E_r||_\\infty$;R;$t_{sol}$;$\\mu(\\tilde(A))$;\n")
    for i in range(6):
        K = 5 * 2 ** i

        y_ = np.array([-np.cos((2 * i - 1) / (2 * (N + 1)) * np.pi) for i in range(1, N + 2)])

        poly = ChebyshevPolyManager()
        points = np.arange(0, 1 + 1e-10, 1 / K)

        global_matrix = np.zeros((K * (N + 5), K * (N + 1)))
        global_vector = np.zeros((K * (N + 5)))
        init_matrix()


        start = time.perf_counter()
        result = np.linalg.lstsq(global_matrix, global_vector)[0]
        end = time.perf_counter()

        X_cor = np.linspace(0, 1, 100)
        CoeffMatrix = np.zeros((K, N + 1))
        for j in range(K):
            CoeffMatrix[j] = result[j * (N + 1):(j + 1) * (N + 1)]

        u_ = np.array([u(x, CoeffMatrix, points) for x in X_cor])
        u_ex = np.array([u_exact(i) for i in X_cor])
        err_rinf.append(np.sqrt(np.trapz((u_ - u_ex) ** 2, X_cor) / np.trapz(u_ ** 2, X_cor)))
        err_ainf.append(np.max(np.abs(u_ - u_ex)))

        plot_solution(X_cor, u_, u_ex)

        times.append(end - start)
        if i == 0:
            fl.write(f"{K};{err_rinf[-1]:0.2e};-;{err_ainf[-1]:0.2e};-;{times[-1]:0.2e};{np.linalg.cond(global_matrix):0.2e};\n")
        else:
            fl.write(f"{K};{err_rinf[-1]:0.2e};{np.log2(err_rinf[-2]/err_rinf[-1]):0.2e};{err_ainf[-1]:0.2e};{np.log2(err_ainf[-2]/err_ainf[-1]):0.2e};{times[-1]:0.2e};{np.linalg.cond(global_matrix):0.2e};\n")

