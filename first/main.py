import os

import numpy as np
from sys import argv
import time

from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation


class QR:
    def __init__(self, q: np.ndarray, r: np.ndarray):
        self.q = q
        self.r = r

    def check(self, a_transformed: np.ndarray) -> None:
        assert np.allclose(np.dot(self.q.transpose(), self.r), a_transformed)


def givens_rotations(a: np.ndarray, b: np.ndarray) -> tuple:
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

    result = QR(gm, a)

    return result, swaps

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

def nrmse_score(sigma: np.ndarray, rr: np.ndarray) -> float:
    c = 1 / max(sigma)
    ssum = np.sum(np.abs(rr - sigma) ** 2)
    ssum /= len(sigma)
    ssum = np.sqrt(ssum)
    return c * ssum


def main(file: str, n: int, mode: str):
    # функция для построения графиков: data - исходные данные (x, y), approximation_data - данные численного алгоритма (x, f(x))
    def plot_graphics(data, approximation_data):
        fig = plt.figure(figsize=(12, 8), frameon=True)
        plt.style.use('ggplot')
        plt.rcParams["mathtext.fontset"] = "cm"
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams['font.size'] = 37
        plt.rcParams['text.color'] = 'black'
        plt.rcParams['xtick.color'] = 'black'
        plt.rcParams['ytick.color'] = 'black'
        plt.rcParams['axes.labelcolor'] = 'black'
        ax = fig.add_subplot(111)

        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['left'].set_color('black')

        ax.set(facecolor='w')
        ax.grid('axis = "both"', color='gray')

        ax.set_xlabel('$x$', labelpad=-10)
        ax.set_ylabel('$y$', rotation=0, labelpad=20)

        ax.plot(data[:, 0], data[:, 1], color='blue', linestyle='-', linewidth=3, label='Данные')

        ax.plot(data[:, 0], approximation_data, color='red', linestyle='-', linewidth=2,
                label=f'Приближение при N = {n}{mode}')
        ax.legend(loc=4)

        path = "./pic/"
        if not os.path.exists(path):
            os.makedir(path)

        plt.savefig(f"./pic/{file[:file.find('.')]}{mode}-{n}.jpeg")

    n += 1

    f = open("./data/" + file, "r")
    lines = np.array(list(map(lambda x: list(map(float, x.split())), f.readlines())))

    m = lines.shape[0]

    a = np.empty((m, n))
    for i in range(m):
        for j in range(n):
            a[i, j] = lines[i][0] ** j

    aT = a.transpose()
    if mode == "-qr":
        cond = np.linalg.cond(a)
        condStr = "A"
    elif mode == "-ne":
        cond = np.linalg.cond(np.dot(aT, a))
        condStr = "A^T * A"
    else:
        print("Incorrect run mode. You should run with '-qr' for QR decomposition or '-ne' for normal equation method.")
        exit()

    a_c = a.copy()
    llines = lines.copy()
    b = lines[:, 1]

    start = time.time()

    if mode == "-qr":
        # QR-decomposition
        qr, swaps = givens_rotations(a, b)

        result = gauss(qr.r,  b)
    elif mode == "-ne":
        aT = a.transpose()

        # Normal eq
        b = np.dot(aT, b)
        a = np.dot(aT, a)

        result = np.linalg.solve(a, b)
    else:
        print("Incorrect run mode. You should run with '-qr' for QR decomposition or '-ne' for normal equation method.")
        exit()

    rr = lines.copy()
    f = lambda x: sum([(x ** _i) * result[_i] for _i in range(n)])
    for i in range(m):
        rr[i, 1] = f(rr[i, 0])

    t = time.time() - start

    rr = list(rr)
    rr.sort(key=lambda x: x[0])
    rr = np.array(rr)
    plot_graphics(llines, rr[:, 1])

    if argv[-1] != "-gencsv":
        print(f"N = {n - 1}, cond_2({condStr}) = {cond:.2e}, t = {t:0.2e}, NRMSE = {nrmse_score(llines[:, 1], rr[:, 1]):0.2e}")
    else:
        print(f"{n - 1}; {condStr}; {cond:.2e}; {t:.4e}; {nrmse_score(llines[:, 1], rr[:, 1]):0.2e};")


# You should run script with "python main.py *data_NUMBER*.txt *polynomial degree* -*mode (qr or ne)* (-genvsc)"
if __name__ == '__main__':
    main(argv[1], int(argv[2]), argv[3])