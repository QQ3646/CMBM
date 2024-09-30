import numpy as np
from sys import argv
import time

from numpy.ma.core import shape


class QR:
    def __init__(self, q: np.ndarray, r: np.ndarray):
        self.q = q
        self.r = r

    def check(self, a_transformed: np.ndarray) -> None:
        assert np.allclose(np.dot(self.q.transpose(), self.r), a_transformed)

def givens_rotations(a: np.ndarray, b: np.ndarray) -> tuple:
    def calculate_theta() -> tuple:
        from numpy import sqrt

        t = sqrt(a[i, i] ** 2 + a[j, i] ** 2)
        if abs(t) < 1e-6:
            return 1, 0
        else:
            return a[i, i] / t, -a[j, i] / t

    #                     theta = [cos \theta, sin \theta]
    def g(i: int, j: int, theta: tuple) -> np.ndarray:
        m = a.shape[0]

        result = np.eye(m)  # E
        result[i, i] = result[j, j] = theta[0]
        result[i, j] = -theta[1]
        result[j, i] = theta[1]  # G(i, j, \theta)

        return result

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
    gm = np.eye(m)

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

def nrmse(sigma: np.ndarray, rr: np.ndarray) -> float:
    c = 1 / max(sigma)
    ssum = np.sum(np.abs(rr - sigma) ** 2)
    ssum /= len(sigma)
    ssum = np.sqrt(ssum)
    return c * ssum


f = open("data_2.txt", "r")
lines = np.array(list(map(lambda x: list(map(float, x.split())), f.readlines())))
# lines = np.array([[0,0], [0.2, -2.321928094887362],[0.4, -1.3219280948873622],[0.6000000000000001, -0.736965594166206],[0.8, -0.3219280948873623]])

m = lines.shape[0]
n = int(argv[1])

a = np.empty((m, n))
for i in range(m):
    for j in range(n):
        a[i, j] = lines[i][0] ** j

a_c = a.copy()
llines = lines.copy()
b = lines[:, 1]

start = time.time()

qr, swaps = givens_rotations(a, b)
result = gauss(qr.r,  b)

rr = lines.copy()
for i, j in swaps:
    rr[[i, j], :] = rr[[j, i], :]
f = lambda x: sum([(x ** _i) * result[_i] for _i in range(n)])
for i in range(m):
    rr[i, 1] = f(rr[i, 0])

t = time.time() - start

rr = list(rr)
rr.sort(key=lambda x: x[0])
rr = np.array(rr)
print(f"N = {n}, t = {t}, NRMSE = {nrmse(llines[:, 1], rr[:, 1])}")
