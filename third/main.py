# 5 variant
import os
import time

import numpy as np
from matplotlib import pyplot as plt

C = 1
L = 2
a = 0.6

q = lambda x: np.sin(np.pi * x / L)
# M(x) : https://www.wolframalpha.com/input?i2d=true&i=Divide%5B1%2C2%5D+*+%5C%2840%29x+*+Integrate%5Bsin%5C%2840%29Divide%5Bpi+*+s%2CL%5D%5C%2841%29%2C%7Bs%2C0%2CL%7D%5D%5C%2841%29+-+Integrate%5Bsin%5C%2840%29Divide%5Bpi+*+s%2CL%5D%5C%2841%29%2C%7Bs%2C0%2Cx%7D%5D
M = lambda x: (L * x) / np.pi - (L * (np.pi * x - L * np.sin((np.pi * x) / L))) / np.pi ** 2

E = 200 * 10 ** 9
b = 0.1
h = 0.1

f = k = lambda x: (12 * M(x)) / (E * b * h ** 3)

# Exact solution:
# https://www.wolframalpha.com/input?i2d=true&i=D%5Bu%5C%2840%29x%5C%2841%29%2C%7Bx%2C2%7D%5D+%3D+12*Divide%5B%5C%2840%29Divide%5B%5C%2840%29L+x%5C%2841%29%2C%CF%80%5D+-+Divide%5B%5C%2840%29L+-+L+cos%5C%2840%29Divide%5B%5C%2840%29%CF%80+x%5C%2841%29%2CL%5D%5C%2841%29%5C%2841%29%2C%CF%80%5D%5C%2841%29%2CSubscript%5Bx%2C1%5DSubscript%5Bx%2C2%5DPower%5BSubscript%5Bx%2C3%5D%2C3%5D%5D
# where x_1 = E, x_2 = b, x_3 = h
#
# After variables defining we get
# https://www.wolframalpha.com/input?i2d=true&i=D%5Bu%5C%2840%29x%5C%2841%29%2C%7Bx%2C2%7D%5D+%3D+12*Divide%5B%5C%2840%29Divide%5B%5C%2840%292x%5C%2841%29%2C%CF%80%5D+-+Divide%5B%5C%2840%292+-+2+cos%5C%2840%29Divide%5B%5C%2840%29%CF%80+x%5C%2841%29%2C2%5D%5C%2841%29%5C%2841%29%2C%CF%80%5D%5C%2841%29%2C200+*+Power%5B10%2C9%5D+*+0.1+*+Power%5B0.1%2C3%5D%5D
# Looks like ancient horror, Lovecraft would have loved it.
# After that we need to define c_1 (const value) and c_2 (x^2 coefficient).
#
# uE(x) = 0, we get that c_1 = 1.54807 * 10^(-7)
# https://www.wolframalpha.com/input?i2d=true&i=Subscript%5Bc%2C1%5D-1.54807+*+Power%5B10%2C-7%5D*cos%5C%2840%290%5C%2841%29+%3D+0
#
# uE(L = 2) = 0, we get that c_2 = -2.7483 * 10^(-8)
# https://www.wolframalpha.com/input?i2d=true&i=Subscript%5Bc%2C2%5D*2+%2B+1.54807+*+Power%5B10%2C%5C%2840%29-7%5C%2841%29%5D%2B6.3662*Power%5B10%2C-8%5D*Power%5B2%2C3%5D-1.90986*Power%5B10%2C-7%5D*Power%5B2%2C2%5D+-+1.54807*Power%5B10%2C-7%5D*cos%5C%2840%291.5708*2%5C%2841%29+%3D+0
#
# After all, we can write down the solution:
# uE = lambda x: (-2.7483 * 10 ** (-8)) * x + \
#                (1.54807 * 10 ** (-7)) + \
#                (6.3662 * 10 ** (-8)) * x ** 3 + \
#                (1.90986 * 10 ** (-7)) * x ** 2 - \
#                (1.54807 * 10 ** (-7)) * np.cos(1.5708 * x)
uE = lambda x: (1.3076 * 10 ** (-13)) * x - \
               (6.15959 * 10 ** (-9) * L ** 4) * np.sin(np.pi * x / L)

local_matrix = np.float64([[1, -1], [-1, 1]])
local_rMatrix = np.float64([[2, 1], [1, 2]])
def init_mj():
    cx = 1
    N_ = N + 1

    global_matrix = np.zeros((N_, N_))
    global_rMatrix = np.zeros((N_, N_))

    for i in range(N):
        global_matrix[i: i + 2, i: i + 2] += cx / Le * local_matrix
        global_rMatrix[i: i + 2, i: i + 2] += -Le / 6 * local_rMatrix

    # Zeroing
    global_rMatrix[0] = 0
    global_rMatrix[-1] = 0

    global_matrix[0] = 0
    global_matrix[-1] = 0
    global_matrix[0, 0] = 1
    global_matrix[-1, -1] = 1

    f_vector = f(points)
    global_rMatrix = np.dot(global_rMatrix, f_vector)

    return global_matrix, global_rMatrix


class fem_solution:
    def __init__(self, points, coeffs):
        self.points = points
        self.coeffs = coeffs

    def value(self, x):
        if x in points:
            return self.coeffs[np.where(points == x)[0][0]]
        else:
            i = len(points[points < x]) - 1
            u_left, u_right = coeffs[i:i + 2]
            x_left, x_right = points[i:i + 2]
            a1 = (u_right - u_left) / (x_right - x_left)
            a0 = (u_left * x_right - u_right * x_left) / (x_right - x_left)
            return a1 * x + a0

def draw_plot(points, coeffs, N):
    num_solution = fem_solution(points, coeffs).value
    X = np.linspace(0, L, 10000 + 1, endpoint=True)
    u = np.float64([uE(x) for x in X])
    u_h = np.float64([num_solution(x) for x in X])

    plt.clf()
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16

    color_exact, color_num = '#1f77b4', '#ff7f0e'
    plt.xlim(0, L)
    plt.xticks(np.linspace(0, L, num=9, endpoint=True))
    plt.grid(True)
    plt.plot(X, u, label='Точное решение', color=color_exact, linestyle='-', linewidth=2)
    plt.plot(X, u_h, label='Решение МКЭ', color=color_num, linestyle='--', linewidth=2)
    plt.scatter(points, coeffs, color=color_num, s=20)
    plt.legend()

    path = "./pic/"
    if not os.path.exists(path):
        os.mkdir(path)

    plt.savefig(f"./pic/{N}.jpeg")

Q = 10000
error_inf = []
error_2 = []
cond = []

with open("output.csv", "w") as fl:
    fl.write("N;||E_r||_\\infty;R;||E_r||_{L_2};R;\\nu([K]);t (sec);\n")
    for i in range(1, 8):
        N = 2 ** i

        points = np.linspace(0, L, num=N + 1, endpoint=True)


        Le = points[1] - points[0]

        start = time.time()
        gm, grm = init_mj()
        coeffs = np.linalg.solve(gm, grm)

        num_solution = fem_solution(points, coeffs).value
        t = time.time() - start

        X = np.linspace(0, L, Q, endpoint=True)
        u = np.float64([uE(x) for x in X])
        u_h = np.float64([num_solution(x) for x in X])

        error_inf.append(max(np.abs(u - u_h)) / max(np.abs(u)))
        error_2.append(np.sqrt(np.trapz((u - u_h) ** 2, X) / np.trapz(u ** 2, X)))
        cond.append(np.linalg.cond(gm))

        draw_plot(points, coeffs, N=N)
        if i == 1:
            fl.write(f"{N};{error_inf[-1]:0.2e};-;{error_2[-1]:0.2e};-;{cond[-1]:0.2e};{t:0.4e};\n")
        else:
            fl.write(f"{N};{error_inf[-1]:0.2e};{np.log2(error_inf[-2] / error_inf[-1]):0.2e};{error_2[-1]:0.2e};{np.log2(error_2[-2] / error_2[-1]):0.2e};{cond[-1]:0.2e};{t:0.4e};\n")

