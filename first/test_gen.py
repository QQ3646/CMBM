import numpy as np

print('[', end='')
n = 5
for i in range(n):
    print(f"[{1 / n * i}, {np.log2(1 / n * i)}]{',' if i != n - 1 else ''}", end='')

print(']', end='')