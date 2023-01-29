import numpy as np
from numpy.linalg import solve

np.set_printoptions(precision=3)


# funkce řídící logiku algoritmu od začátku do konce
def solve_problem():
    # počet úseků, tedy počet členů je n+1, m+1
    n = 10
    m = 10
    # krok
    k = 1 / n
    h = 1 / m
    # tabelace x_i, t_j
    x_i = np.linspace(0, 1, num=(n + 1))
    t_j = np.linspace(0, 1, num=(m + 1))

    # počáteční podmínka - u v bodech x_i, t = 0
    u_i0 = 4 * (x_i - .5)**2

    u_final = np.zeros((m + 1, n + 1))
    u_final[0, :] = u_i0

    C = k / h / h
    D = lambda u_ij: 1 + C + u_ij
    u_ij = u_i0

    A = np.zeros((n - 1, n - 1), dtype='float64')
    for i in range(2, n - 1):
        k = i - 1
        A[k, (k - 1):(k + 2)] = [-C / 2, D(u_ij[i]), -C / 2]
    A[0, 0:2] = [D(u_ij[1]), -C / 2]
    A[-1, -2:] = [-C / 2, D(u_ij[-2])]

    b = np.zeros((n - 1, 1), dtype='float64')
    for i in range(1, n):
        k = i - 1
        b[k] = C / 2 * (u_ij[i - 1] - 2 * u_ij[i] + u_ij[i + 1]) + u_ij[i]
    b[(0, -1), 0] += C / 2

    u_new = solve(A, b)

    # print(np.concatenate((A,u_new,b), axis=1))
    u_final[1, (0, -1)] = 1
    u_final[1, 1:-1] = u_new.T
    print(u_final)


# vykreslit 3D graf z řešení parciální diferenciální rovnice
def draw_result(x_i, t_j, u):
    pass


# spustit program
solve_problem()
