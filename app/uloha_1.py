import numpy as np
from scipy.integrate import solve_ivp


# funkce řídící logiku algoritmu od začátku do konce
def solve_problem():
    eps = 1e-3  # mez pro stanovení konvergence
    k = 0  # iterace určení eta
    eta = 0  # počáteční odhad (nulová koncentrace v jádru katalytické částice)

    # závislé proměnné sady diferenciálních rovnic: y1, y2, p1, p2
    # dohromady tvoří vektor u = (y1, y2, p1, p2)

    # derivace závislých proměnných:
    dy1 = lambda x, y1, y2, p1, p2: y2
    dy2 = lambda x, y1, y2, p1, p2: y1**2 - y2 / x
    dp1 = lambda x, y1, y2, p1, p2: p2
    dp2 = lambda x, y1, y2, p1, p2: p2

    # vektor derivací ve formě použitelné pro scipy funkci solve_ivp
    du = lambda x, u: (dy1(x, *u), dy2(x, *u), dp1(x, *u), dp2(x, *u))

    # počáteční podmínky - hodnoty y1(0), y2(0) etc.
    y1_i = eta
    y2_i = 0
    p1_i = 1
    p2_i = 0

    u_i = (y1_i, y2_i, p1_i, p2_i)

    x_span = (1e-6, 1)

    sol = solve_ivp(du, x_span, u_i, method='RK45', max_step=1e-3)
    x = sol.t
    u = sol.y
    # koncové hodnoty...
    y1_e, y2_e, p1_e, p2_e = u[:, -1]


# spustit program
solve_problem()
