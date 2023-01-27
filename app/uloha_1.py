import numpy as np
from scipy.integrate import solve_ivp


# funkce řídící logiku algoritmu od začátku do konce
def solve_problem():
    eta = 0.5  # počáteční odhad (nulová koncentrace v jádru katalytické částice)
    eps = 1e-3  # mez pro stanovení konvergence eta
    k = 0  # čítač iterací při určení eta
    eta_prev = eta + 2 * eps  # předchozí hodnota eta; nyní pouze dočasná hodnota pro umožnění prvního průchodu while cyklem

    max_iter = 1e3  # bezpečnostní opatření

    # závislé proměnné sady diferenciálních rovnic: y1, y2, p1, p2
    # dohromady tvoří vektor u = (y1, y2, p1, p2)

    # derivace závislých proměnných:
    dy1 = lambda x, y1, y2: y2
    dy2 = lambda x, y1, y2: y1**2 - y2 / x
    dp1 = lambda x, y1, y2, p1, p2: p2
    dp2 = lambda x, y1, y2, p1, p2: 2 * y1 * p1 - p2 / x

    # vektor derivací ve formě použitelné pro scipy funkci solve_ivp
    du = lambda x, u: (dy1(x, *u[0:2]), dy2(x, *u[0:2]), dp1(x, *u), dp2(x, *u))

    # cyklus pro opakovaný výpočet eta, dokud není splněno kritérium konvergence
    while abs(eta - eta_prev) > eps:
        if k > max_iter:
            raise Exception(f'Konvergence nedosažena při {k} iteracích')
            break

        # počáteční podmínky - hodnoty y1(0), y2(0), p1(0), p2(0)
        y1_i = eta
        y2_i = 0
        p1_i = 1
        p2_i = 0
        u_i = (y1_i, y2_i, p1_i, p2_i)

        # rozsah nezávislé veličiny x; nelze počítat přesně od nuly (neurčitý výraz dy2)
        x_span = (1e-6, 1)

        # získat řešení pro dané eta
        sol = solve_ivp(du, x_span, u_i, method='RK45', max_step=1e-3)
        x = sol.t
        u = sol.y
        # okrajové hodnoty y1(1), y2(1), p1(1), p2(1)
        y1_e, y2_e, p1_e, p2_e = u[:, -1]

        # vyčíslení theta(eta) (vyjadřuje residuum okrajové podmínky) a její derivace d(theta(eta)) / d(eta)
        theta = y1_e - 1
        dtheta = p1_e

        # uložení staré hodnoty eta a získání nové Newtonovou metodou
        k += 1
        eta_prev = eta
        eta = eta - theta / dtheta
        print('k = {:.0f}; eta = {:6.3f}; theta = {:6.3f}'.format(k, eta, theta))

    print('Dosažena konvergence')


# spustit program
solve_problem()
