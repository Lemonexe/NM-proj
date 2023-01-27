from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt


# funkce řídící logiku algoritmu od začátku do konce
def solve_problem():
    eta = 0.5  # počáteční odhad
    eps = 1e-4  # mez pro stanovení konvergence eta (dle volby)
    k = 0  # čítač iterací při určení eta
    eta_prev = eta + 2*eps  # předchozí hodnota eta; nyní pouze dočasná hodnota pro umožnění prvního průchodu while cyklem

    max_iter = 1e3  # omezení počtu iterací jako bezpečnostní opatření

    # závislé proměnné sady diferenciálních rovnic: y1, y2, p1, p2
    # dohromady tvoří vektor u = (y1, y2, p1, p2)

    # derivace závislých proměnných:
    dy1 = lambda x, y1, y2: y2
    dy2 = lambda x, y1, y2: y1**2 - y2/x
    dp1 = lambda x, y1, y2, p1, p2: p2
    dp2 = lambda x, y1, y2, p1, p2: 2*y1*p1 - p2/x

    # vektor derivací ve formě použitelné pro scipy funkci solve_ivp
    du = lambda x, u: (dy1(x, *u[0:2]), dy2(x, *u[0:2]), dp1(x, *u), dp2(x, *u))

    # rozsah nezávislé veličiny x; nelze počítat přesně od nuly (neurčitý výraz dy2)
    x_span = (1e-8, 1)

    # cyklus pro opakovaný výpočet eta, dokud není splněno kritérium konvergence
    while abs(eta - eta_prev) > eps:
        k += 1
        if k > max_iter:
            raise Exception(f'Konvergence nedosažena při {k} iteracích')

        # získat vektor počátečních podmínek, viz deklarace get_u_i
        u_i = get_u_i(eta)

        # získat řešení pro dané eta
        sol = solve_ivp(du, x_span, u_i, method='RK45', max_step=1e-3)
        u = sol.y
        # okrajové hodnoty y1(1), y2(1), p1(1), p2(1)
        y1_e, y2_e, p1_e, p2_e = u[:, -1]

        # vyčíslení theta(eta) (vyjadřuje residuum okrajové podmínky) a její derivace d(theta(eta)) / d(eta)
        theta = y1_e - 1
        dtheta = p1_e

        # uložení staré hodnoty eta a získání nové Newtonovou metodou
        eta_prev = eta
        eta = eta - theta/dtheta
        print('k = {:.0f}    eta ={:7.4f}    theta ={:7.4f}'.format(k, eta, theta))

    print('Dosažena konvergence, zahájeno poslední řešení')

    # získat řešení pro konečnou hodnotu eta
    # přičemž závislé proměnné sady diferenciálních rovnic jsou nyní pouze u = (y1, y2)
    # toto řešení je provedeno s větší přesností (nižší max_step)
    du = lambda x, u: (dy1(x, *u), dy2(x, *u))
    u_i = get_u_i(eta)[0:2]
    sol = solve_ivp(du, x_span, u_i, method='RK45', max_step=1e-4)
    x = sol.t
    u = sol.y
    y1 = u[0, :]
    y2 = u[1, :]
    print('Okrajové hodnoty výsledného řešení:')
    print('y(1) ={:7.4f}    dy(1)/dx ={:7.4f}'.format(y1[-1], y2[-1]))
    print('Pro úplné řešení viz graf')
    draw_result(x, y1, y2)


# pro dané eta vygenerovat vektor hodnot počátečních podmínek u = y1(0), y2(0), p1(0), p2(0)
def get_u_i(eta):
    y1_i = eta
    y2_i = 0
    p1_i = 1
    p2_i = 0
    return (y1_i, y2_i, p1_i, p2_i)


# vykreslit graf z řešení diferenciální rovnice
def draw_result(x, y1, y2):
    plt.plot(x, y1, '-k', label='y')
    plt.plot(x, y2, ':k', label='dy/dx')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('Řešení úlohy 1')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='lower right')
    plt.show()


# spustit program
solve_problem()
