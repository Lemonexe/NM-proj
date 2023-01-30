import click  # pomůcka pro CLI interakci s programem
import numpy as np
from numpy.linalg import solve
from matplotlib import pyplot as plt
from matplotlib import cm


@click.command()
@click.option('-n', default=100, metavar='n', help='počet úseků v intervalu x_i')
@click.option('-m', default=100, metavar='m', help='počet úseků v intervalu t_j')
@click.option('--delta', default=1.0, metavar='delta', help='bezrozměrná rychlostní konstanta reakce (jen na hraní)')
@click.option('--diff', default=1.0, metavar='diff', help='bezrozměrný difúzní koeficient (jen na hraní)')
@click.option('--log', is_flag=True, metavar='log', help='vypsat matici U do konzole (doporučeno jen pro malé n,m)')
@click.option('--colormap', is_flag=True, metavar='colormap', help='vykreslit graf jako barevnou mapu namísto 3D')
# funkce řídící logiku algoritmu od začátku do konce
def solve_problem(n, m, delta, diff, log, colormap):
    k = 1 / n  # krok k pro x_i
    h = 1 / m  # krok h pro t_j

    # tabelace x_i, t_j
    x_i = np.linspace(0, 1, num=(n + 1))
    t_j = np.linspace(0, 1, num=(m + 1))
    # pozn. n, m jsou počty úseků intervalů, tedy počty členů jsou n+1, m+1
    # např. x_i má n+1 bodů, první je x_i[0], poslední je x_i[n]

    # počáteční podmínka: u v bodech x_i, t = 0
    u_i0 = 4 * (x_i - .5)**2

    # matice bodů řešení U, kde u[j, i] = u(x_i, t_j)
    U = np.zeros((m + 1, n + 1))
    U[0, :] = u_i0  # první řádek (j=0) známe

    # C = alpha, D = beta (oproti protokolu přejmenováno pro snazší zápis)
    C = diff * k / h / h  # alpha = k/h^2
    D = lambda u: 1 + C + delta*k*u  # prostřední člen pásu matice (je závislý na u[j, i] z předchozího řádku u)

    # je třeba získat nový řádek u[j+1, :], přičemž jeho krajní body, čili i=0, i=n jsou přímo vyjádřeny okrajovou podmínkou
    # prostřední body, odpovídající i=1 až i=(n-1) včetně, jsou řešením matice soustavy A · u_new = b, matice je tedy tvaru (n-1)·(n-1)
    # důsledkem je, že číslování matice A a vektoru b se liší od číslování u,x,t; např. druhý člen u_new, což je u[j+1, 1], odpovídá pozici A[r, 0] pro každou rovnici r
    # přičemž rovnice r je vyjádřením PDR pro bod x_(r+1), t_(j+1)
    # cenou za linearizaci problému je to, že se matice A mění každou iteraci j (člen D závislý na u[j, :])

    for j in range(0, m):
        u_ij = U[j, :]  # předchozí řádek

        A = np.zeros((n - 1, n - 1), dtype='float64')

        # první a poslední řádek matice A, pro i=1, i=(n-1)
        A[0, 0:2] = [D(u_ij[1]), -C / 2]
        A[-1, -2:] = [-C / 2, D(u_ij[-2])]

        # ostatní řádky matice A pro i = 2 až i=(n-2) včetně
        for i in range(2, n - 1):
            r = i - 1
            A[r, (r - 1):(r + 2)] = [-C / 2, D(u_ij[i]), -C / 2]

        # vektor pravých stran b je tvaru (n-1)·1, přičemž k je opět čítač PDR
        b = np.zeros((n - 1, 1), dtype='float64')
        # všechny řádky b obsahují tento výraz...
        for i in range(1, n):
            r = i - 1
            b[r] = C / 2 * (u_ij[i - 1] - 2 * u_ij[i] + u_ij[i + 1]) + u_ij[i]

        # ...k prvním a posledním řádkům je ještě přičten výraz vyjadřující známé hodnoty pro i=0, i=n
        b[(0, -1), 0] += C / 2

        # nyní je vyřešena soustava rovnic a sestaven nový řádek u[j+1, :]
        u_new = solve(A, b)

        U[j + 1, (0, -1)] = 1  # okrajové hodnoty známe z okrajové podmínky
        U[j + 1, 1:-1] = u_new.T  # prostřední hodnoty z řešení soustavy rovnic
    click.echo('Řešení dokončeno, viz graf')

    if log:
        np.set_printoptions(precision=3)
        click.echo(U)
    elif colormap:
        draw_colormap(x_i, t_j, U)
    else:
        draw_3D_plot(x_i, t_j, U)


# vykreslit barevnou mapu z řešení parciální diferenciální rovnice
def draw_colormap(x_i, t_j, U):
    plt.title('Řešení úlohy 2 (u znázorněno barvou)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.imshow(U, interpolation='none', extent=[np.min(x_i), np.max(x_i), np.max(t_j), np.min(t_j)])
    plt.colorbar()
    plt.show()


# vykreslit 3D graf z řešení parciální diferenciální rovnice
def draw_3D_plot(x_i, t_j, U):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X, Y = np.meshgrid(x_i, t_j)  # připravit síť pro rendering 3D grafu
    surface = ax.plot_surface(X, Y, U, cmap=cm.gnuplot, linewidth=0)
    fig.colorbar(surface, shrink=0.5, aspect=10)
    ax.set_title('Řešení úlohy 2')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    plt.show()


# spustit program
if __name__ == '__main__':  # možné pouze z konzole
    solve_problem()
