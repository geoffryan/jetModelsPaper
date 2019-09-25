import numpy as np
import matplotlib.pyplot as plt
import paperPlots as pp


def calc_gG(thV, thC):

    athV = np.atleast_1d(thV)
    g = np.empty(athV.shape)
    for i in range(len(g)):
        Y = np.empty(5)
        Y[0] = athV[i]
        Y[2] = thC
        Y[3] = 20*thC
        g[i] = pp.calcGEff(0, Y)

    return g


def calc_gPL(thV, thC, b):

    athV = np.atleast_1d(thV)
    g = np.empty(athV.shape)
    for i in range(len(g)):
        Y = np.empty(5)
        Y[0] = athV[i]
        Y[2] = thC
        Y[3] = 20*thC
        Y[4] = b
        g[i] = pp.calcGEff(4, Y)

    return g


x = np.linspace(0.0, 15.0, 300)

gG = calc_gG(x, 1.0)
gPL2 = calc_gPL(x, 1.0, 2)
gPL6 = calc_gPL(x, 1.0, 6)
gPL9 = calc_gPL(x, 1.0, 9)

p = 2.17
al = 0.9
dal = 0.06
al1 = 2*p
al2 = 3.0
s = 1.0

g = (8*al + 3*al1 - 3*s) / (al2 - al)

dgda = (8*al2 + 3*al1 - 3*s) / ((al2-al)*(al2-al))

dg = np.fabs(dgda) * dal

print("al = {0:.3f} +/- {1:.3f}".format(al, dal))
print("g  = {0:.3f} +/- {1:.3f}".format(g, dg))

lw = 1.5
"""
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
"""
fig, ax = plt.subplots(1, 1)
ax.plot(x, gG, lw=lw,   ls='-', color='C2', label=r'Gaussian')
ax.plot(x, gPL2, lw=lw, ls='-.', color='C1', label=r'PL $b=2$')
ax.plot(x, gPL6, lw=lw, ls=':', color='C3', label=r'PL $b=6$')
ax.plot(x, gPL9, lw=lw, ls='--', color='C0', label=r'PL $b=9$')

ax.fill_between(x, g-dg, g+dg, color='lightgrey', label='GW170817')

"""
ax.legend(loc='upper left', fontsize=legendsize)
ax.set_xlabel(r'$\theta_V / \theta_C$', fontsize=fontsize)
ax.set_ylabel(r'g', fontsize=fontsize)
ax.tick_params(labelsize=ticksize)
"""
ax.legend(loc='upper left')
ax.set_xlabel(r'$\theta_V / \theta_C$')
ax.set_ylabel(r'g')
ax.tick_params()

ax.set_ylim(0, 16)
ax.set_xlim(x.min(), x.max())

fig.tight_layout()

plotname = "g_plot.pdf"
print("Saving " + plotname)
fig.savefig(plotname)
