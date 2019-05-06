import numpy as np
import matplotlib.pyplot as plt


def calc_gG(thV, thC):

    return 0.25*thV*thV/(thC*thC)


def calc_gPL(thV, thC, b):

    th = 0.2*thV + 2.0*thC/3.0

    g = (thV-th) * b * th / (thC*thC + th*th)

    return g


x = np.linspace(0.0, 10.0, 300)

gG = calc_gG(x, 1.0)
gPL2 = calc_gPL(x, 1.0, 2)
gPL4 = calc_gPL(x, 1.0, 4)
gPL6 = calc_gPL(x, 1.0, 6)

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

lw = 3.0
fontsize = 24
ticksize = 18
legendsize = 18
"""
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
"""
fig, ax = plt.subplots(1, 1)
ax.plot(x, gG, lw=lw, label=r'Gaussian')
ax.plot(x, gPL2, lw=lw, label=r'Power Law $b=2$')
ax.plot(x, gPL4, lw=lw, label=r'Power Law $b=4$')
ax.plot(x, gPL6, lw=lw, label=r'Power Law $b=6$')

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
