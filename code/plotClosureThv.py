import numpy as np
import matplotlib.pyplot as plt
import afterglowpy as grb
import paperPlots as pp

"""
def plotOnAxes(ax, jetType, Y, regime, thVs=None, thCs=None, thWs=None,
               name=None):

    if thVs is None and thCs is None and thWs is None:
        return

    N = 0
    if thVs is not None:
        N = max(N, len(thVs))
    if thCs is not None:
        N = max(N, len(thCs))
    if thWs is not None:
        N = max(N, len(thWs))

    for i in range(N):
        if thVs is not None:
            Y[0] = thVs[i]
        if thCs is not None:
            Y[2] = thCs[i]
"""

thV = 0.0
E0 = 1.0e53
thC = 0.05
thW = 6*thC
b = 4
L0 = 0
ts = 0
q = 0
n0 = 1.0e-3
p = 2.2
epse = 0.1
epsB = 0.001
xiN = 1.0
dL = 1.0e28
z = 0.5454

Y = np.array([thV, E0, thC, thW, b, L0, ts, q, n0, p, epse, epsB, xiN, dL])
Z = {'z': z, 'spread': True}


t = np.geomspace(1.0e3, 1.0e8, 100)
nu = np.empty(t.shape)
nu[:] = 1.0e14

figG, axG = plt.subplots(1, 1)
figPL, axPL = plt.subplots(1, 1)

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
thVs = np.array([thC, 2*thC, 4*thC, 6*thC, 8*thC])
thV = 0.3
thCs = np.array([thV/8, thV/6, thV/4, thV/2])

# for i, thV in enumerate(thVs):
for i, thC in enumerate(thCs):

    Y[0] = thV
    Y[2] = thC

    Y[3] = thV-0.05

    alG = pp.calcSlopeStructEff(0, Y, 'G')
    alPL = pp.calcSlopeStructEff(4, Y, 'G')

    tb = 0.2 * np.power(9*E0/(16*np.pi*n0*grb.mp*grb.c**5), 1.0/3.0)\
        * np.power(2*np.sin(0.5*(thC+thV)), 8.0/3.0)
    t0 = tb / 5

    i0 = np.searchsorted(t, t0)

    Fnu = grb.fluxDensity(t, nu, 0, 0, *Y)
    # axG.plot(t/tb, Fnu/Fnu[ib], color=colors[i])
    # axG.plot(t/tb, np.power(t/tb, alG),
    #          color=colors[i], ls='--')
    axG.plot(t, Fnu, color=colors[i])
    axG.plot(t, Fnu[i0] * np.power(t/t0, alG),
             color=colors[i], ls='--', lw=3.0, alpha=0.5)

    Fnu = grb.fluxDensity(t, nu, 4, 0, *Y)
    # axPL.plot(t/tb, Fnu/Fnu[ib], color=colors[i])
    # axPL.plot(t/tb, np.power(t/tb, alPL),
    #           color=colors[i], ls='--')
    axPL.plot(t, Fnu, color=colors[i])
    axPL.plot(t, Fnu[i0] * np.power(t/t0, alPL),
              color=colors[i], ls='--', lw=3.0, alpha=0.5)

figs = [figG, figPL]
axs = [axG, axPL]
for ax in axs:
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$t/t_b$')
    ax.set_ylabel(r'$F_\nu/F_\nu(t_b)$')
    # ax.set_xlim(1.0e-2, 1.0e1)
    # ax.set_ylim(1.0e-2, 1.0e2)
    ax.set_xlim(1.0e5, 1.0e8)
    ax.set_ylim(1.0e-9, 1.0e-4)

for fig in figs:
    fig.tight_layout()


plt.show()
