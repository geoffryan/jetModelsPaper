import numpy as np
import grbpy as grb
import matplotlib.pyplot as plt
# import paperPlots as pp


def plotPanel(fig, gs0, t, nu, thV, Y, Z, modelLegend=False):

    gs = gs0.subgridspec(3, 1, hspace=0.0)
    axF = fig.add_subplot(gs[0:-1, 0])
    axa = fig.add_subplot(gs[-1, 0])

    tfac = 0.3

    tb = t * (1.0 + tfac)
    ta = t / (1.0 - tfac)

    c = ['C0', 'C2', 'C1', 'C3']
    ls = ['--', '-', '-.', ':']

    thC = Y[2]

    Y[0] = thV

    FnuTH = grb.fluxDensity(t, nu, -1, 0, *Y, **Z)
    FnuTHa = grb.fluxDensity(ta, nu, -1, 0, *Y, **Z)
    FnuTHb = grb.fluxDensity(tb, nu, -1, 0, *Y, **Z)

    FnuG = grb.fluxDensity(t, nu, 0, 0, *Y, **Z)
    FnuGa = grb.fluxDensity(ta, nu, 0, 0, *Y, **Z)
    FnuGb = grb.fluxDensity(tb, nu, 0, 0, *Y, **Z)

    Y[4] = 2
    FnuPL1 = grb.fluxDensity(t, nu, 4, 0, *Y, **Z)
    FnuPL1a = grb.fluxDensity(ta, nu, 4, 0, *Y, **Z)
    FnuPL1b = grb.fluxDensity(tb, nu, 4, 0, *Y, **Z)

    Y[4] = 6
    # Y[2] = np.sqrt(3)*thC
    FnuPL2 = grb.fluxDensity(t, nu, 4, 0, *Y, **Z)
    FnuPL2a = grb.fluxDensity(ta, nu, 4, 0, *Y, **Z)
    FnuPL2b = grb.fluxDensity(tb, nu, 4, 0, *Y, **Z)
    Y[2] = thC

    alTH = np.log(FnuTHb/FnuTHa) / np.log(tb/ta)
    alG = np.log(FnuGb/FnuGa) / np.log(tb/ta)
    alPL1 = np.log(FnuPL1b/FnuPL1a) / np.log(tb/ta)
    alPL2 = np.log(FnuPL2b/FnuPL2a) / np.log(tb/ta)

    axF.plot(t, FnuTH, label='Top-Hat', color=c[0], ls=ls[0])
    axF.plot(t, FnuG, label='Gaussian Jet', color=c[1], ls=ls[1])
    axF.plot(t, FnuPL1, label='Power Law Jet $b=2$', color=c[2], ls=ls[2])
    axF.plot(t, FnuPL2, label='Power Law Jet $b=6$', color=c[3], ls=ls[3])

    axa.plot(t, alTH, color=c[0], ls=ls[0])
    axa.plot(t, alG, color=c[1], ls=ls[1])
    axa.plot(t, alPL1, color=c[2], ls=ls[2])
    axa.plot(t, alPL2, color=c[3], ls=ls[3])

    axa.set_xscale('log')
    axa.set_yscale('linear')
    axF.set_xscale('log')
    axF.set_yscale('log')

    axF.get_xaxis().set_visible(False)

    axa.set_yticks([-2, 0, 2, 4])

    axF.set_ylabel(r'$F_{\nu}$ (1 keV)')
    axa.set_xlabel(r'$t$ (s)')
    axa.set_ylabel(r'$\alpha$')

    axF.set_xlim(t[0], t[-1])
    axF.set_ylim(1.0e-10, 1.0e-3)
    axa.set_xlim(t[0], t[-1])
    axa.set_ylim(-3, 5)

    if modelLegend:
        axF.legend(loc='lower left')

    axF.text(0.95, 0.95, r'$\theta_{{\mathrm{{obs}}}} = $ {0:.2f} rad'.format(
             thV), transform=axF.transAxes, horizontalalignment='right',
             verticalalignment='top')


thV = 0.0
E0 = 1.0e53
thC = 0.08
thW = 3*thC
b = 2
L0 = 0
ts = 0
q = 0
n0 = 1.0
p = 2.2
epse = 0.1
epsB = 0.01
xiN = 1.0
dL = 1.0e28
z = 0.5454

Y0 = np.array([thV, E0, thC, thW, b, L0, ts, q, n0, p, epse, epsB, xiN, dL])
Z = {'z': z}

fig = plt.figure(figsize=(8, 6))
gs = fig.add_gridspec(2, 2)

t = np.geomspace(1.0e4, 1.0e8, 100)
nu = np.empty(t.shape)
nu[:] = 2.41798926e17  # 1 keV

plotPanel(fig, gs[0, 0], t, nu, 0.0, Y0, Z, modelLegend=True)
plotPanel(fig, gs[0, 1], t, nu, 2*thC, Y0, Z)
plotPanel(fig, gs[1, 0], t, nu, 4*thC, Y0, Z)
plotPanel(fig, gs[1, 1], t, nu, 6*thC, Y0, Z)

fig.tight_layout()

figname = "lc_thV_model_multi.pdf"
print("Saving " + figname)
fig.savefig(figname)
plt.close(fig)
