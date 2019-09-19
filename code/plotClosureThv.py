import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import afterglowpy as grb
import paperPlots as pp


thV = 0.0
E0 = 1.0e53
thC = 0.05
thW = 5*thC
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

Y2 = Y.copy()
Y2[4] = 2
Y6 = Y.copy()
Y6[4] = 6

t = np.geomspace(1.0e3, 1.0e8, 100)
nu = np.empty(t.shape)
nu[:] = 1.0e14

figG, axG = plt.subplots(1, 1)
figPL, axPL = plt.subplots(1, 1)

width = 7.5
height = 3

figM = plt.figure(figsize=(width, height))
gs = figM.add_gridspec(1, 3, wspace=0.0)
axMG = figM.add_subplot(gs[0])
axMPL2 = figM.add_subplot(gs[1])
axMPL6 = figM.add_subplot(gs[2])

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
thVs = np.array([thC, 2*thC, 4*thC, 6*thC, 8*thC])
thV = 0.3
thCs = np.array([thV/8, thV/6, thV/4, thV/2])

solid_line = mpl.lines.Line2D([], [], ls='-', color='grey')
dashed_line = mpl.lines.Line2D([], [], ls='--', color='grey', alpha=0.5,
                               lw=2.5)

handles = [solid_line, dashed_line]
labels = [r'\tt{afterglowpy}', r'$F_\nu \propto t^{\alpha_{\mathrm{struct}}}$']

handles2 = []
labels2 = []

# for i, thV in enumerate(thVs):
for i, thC in enumerate(thCs):

    Y[0] = thV
    Y[2] = thC

    Y[3] = thV-0.05

    Y2[:4] = Y[:4]
    Y6[:4] = Y[:4]

    alG = pp.calcSlopeStructEff(0, Y, 'G')
    alPL = pp.calcSlopeStructEff(4, Y, 'G')
    alPL2 = pp.calcSlopeStructEff(4, Y2, 'G')
    alPL6 = pp.calcSlopeStructEff(4, Y6, 'G')

    label = (r'$\theta_{{\mathrm{{obs}}}} / \theta_{{\mathrm{{c}}}}'
             r' = {0:.0f}$'.format(thV/thC))

    tb = 0.2 * np.power(9*E0/(16*np.pi*n0*grb.mp*grb.c**5), 1.0/3.0)\
        * np.power(2*np.sin(0.5*(thC+thV)), 8.0/3.0)
    t0 = tb / 5

    i0 = np.searchsorted(t, t0)

    subInd = t < 1.2*tb

    Fnu = grb.fluxDensity(t, nu, 0, 0, *Y)
    axG.plot(t, Fnu, color=colors[i])
    axG.plot(t[subInd], Fnu[i0] * np.power(t/t0, alG)[subInd],
             color=colors[i], ls='--', lw=3.0, alpha=0.5)

    axMG.plot(t, Fnu, color=colors[i])
    axMG.plot(t[subInd], Fnu[i0] * np.power(t/t0, alG)[subInd],
              color=colors[i], ls='--', lw=3.0, alpha=0.5)

    subInd = t < 2.0*tb
    Fnu = grb.fluxDensity(t, nu, 4, 0, *Y)
    axPL.plot(t, Fnu, color=colors[i], label=label)
    axPL.plot(t[subInd], Fnu[i0] * np.power(t/t0, alPL)[subInd],
              color=colors[i], ls='--', lw=3.0, alpha=0.5)

    Fnu = grb.fluxDensity(t, nu, 4, 0, *Y2)
    line, = axMPL2.plot(t, Fnu, color=colors[i], label=label)
    axMPL2.plot(t[subInd], Fnu[i0] * np.power(t/t0, alPL2)[subInd],
                color=colors[i], ls='--', lw=3.0, alpha=0.5)

    Fnu = grb.fluxDensity(t, nu, 4, 0, *Y6)
    axMPL6.plot(t, Fnu, color=colors[i], label=label)
    axMPL6.plot(t[subInd], Fnu[i0] * np.power(t/t0, alPL6)[subInd],
                color=colors[i], ls='--', lw=3.0, alpha=0.5)

    handles2.append(line)
    labels2.append(label)

figs = [figG, figPL]
axs = [axG, axPL, axMG, axMPL2, axMPL6]
for ax in axs:
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$t$ (s)')
    ax.set_xlim(1.0e5, 3.0e7)
    ax.set_ylim(1.0e-9, 3.0e-4)

axG.set_ylabel(r'$F_\nu$ [$10^{14}$ Hz] (mJy)')
axPL.set_ylabel(r'$F_\nu$ [$10^{14}$ Hz] (mJy)')
axMG.set_ylabel(r'$F_\nu$ [$10^{14}$ Hz] (mJy)')

axMPL2.set_yticklabels([])
axMPL6.set_yticklabels([])

axG.legend(handles, labels)
axPL.legend()
axMPL2.legend(handles+handles2, labels+labels2)

axG.text(0.98, 0.95, r'Gaussian Jet', transform=axG.transAxes,
         horizontalalignment='right', verticalalignment='top')
axPL.text(0.98, 0.95, r'Power Law Jet ($b={0:.0f}$)'.format(b),
          transform=axPL.transAxes,
          horizontalalignment='right', verticalalignment='top')

axMG.text(0.98, 0.98, r'Gaussian Jet',
          transform=axMG.transAxes,
          horizontalalignment='right', verticalalignment='top')
axMPL2.text(0.98, 0.98, r'Power Law Jet ($b=2$)',
            transform=axMPL2.transAxes,
            horizontalalignment='right', verticalalignment='top')
axMPL6.text(0.98, 0.98, r'Power Law Jet ($b=6$)',
            transform=axMPL6.transAxes,
            horizontalalignment='right', verticalalignment='top')

figG.tight_layout()
figPL.tight_layout()
figM.tight_layout()

nameG = "lc_closure_Gaussian.pdf"
print("Saving " + nameG)
figG.savefig(nameG)

namePL = "lc_closure_powerlaw4.pdf"
print("Saving " + namePL)
figPL.savefig(namePL)

nameM = "lc_closure_multi.pdf"
print("Saving " + nameM)
figM.savefig(nameM)


# plt.show()
