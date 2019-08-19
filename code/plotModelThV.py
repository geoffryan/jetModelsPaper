import numpy as np
import grbpy as grb
import matplotlib.pyplot as plt
# import paperPlots as pp


def plotPanel(fig, gs0, t, nu, jetType, Y, Z, thVLegend=False):

    gs = gs0.subgridspec(3, 1, hspace=0.0)
    axF = fig.add_subplot(gs[0:-1, 0])
    axa = fig.add_subplot(gs[-1, 0])
    # axs = fig.add_subplot(gs[-1, 0])

    tfac = 0.3

    tb = t * (1.0 + tfac)
    ta = t / (1.0 - tfac)

    b = Y[4]

    if jetType == -1:
        c = 'C0'
        ls = '--'
        name = 'top hat'
    elif jetType == 0:
        c = 'C2'
        ls = '-'
        name = 'Gaussian'
    elif jetType == 4 and b < 4:
        c = 'C1'
        ls = '-.'
        name = 'power law b={0:d}'.format(int(b))
    elif jetType == 4 and b >= 4:
        c = 'C3'
        ls = ':'
        name = 'power law b={0:d}'.format(int(b))
    else:
        c = 'C4'
        ls = '-'
        name = 'unknown'

    thC = Y[2]

    thVs = np.array([0.0, 0.5*thC, 0.9*thC, 2*thC, 4*thC, 6*thC, 8*thC])
    alpha = np.linspace(0.2, 1.0, len(thVs))

    th = np.linspace(0.0, Y[3], 300)
    # phi = np.zeros(th.shape)
    # t_th = np.empty(th.shape)
    nu_th = np.empty(th.shape)
    nu_th[:] = nu[0]

    for i, thV in enumerate(thVs):

        Y[0] = thV

        Fnu = grb.fluxDensity(t, nu, jetType, 0, *Y, **Z)
        Fnua = grb.fluxDensity(ta, nu, jetType, 0, *Y, **Z)
        Fnub = grb.fluxDensity(tb, nu, jetType, 0, *Y, **Z)

        al = np.log(Fnub/Fnua) / np.log(tb/ta)

        """
        dOma = np.empty(Fnu.shape)
        dOmb = np.empty(Fnu.shape)
        dOm = np.empty(Fnu.shape)
        thsa = np.empty(Fnu.shape)
        thsb = np.empty(Fnu.shape)
        ths = np.empty(Fnu.shape)
        for j in range(len(t)):
            t_th[:] = ta[j]
            Inu = grb.intensity(th, phi, t_th, nu_th, jetType, 0, *Y, **Z)
            dOma[j] = Fnua[j] / Inu.max()
            thsa[j] = th[Inu.argmax()]
            t_th[:] = tb[j]
            Inu = grb.intensity(th, phi, t_th, nu_th, jetType, 0, *Y, **Z)
            dOmb[j] = Fnub[j] / Inu.max()
            thsb[j] = th[Inu.argmax()]
            t_th[:] = t[j]
            Inu = grb.intensity(th, phi, t_th, nu_th, jetType, 0, *Y, **Z)
            dOm[j] = Fnu[j] / Inu.max()
            ths[j] = th[Inu.argmax()]
        _, _, us, _ = grb.jet.shockVals(ths, np.zeros(t.shape),
                                        t, jetType, *Y)
        _, _, usa, _ = grb.jet.shockVals(thsa, np.zeros(t.shape),
                                         ta, jetType, *Y)
        _, _, usb, _ = grb.jet.shockVals(thsb, np.zeros(t.shape),
                                         tb, jetType, *Y)

        som = np.log(dOmb/dOma) / np.log(usb/usa)
        """

        axF.plot(t, Fnu,
                 label=r'$\theta_{{\mathrm{{obs}}}} = $ {0:.2f} rad'.format(
                    thV), color=c, ls=ls, alpha=alpha[i])

        axa.plot(t, al, color=c, ls=ls, alpha=alpha[i])
        # axs.plot(t, som, color=c, ls=ls, alpha=alpha[i])

    axa.set_xscale('log')
    axa.set_yscale('linear')
    axF.set_xscale('log')
    axF.set_yscale('log')
    # axs.set_xscale('log')
    # axs.set_yscale('linear')

    axF.get_xaxis().set_visible(False)

    axa.set_yticks([-4, -2, 0, 2, 4])

    axF.set_ylabel(r'$F_{\nu}$ (1 keV)')
    axa.set_xlabel(r'$t$ (s)')
    axa.set_ylabel(r'$\alpha$')

    axF.set_xlim(t[0], t[-1])
    axF.set_ylim(1.0e-10, 1.0e-3)
    axa.set_xlim(t[0], t[-1])
    axa.set_ylim(-4, 5)

    # axs.set_ylim(-3.1, 0.1)

    if thVLegend:
        axF.legend(loc='lower left')

    axF.text(0.95, 0.95, name,
             transform=axF.transAxes, horizontalalignment='right',
             verticalalignment='top')


thV = 0.0
E0 = 1.0e53
thC = 0.08
thW = 5*thC
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

plotPanel(fig, gs[0, 0], t, nu, -1, Y0, Z, thVLegend=True)
plotPanel(fig, gs[0, 1], t, nu, 0, Y0, Z)
plotPanel(fig, gs[1, 0], t, nu, 4, Y0, Z)
Y0[4] = 6
plotPanel(fig, gs[1, 1], t, nu, 4, Y0, Z)

fig.tight_layout()

figname = "lc_model_thV_multi.pdf"
print("Saving " + figname)
fig.savefig(figname)
plt.close(fig)
