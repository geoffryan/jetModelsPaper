import sys
import math
import numpy as np
import scipy.optimize as opt
import h5py as h5
import matplotlib.pyplot as plt
import afterglowpy as grb
import paperPlots as pp


def f_pl2(t, t0, F0, b0, b1):
    F = np.empty(t.shape)
    A = t <= t0
    B = ~A
    F[A] = F0*np.power(t[A]/t0, b0)
    F[B] = F0*np.power(t[B]/t0, b1)
    return F


def lf_pl2(t, lt0, lF0, b0, b1):
    t0 = math.pow(10.0, lt0)
    F0 = math.pow(10.0, lF0)
    return f_pl2(t, t0, F0, b0, b1)


def f_spl2(t, t0, F0, b0, b1, s):
    F = F0 * np.power(t/t0, b0) * np.power(0.5 * (1 + np.power(t/t0, s)),
                                           (b1-b0)/s)
    return F


def lf_spl2(t, lt0, lF0, b0, b1, ls):
    t0 = math.pow(10.0, lt0)
    F0 = math.pow(10.0, lF0)
    s = math.pow(10.0, ls)

    return f_spl2(t, t0, F0, b0, b1, s)


def chi2(x, t, F, f):
    y = x.copy()
    Fm = f(t, *y)
    dF = (F-Fm)/(0.1*F)
    return (dF*dF).sum()


def betaRegime(regime, Y):

    p = Y[9]
    if regime == 'D':
        return 1.0/3.0
    elif regime == 'E':
        return 1.0/3.0
    elif regime == 'F':
        return -0.5
    elif regime == 'G':
        return 0.5*(1.0-p)
    else:
        return -0.5*p


def findJetBreak(jetType, Y, Z, regime, NU, printMode=None, returnAll=False,
                 ax=None):

    tN0 = pp.calcTN0(Y)
    tb = 10*tN0

    thV = Y[0]
    thC = Y[2]
    thW = Y[3]

    chip = 2*math.sin(0.5*(thC+thV))

    tj_guess = 0.24 * tN0 * math.pow(chip, 8.0/3.0)

    if jetType == -1:
        if thV <= thC:
            ta = 1.0e-3 * tj_guess
        else:
            tC = 2*pp.calcTC(jetType, Y, regime)
            ta = 2*tC
    else:
        if thV <= thC and thV <= thW:
            ta = 1.0e-3 * tj_guess
        elif thV <= thW:
            ta = 1.0e-3 * tj_guess
        else:
            tW = pp.calcTW(jetType, Y, regime)
            ta = 2.0*tW

    if ta == 0.0:
        print(ta, tb, thV, thC, thW)

    t = np.geomspace(ta, tb, 200)

    theta = np.zeros(t.shape)
    phi = np.zeros(t.shape)

    """
    _, _, u, _ = grb.jet.shockVals(theta, phi, t, jetType, *Y, **Z)
    while u.max() < 1.0:
        ta = ta / 10.0
        print("Retrying with ta = {0:.3e} s".format(ta))
        t = np.geomspace(ta, tb, 200)
        _, _, u, _ = grb.jet.shockVals(theta, phi, t, jetType, *Y, **Z)

    rel = (u >= 0.5) & (u < 1.0e5)
    t = t[rel]
    u = u[rel]
    theta = theta[rel]
    phi = phi[rel]
    """

    ta = 5.0e-2 * tj_guess
    tb = 2.0e1 * tj_guess
    t = np.geomspace(ta, tb, 200)
    theta = np.zeros(t.shape)
    phi = np.zeros(t.shape)
    _, _, u, _ = grb.jet.shockVals(theta, phi, t, jetType, *Y, **Z)

    nu = np.empty(t.shape)
    nu[:] = NU
    nufac = 1.01
    Fnu = grb.fluxDensity(t, nu, jetType, 0, *Y, **Z)
    Fnua = grb.fluxDensity(t, nu/nufac, jetType, 0, *Y, **Z)
    Fnub = grb.fluxDensity(t, nu*nufac, jetType, 0, *Y, **Z)
    beta = np.log(Fnub/Fnua) / np.log(nufac*nufac)

    right = (np.fabs(beta - betaRegime(regime, Y)) < 0.02)\
        & (t > 3*ta) & (t < tb/3)
    t_fit = t[right]
    # nu_fit = nu[right]
    # u_fit = u[right]
    # theta_fit = theta[right]
    # phi_fit = phi[right]
    Fnu_fit = Fnu[right]

    t_guess = 0.24*tN0*np.power(2*np.sin(0.5*(thC+thV)), 8.0/3.0)
    i0s = [1, len(t_fit)//4, len(t_fit)//2, 3*len(t_fit)//4, len(t_fit)-2]
    i0 = np.searchsorted(t_fit, t_guess)

    if i0 > 1 and i0 < len(t_fit)-2:
        i0s.append(i0)

    bmin = -10
    bmax = 10

    bounds = [(math.log10(t_fit.min())-1, math.log10(t_fit.max())+1),
              (math.log10(Fnu_fit.min())-1, math.log10(Fnu_fit.max())+1),
              (bmin, bmax), (bmin, bmax), (-1.5, 1.5)]

    chi2min = np.inf
    x_best = None

    for i0 in i0s:
        if printMode is 'all':
            print("i0 = {0:d}".format(i0))
        lt0 = math.log10(t_fit[i0])
        lF0 = math.log10(Fnu_fit[i0])
        ls = 1.0

        b0 = math.log(Fnu_fit[i0]/Fnu_fit[0]) / math.log(t_fit[i0]/t_fit[0])
        b1 = math.log(Fnu_fit[-1]/Fnu_fit[i0]) / math.log(t_fit[-1]/t_fit[i0])

        if b0 < bmin:
            b0 = bmin+0.1
        if b0 > bmax:
            b0 = bmax-0.1
        if b1 < bmin:
            b1 = bmin+0.1
        if b1 > bmax:
            b1 = bmax-0.1

        x0 = [lt0, lF0, b0, b1, ls]
        func = lf_spl2

        if printMode is 'all':
            print("chi2(x0) = {0:.3e}".format(
                  chi2(x0, t_fit, Fnu_fit, func)))

        res = opt.minimize(chi2, x0, (t_fit, Fnu_fit, func),
                           bounds=bounds, method='TNC',
                           options={'maxiter': 8000})
        if printMode is 'all':
            print("chi2(x1) = {0:.3e}".format(
                  chi2(res.x, t_fit, Fnu_fit, func)))
        if printMode is 'all':
            print(res)
        elif printMode is 'summary':
            print("Success: " + str(res.success) + " chi2={0:.2e}".format(
                  res.fun))

        if res.fun < chi2min:
            chi2min = res.fun
            x_best = res.x

    x = x_best

    if x[3] - x[2] > -0.74:
        print("   is that even a jetbreak? b0={0:.2f} b1={1:.2f} thV={2:.2f}"
              .format(x[2], x[3], thV))

    if ax is not None:
        ax.plot(t, Fnu)
        ax.plot(t_fit, func(t_fit, *x0), lw=1, color='k', ls='--')
        ax.plot(t_fit, func(t_fit, *x), lw=1, color='k')

        # ax2 = ax.twinx()
        # ax2.plot(t, u, color='tab:orange')
        # ax2.set_yscale('log')

        ax.set_xscale('log')
        ax.set_yscale('log')

    if returnAll:
        return x

    return math.pow(10.0, x[0])


def calcJetBreakGrid(thVs, thCs, thWs, jetType, Y, Z, regime, NU,
                     printMode=None, figName=None):

    NC = len(thCs)
    NV = len(thVs)

    tj = np.empty((NC, NV))
    Fj = np.empty((NC, NV))
    b0 = np.empty((NC, NV))
    b1 = np.empty((NC, NV))
    s = np.empty((NC, NV))

    panelWidth = 2.0
    panelHeight = 2.0

    fig, ax = plt.subplots(NC, NV, figsize=(panelWidth*NV, panelHeight*NC))

    for i, thC in enumerate(thCs):
        print("thC = {0:.3f} ({1:d} of {2:d})".format(thC, i+1, NC))
        for j, thV in enumerate(thVs):
            Y[0] = thV
            Y[2] = thC
            Y[3] = thWs[i]
            x = findJetBreak(jetType, Y, Z, regime, NU, printMode,
                             ax=ax[i, j], returnAll=True)
            tj[i, j] = math.pow(10.0, x[0])
            Fj[i, j] = math.pow(10.0, x[1])
            b0[i, j] = x[2]
            b1[i, j] = x[3]
            s[i, j] = math.pow(10.0, x[4])

    if figName is not None:
        print("Saving " + figName)
        fig.savefig(figName)
        plt.close(fig)

    return tj, Fj, b0, b1, s


def plotJetBreaks(thVs, thCs, thWs, Y, Z, regime, NU, printMode=None):

    tN0 = 0.24*pp.calcTN0(Y)
    chiC = 2*np.sin(0.5*thCs)
    thp = np.linspace(thCs.min()+thVs.min(), thCs.max()+thVs.max(), 100)
    chip = 2*np.sin(0.5*thp)

    print("Top Hat")
    figName = 'jetbreak_lcGrid_tophat_{0}.pdf'.format(regime)
    res_th = calcJetBreakGrid(thVs, thCs, thWs, -1, Y, Z, regime, NU,
                              figName=figName)

    print("Gaussian")
    figName = 'jetbreak_lcGrid_Gaussian_{0}.pdf'.format(regime)
    res_g = calcJetBreakGrid(thVs, thCs, thWs, 0, Y, Z, regime, NU,
                             figName=figName)

    # thWs = 0.5*thWs
    print("Powerlaw2")
    Y[4] = 2
    figName = 'jetbreak_lcGrid_powerlaw2_{0}.pdf'.format(regime)
    res_pl2 = calcJetBreakGrid(thVs, thCs, thWs, 4, Y, Z, regime, NU,
                               figName=figName)

    print("Powerlaw6")
    Y[4] = 6
    figName = 'jetbreak_lcGrid_powerlaw6_{0}.pdf'.format(regime)
    res_pl6 = calcJetBreakGrid(thVs, thCs, thWs, 4, Y, Z, regime, NU,
                               figName=figName)
    tj_th = res_th[0]
    tj_g = res_g[0]
    tj_pl2 = res_pl2[0]
    tj_pl6 = res_pl6[0]

    name = "jetbreak_data_{0}.h5".format(regime)
    print("Saving data to: " + name)
    f = h5.File(name, "w")
    f.create_dataset("Y", data=Y)
    f.create_dataset("thV", data=thVs)
    f.create_dataset("thC", data=thCs)
    f.create_dataset("tj_tophat", data=tj_th)
    f.create_dataset("tj_gaussian", data=tj_g)
    f.create_dataset("tj_powerlaw2", data=tj_pl2)
    f.create_dataset("tj_powerlaw6", data=tj_pl6)
    f.create_dataset("Fj_tophat", data=res_th[1])
    f.create_dataset("Fj_gaussian", data=res_g[1])
    f.create_dataset("Fj_powerlaw2", data=res_pl2[1])
    f.create_dataset("Fj_powerlaw6", data=res_pl6[1])
    f.create_dataset("b0_tophat", data=res_th[2])
    f.create_dataset("b0_gaussian", data=res_g[2])
    f.create_dataset("b0_powerlaw2", data=res_pl2[2])
    f.create_dataset("b0_powerlaw6", data=res_pl6[2])
    f.create_dataset("b1_tophat", data=res_th[3])
    f.create_dataset("b1_gaussian", data=res_g[3])
    f.create_dataset("b1_powerlaw2", data=res_pl2[3])
    f.create_dataset("b1_powerlaw6", data=res_pl6[3])
    f.create_dataset("s_tophat", data=res_th[4])
    f.create_dataset("s_gaussian", data=res_g[4])
    f.create_dataset("s_powerlaw2", data=res_pl2[4])
    f.create_dataset("s_powerlaw6", data=res_pl6[4])
    f.close()

    figV, axV = plt.subplots(1, 1)
    for i, thC in enumerate(thCs):
        axV.plot(thVs, tj_th[i, :], color='tab:blue', ls='--', marker='.')
        axV.plot(thVs, tj_g[i, :], color='tab:green', ls='-', marker='.')
        axV.plot(thVs, tj_pl2[i, :], color='tab:orange', ls='-.', marker='.')
        axV.plot(thVs, tj_pl6[i, :], color='tab:red', ls=':', marker='.')
    axV.set_xscale('linear')
    axV.set_yscale('log')
    axV.set_xlabel(r'$\theta_{\mathrm{obs}}$')
    axV.set_ylabel(r'$t_b$ (s)')

    figV.tight_layout()
    name = 'jetbreak_thV_{0}.pdf'.format(regime)
    print("Saving " + name)
    figV.savefig(name)
    plt.close(figV)

    figC, axC = plt.subplots(1, 1)
    for j, thV in enumerate(thVs):
        axC.plot(thCs, tj_th[:, j], color='tab:blue', ls='--', marker='.')
        axC.plot(thCs, tj_g[:, j], color='tab:green', ls='-', marker='.')
        axC.plot(thCs, tj_pl2[:, j], color='tab:orange', ls='-.', marker='.')
        axC.plot(thCs, tj_pl6[:, j], color='tab:red', ls=':', marker='.')
    axC.set_xscale('linear')
    axC.set_yscale('log')
    axC.set_xlabel(r'$\theta_{\mathrm{c}}$')
    axC.set_ylabel(r'$t_b$ (s)')

    figC.tight_layout()
    name = 'jetbreak_thC_{0}.pdf'.format(regime)
    print("Saving " + name)
    figC.savefig(name)
    plt.close(figC)

    thVoCs = (thVs[None, :] / thCs[:, None]).flat

    figVoC, axVoC = plt.subplots(1, 1)
    axVoC.plot(thVoCs, tj_th.flat, color='tab:blue', ls='', marker='.')
    axVoC.plot(thVoCs, tj_g.flat, color='tab:green', ls='', marker='.')
    axVoC.plot(thVoCs, tj_pl2.flat, color='tab:orange', ls='', marker='.')
    axVoC.plot(thVoCs, tj_pl6.flat, color='tab:red', ls='', marker='.')
    axVoC.set_xscale('linear')
    axVoC.set_yscale('log')
    axVoC.set_xlabel(r'$\theta_{\mathrm{obs}} / \theta_{\mathrm{c}}$')
    axVoC.set_ylabel(r'$t_b$ (s)')

    figVoC.tight_layout()
    name = 'jetbreak_thVoC_{0}.pdf'.format(regime)
    print("Saving " + name)
    figVoC.savefig(name)
    plt.close(figVoC)

    thVpCs = (thVs[None, :] + thCs[:, None]).flat

    figVpC, axVpC = plt.subplots(1, 1)
    axVpC.plot(thp, tN0*np.power(chip, 8.0/3.0), ls='-', color='lightgrey')
    axVpC.plot(thVpCs, tj_th.flat, color='tab:blue', ls='', marker='.')
    axVpC.plot(thVpCs, tj_g.flat, color='tab:green', ls='', marker='.')
    axVpC.plot(thVpCs, tj_pl2.flat, color='tab:orange', ls='', marker='.')
    axVpC.plot(thVpCs, tj_pl6.flat, color='tab:red', ls='', marker='.')
    axVpC.set_xscale('log')
    axVpC.set_yscale('log')
    axVpC.set_xlabel(r'$\theta_{\mathrm{obs}} + \theta_{\mathrm{c}}$')
    axVpC.set_ylabel(r'$t_b$ (s)')

    figVpC.tight_layout()
    name = 'jetbreak_thVpC_{0}.pdf'.format(regime)
    print("Saving " + name)
    figVpC.savefig(name)
    plt.close(figVpC)

    figV0, axV0 = plt.subplots(1, 1)
    axV0.plot(thCs, tN0*np.power(chiC, 8.0/3.0), ls='-', color='lightgrey')
    axV0.plot(thCs, tj_th[:, 0], color='tab:blue', ls='--')
    axV0.plot(thCs, tj_g[:, 0], color='tab:green', ls='-')
    axV0.plot(thCs, tj_pl2[:, 0], color='tab:orange', ls='-.')
    axV0.plot(thCs, tj_pl6[:, 0], color='tab:red', ls=':')
    axV0.text(0.95, 0.95,
              r"$\theta_{{\mathrm{{obs}}}} = {0:.1f}$ rad"
              .format(thVs[0]), transform=axV0.transAxes,
              horizontalalignment='right', verticalalignment='top')
    axV0.set_xscale('log')
    axV0.set_yscale('log')
    axV0.set_xlabel(r'$\theta_{\mathrm{c}}$')
    axV0.set_ylabel(r'$t_b$ (s)')

    figV0.tight_layout()
    name = 'jetbreak_thV0_{0}.pdf'.format(regime)
    print("Saving " + name)
    figV0.savefig(name)
    plt.close(figV0)

    figVG, axVG = plt.subplots(len(thCs), 1, figsize=(3.5, len(thCs)*2.5))
    for i, thC in enumerate(thCs):
        chip = 2*np.sin(0.5*(thC + thVs))
        chiV = 2*np.sin(0.5*thVs)
        axVG[i].plot(thVs, tN0*np.power(chip, 8.0/3.0), ls='-',
                     color='lightgrey')
        axVG[i].plot(thVs, tN0*np.power(thC+thVs, 8.0/3.0), ls=':',
                     color='lightgrey')
        axVG[i].plot(thVs, tN0*np.power(chiV, 8.0/3.0), ls='--',
                     color='lightgrey')
        axVG[i].plot(thVs, tN0*np.power(thVs, 8.0/3.0), ls='-.',
                     color='lightgrey')
        axVG[i].plot(thVs, tj_th[i, :], color='tab:blue', ls='--')
        axVG[i].plot(thVs, tj_g[i, :], color='tab:green', ls='-')
        axVG[i].plot(thVs, tj_pl2[i, :], color='tab:orange', ls='-.')
        axVG[i].plot(thVs, tj_pl6[i, :], color='tab:red', ls=':')
        axVG[i].text(0.95, 0.95,
                     r"$\theta_{{\mathrm{{c}}}} = {0:.2f}$ rad"
                     .format(thCs[i]), transform=axVG[i].transAxes,
                     horizontalalignment='right', verticalalignment='top')
        axVG[i].set_xscale('linear')
        axVG[i].set_yscale('log')
        axVG[i].set_xlabel(r'$\theta_{\mathrm{obs}}$')
        axVG[i].set_ylabel(r'$t_b$ (s)')

    figVG.tight_layout()
    name = 'jetbreak_thVG_{0}.pdf'.format(regime)
    print("Saving " + name)
    figVG.savefig(name)
    plt.close(figVG)

    figCG, axCG = plt.subplots(1, len(thVs), figsize=(len(thVs)*3.5, 2.5))
    for j, thV in enumerate(thVs):
        chip = 2*np.sin(0.5*(thCs + thV))
        axCG[j].plot(thCs, tN0*np.power(chip, 8.0/3.0), ls='-',
                     color='lightgrey')
        axCG[j].plot(thCs, tN0*np.power(thCs+thV, 8.0/3.0), ls=':',
                     color='lightgrey')
        axCG[j].plot(thCs, tj_th[:, j], color='tab:blue', ls='--')
        axCG[j].plot(thCs, tj_g[:, j], color='tab:green', ls='-')
        axCG[j].plot(thCs, tj_pl2[:, j], color='tab:orange', ls='-.')
        axCG[j].plot(thCs, tj_pl6[:, j], color='tab:red', ls=':')
        axCG[j].text(0.95, 0.95,
                     r"$\theta_{{\mathrm{{obs}}}} = {0:.2f}$ rad"
                     .format(thVs[j]), transform=axCG[j].transAxes,
                     horizontalalignment='right', verticalalignment='top')
        axCG[j].set_xscale('log')
        axCG[j].set_yscale('log')
        axCG[j].set_xlabel(r'$\theta_{\mathrm{c}}$')
        axCG[j].set_ylabel(r'$t_b$ (s)')

    figCG.tight_layout()
    name = 'jetbreak_thCG_{0}.pdf'.format(regime)
    print("Saving " + name)
    figCG.savefig(name)
    plt.close(figCG)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        regime = 'G'
    else:
        regime = sys.argv[1]

    NU, _, Y, _ = pp.getRegimePars(regime)

    Y[0] = 0.0
    Y[2] = 0.1
    Y[3] = 0.6
    Z = {}

    """
    fig, ax = plt.subplots(1, 1)
    findJetBreak(-1, Y, Z, regime, NU, printMode='all', ax=ax)

    Y[0] = 0.2
    fig, ax = plt.subplots(1, 1)
    findJetBreak(-1, Y, Z, regime, NU, printMode='all', ax=ax)
    """

    thVs = np.linspace(0.0, 1.0, 11)
    # thVs = [0.0]
    # thVs.extend(np.geomspace(1.0e-2, 1.0, 12))
    # thVs = np.array(thVs)
    thCs = np.linspace(0.04, 0.4, 11)
    thWs = 5*thCs
    thWs[thWs > 0.5*np.pi] = 0.5*np.pi

    """
    print("Top Hat")
    calcJetBreakGrid(thVs, thCs, thWs, -1, Y, Z, regime, NU)

    print("Gaussian")
    calcJetBreakGrid(thVs, thCs, thWs, 0, Y, Z, regime, NU)

    thWs = 5*thCs

    print("Powerlaw 2")
    Y[4] = 2
    calcJetBreakGrid(thVs, thCs, thWs, 4, Y, Z, regime, NU)

    print("Powerlaw 6")
    Y[4] = 6
    calcJetBreakGrid(thVs, thCs, thWs, 4, Y, Z, regime, NU)

    plt.show()
    """

    plotJetBreaks(thVs, thCs, thWs, Y, Z, regime, NU)
