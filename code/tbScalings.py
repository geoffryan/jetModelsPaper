import math
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import grbpy as grb


def slopeSpec(p, regime='G'):
    if regime is 'D':
        return 1.0/3.0
    elif regime is 'E':
        return 1.0/3.0
    elif regime is 'F':
        return -0.5
    elif regime is 'G':
        return 0.5*(1-p)
    else:
        return -0.5*p


def slopePre(p, regime='G'):
    if regime is 'D':
        return 0.5
    elif regime is 'E':
        return 1.0/6.0
    elif regime is 'F':
        return -0.25
    elif regime is 'G':
        return 0.75*(1-p)
    else:
        return 0.25*(2-3*p)


def slopePost(p, regime='G'):
    if regime is 'D':
        return -0.25
    elif regime is 'E':
        return -7.0/12.0
    elif regime is 'F':
        return -1.0
    elif regime is 'G':
        return -0.75*p
    else:
        return -0.25*(1+3*p)


def f_pl(t, a, b):
    return a * np.power(t, b)


def f_pl2(t, F0, t0, b0, b1):
    F = np.empty(t.shape)
    A = t <= t0
    B = ~A
    F[A] = F0*np.power(t[A]/t0, b0)
    F[B] = F0*np.power(t[B]/t0, b1)
    return F


def f_pl3(t, F0, t0, t1, b0, b1, b2):
    F = np.empty(t.shape)
    A = t <= t0
    B = (t > t0) & (t <= t1)
    C = t > t1
    F[A] = F0*np.power(t[A]/t0, b0)
    F[B] = F0*np.power(t[B]/t0, b1)
    F[C] = F0*math.pow(t1/t0, b1) * np.power(t[C]/t1, b2)
    return F


def chi2(x, t, F, f):
    y = x.copy()
    if len(x) is 4:
        y[0:2] = np.power(10.0, x[0:2])
    elif len(x) is 6:
        y[0:3] = np.power(10.0, x[0:3])
    Fm = f(t, *y)
    dF = (F-Fm)/(0.1*F)
    return (dF*dF).sum()


def fit_pl(t, F, printOutput=False):

    b0 = np.log(F[-1]/F[0]) / np.log(t[-1]/t[0])
    a0 = (np.log(F[0])*np.log(t[-1])
          - np.log(F[-1])*np.log(t[0])) / np.log(t[-1]/t[0])

    bounds = [(0.0, None), (-4, 4)]

    res = opt.minimize(chi2, [a0, b0], (t, F, f_pl), bounds=bounds,
                       tol=1.0e-16)
    if printOutput or not res.success:
        print(res)

    return res.x, res.fun, 2


def fit_pl2(t, F, printOutput=False):

    N = len(t)
    b0 = np.log(F[-1]/F[0]) / np.log(t[-1]/t[0])
    t0 = t[N//2]
    F0 = F[N//2]

    bounds = [(math.log10(F.min())-1, math.log10(F.max())+1),
              (math.log10(t.min()), math.log10(t.max())),
              (-4, 4), (-4, 4)]

    lF0 = math.log10(F0)
    lt0 = math.log10(t0)

    print("chi2(x0) = {0:.3e}".format(
          chi2([lF0, lt0, b0, b0], t, F, f_pl2)))

    res = opt.minimize(chi2, [lF0, lt0, b0, b0], (t, F, f_pl2),
                       bounds=bounds, method='TNC',
                       options={'maxiter': 8000})
    print("chi2(x1) = {0:.3e}".format(
          chi2(res.x, t, F, f_pl2)))
    if printOutput or not res.success:
        print(res)

    y = res.x.copy()
    y[0:2] = np.power(10.0, res.x[0:2])

    return y, res.fun, 4


def fit_pl3(t, F, printOutput=False):

    N = len(t)
    b0 = np.log(F[-1]/F[0]) / np.log(t[-1]/t[0])
    t0 = t[N//3]
    t1 = t[(2*N)//3]
    F0 = F[N//3]

    lF0 = math.log10(F0)
    lt0 = math.log10(t0)-0.1
    lt1 = math.log10(t1)+0.1

    bounds = [(math.log10(F.min())-1, math.log10(F.max())+1),
              (math.log10(t.min()), math.log10(t.max())),
              (math.log10(t.min()), math.log10(t.max())),
              (-4, 4), (-4, 4), (-4, 4)]

    print("chi2(x0) = {0:.3e}".format(
          chi2([lF0, lt0, lt1, b0, b0, b0], t, F, f_pl3)))
    res = opt.minimize(chi2, [lF0, lt0, lt1, b0, b0, b0], (t, F, f_pl3),
                       bounds=bounds, method='TNC',
                       options={'maxiter': 8000})
    print("chi2(x1) = {0:.3e}".format(
          chi2(res.x, t, F, f_pl3)))
    if printOutput or not res.success:
        print(res)

    y = res.x.copy()
    y[0:3] = np.power(10.0, res.x[0:3])

    return y, res.fun, 6


def findBreaks(regime, NU, jetModel, Y, n=None, plot=False, ax=None, fig=None):

    thV = Y[0]
    E0 = Y[1]
    thC = Y[2]
    n0 = Y[8]
    p = Y[9]

    tN = math.pow(9 * E0 / (16.0*np.pi*n0*grb.mp*grb.c**5), 1.0/3.0)
    chim = 2*np.sin(0.5*np.fabs(thC-thV))
    chip = 2*np.sin(0.5*(thC+thV))

    t = np.geomspace(0.03 * tN * math.pow(chim, 8.0/3.0),
                     min(tN, 3*tN*math.pow(chip, 8.0/3.0)), 300)
    nu = np.empty(t.shape)
    nu[:] = NU

    tRes = 1000
    spread = False

    vfac = 0.25
    Fnu = grb.fluxDensity(t, nu, jetModel, 0, *Y, tRes=tRes, spread=spread)
    Fnuva = grb.fluxDensity(t, nu/(1.0+vfac), jetModel, 0, *Y, tRes=tRes,
                            spread=spread)
    Fnuvb = grb.fluxDensity(t, nu*(1.0+vfac), jetModel, 0, *Y, tRes=tRes,
                            spread=spread)

    be = np.log(Fnuvb/Fnuva) / np.log((1+vfac)*(1+vfac))
    beSpec = slopeSpec(p, 'G')

    be_err = np.fabs((be-beSpec)/beSpec)

    good = be_err < 0.01

    i1 = good.nonzero()[0][0]
    later_inds = (~good)[i1:].nonzero()[0]
    if len(later_inds) > 0:
        i2 = i1 + later_inds[0]
    else:
        i2 = len(t)

    print(i2-i1)

    if plot:
        if ax is None:
            fig, ax = plt.subplots(2, 1, figsize=(6, 4))
        ax.plot(t, Fnu, lw=4)
        ax.plot(t[i1:i2], Fnu[i1:i2])

        ax.set_xscale('log')
        ax.set_yscale('log')

    t = np.geomspace(t[i1], t[i2-1], 100)
    nu = np.empty(t.shape)
    nu[:] = NU
    Fnu = grb.fluxDensity(t, nu, jetModel, 0, *Y, tRes=tRes, spread=spread)

    if n is 1:
        y, chi2, n = fit_pl(t, Fnu, plot)
        if plot:
            ax.plot(t, f_pl(t, *y))
        return 1, y[1]
    elif n is 2:
        y, chi2, n = fit_pl2(t, Fnu, plot)
        if plot:
            ax.plot(t, f_pl2(t, *y))
            ax.axvline(y[1], color='grey', ls='--', alpha=0.5)
        return 2, y[1], y[2], y[3]
    elif n is 3:
        y, chi2, n = fit_pl3(t, Fnu, plot)
        if plot:
            ax.plot(t, f_pl3(t, *y))
            ax.axvline(y[1], color='grey', ls='--', alpha=0.5)
            ax.axvline(y[2], color='grey', ls='--', alpha=0.5)
        return 3, y[1], y[2], y[3], y[4], y[5]
    else:
        y1, chi21, n1 = fit_pl(t, Fnu, plot)
        y2, chi22, n2 = fit_pl2(t, Fnu, plot)
        y3, chi23, n3 = fit_pl3(t, Fnu, plot)

        rc1 = chi21/(n1-1)
        rc2 = chi22/(n2-1)
        rc3 = chi23/(n3-1)

        if plot:
            ax.plot(t, f_pl(t, *y1), color='C2')
            ax.plot(t, f_pl2(t, *y2), color='C3')
            ax.plot(t, f_pl3(t, *y3), color='C4')
            ax.axvline(y2[1], color='C3', ls='--', alpha=0.5)
            ax.axvline(y3[1], color='C4', ls='--', alpha=0.5)
            ax.axvline(y3[2], color='C4', ls='--', alpha=0.5)
            print(rc1, rc2, rc3)
            print(y1[-1])
            print(y2[-2], y2[-1])
            print(y3[-3], y3[-2], y3[-1])

        if rc3 < rc2 and rc3 < rc1:
            return 3, y3[1], y3[2], y3[3], y3[4], y3[5]

        elif rc2 < rc3 and rc2 < rc1:
            return 2, y2[1], y2[2], y2[3]

        else:
            return 1, y1[1]


def makeSlopesPlot():

    thV = 0.75 * 0.02
    E0 = 1.0e52
    thC = 0.02
    thW = 0.4
    b = 5.0
    n0 = 1.0e-3
    p = 2.2
    epse = 0.1
    epsB = 1.0e-4
    xiN = 1.0

    dL = 3.0e26

    Y = np.array([thV, E0, thC, thW, b, 0.0, 0.0, 0.0, n0, p, epse, epsB,
                  xiN, dL])

    tN = math.pow(9 * E0 / (16.0*np.pi*n0*grb.mp*grb.c**5), 1.0/3.0)

    NU = 1.0e17
    regime = 'G'
    jetModel = -1
    tRes = 4000
    spread = False

    tfac = 0.25
    vfac = 0.25
    t = np.geomspace(1.0, 0.03*tN, 1000)
    nu = np.empty(t.shape)
    nu[:] = NU

    findBreaks(regime, NU, jetModel, Y, plot=True)

    Fnu = grb.fluxDensity(t, nu, -1, 0, *Y, tRes=tRes, spread=spread)
    Fnuta = grb.fluxDensity(t/(1.0+tfac), nu, -1, 0, *Y, tRes=tRes,
                            spread=spread)
    Fnutb = grb.fluxDensity(t*(1.0+tfac), nu, -1, 0, *Y, tRes=tRes,
                            spread=spread)
    Fnuva = grb.fluxDensity(t, nu/(1.0+vfac), -1, 0, *Y, tRes=tRes,
                            spread=spread)
    Fnuvb = grb.fluxDensity(t, nu*(1.0+vfac), -1, 0, *Y, tRes=tRes,
                            spread=spread)
    al = np.log(Fnutb/Fnuta) / np.log((1+tfac)*(1+tfac))
    daldt = np.log(Fnutb*Fnuta/(Fnu*Fnu)) / np.log(1.0+tfac)**2
    d2aldt2 = np.zeros(al.shape)
    d2aldt2[1:-1] = (daldt[2:]-daldt[:-2]) / np.log((1+tfac)*(1+tfac))
    be = np.log(Fnuvb/Fnuva) / np.log((1+vfac)*(1+vfac))

    alPre = slopePre(p, 'G')
    alPost = slopePost(p, 'G')
    beSpec = slopeSpec(p, 'G')

    fig, ax = plt.subplots(5, 1, figsize=(4, 7))
    ax[0].plot(t, Fnu)
    ax[1].plot(t, al)
    ax[1].axhline(alPre, color='grey')
    ax[1].axhline(alPost, color='grey')
    ax[2].plot(t, daldt)
    ax[3].plot(t, d2aldt2)
    ax[4].plot(t, be)
    ax[4].axhline(beSpec, color='grey')

    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[1].set_xscale('log')
    ax[2].set_xscale('log')
    ax[3].set_xscale('log')
    ax[4].set_xscale('log')

    ax[2].set_ylim(-1, 1)
    ax[3].set_ylim(-1, 1)


def makeOnAPlot():

    thV = 0.0
    E0 = 1.0e52
    thC = 0.03
    thW = 0.4
    b = 5.0
    n0 = 1.0e-3
    p = 2.2
    epse = 0.1
    epsB = 1.0e-4
    xiN = 1.0

    dL = 3.0e26

    Y = np.array([thV, E0, thC, thW, b, 0.0, 0.0, 0.0, n0, p, epse, epsB,
                  xiN, dL])
    tN = math.pow(9 * E0 / (16.0*np.pi*n0*grb.mp*grb.c**5), 1.0/3.0)

    NU = 1.0e17
    regime = 'G'

    tbTH = []
    tbG = []
    tbPL = []

    thCs = np.array([0.02, 0.04, 0.08, 0.16, 0.32])
    # thCs = np.array([0.08])
    plot = True

    NC = len(thCs)
    if plot:
        figTH, axTH = plt.subplots(1, len(thCs), figsize=(len(thCs)*3, 3))
        figG, axG = plt.subplots(1, len(thCs), figsize=(len(thCs)*3, 3))
        figPL, axPL = plt.subplots(1, len(thCs), figsize=(len(thCs)*3, 3))
    else:
        figTH = None
        figG = None
        figPL = None
        axTH = [None for _ in range(NC)]
        axG = [None for _ in range(NC)]
        axPL = [None for _ in range(NC)]

    for i, thC in enumerate(thCs):
        Y[2] = thC
        Y[3] = 5*thC
        print("theta_c: " + str(thC))
        print("  top hat")
        res = findBreaks(regime, NU, -1, Y, 2, plot=plot, ax=axTH[i])
        tbTH.append(res[1])
        print("  Gaussian")
        res = findBreaks(regime, NU, 0, Y, 2, plot=plot, ax=axG[i])
        tbG.append(res[1])
        print("  Power law")
        res = findBreaks(regime, NU, 4, Y, 2, plot=plot, ax=axPL[i])
        tbPL.append(res[1])

    if plot:
        figTH.tight_layout()
        name = "breaks_alignedOA_lc_TH.png"
        print("Saving " + name)
        figTH.savefig(name)
        plt.close(figTH)

        figG.tight_layout()
        name = "breaks_alignedOA_lc_G.png"
        print("Saving " + name)
        figG.savefig(name)
        plt.close(figG)

        figPL.tight_layout()
        name = "breaks_alignedOA_lc_PL.png"
        print("Saving " + name)
        figPL.savefig(name)
        plt.close(figPL)

    tbTH = np.array(tbTH)
    tbG = np.array(tbG)
    tbPL = np.array(tbPL)

    print(tbTH)
    print(tbG)
    print(tbPL)

    fig, ax = plt.subplots(1, 1)
    ax.plot(thCs, tbTH/tN, marker='o', ls='')
    ax.plot(thCs, tbG/tN, marker='s', ls='')
    ax.plot(thCs, tbPL/tN, marker='^', ls='')
    ax.plot(thCs, np.power(2*np.sin(0.5*thCs), 8./3.), lw=4, ls='--',
            color='grey')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\theta_C$')
    ax.set_ylabel(r'$t_b / t_N$')
    fig.tight_layout()

    name = "breaks_alignedOA_all.pdf"
    print("Saving " + name)
    fig.savefig(name)
    plt.close(fig)


def makeOffAPlot():

    thV = 0.0
    E0 = 1.0e52
    thC = 0.03
    thW = 0.4
    b = 5.0
    n0 = 1.0e-3
    p = 2.2
    epse = 0.1
    epsB = 1.0e-4
    xiN = 1.0

    dL = 3.0e26

    Y = np.array([thV, E0, thC, thW, b, 0.0, 0.0, 0.0, n0, p, epse, epsB,
                  xiN, dL])
    tN = math.pow(9 * E0 / (16.0*np.pi*n0*grb.mp*grb.c**5), 1.0/3.0)

    NU = 1.0e17
    regime = 'G'

    tbTH = []
    tbG = []
    tbPL = []

    thCs = np.array([0.02, 0.04, 0.08, 0.16, 0.32])
    thVoCs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    # thCs = np.array([0.02])
    # thVoCs = np.array([0.25, 0.5, 0.75])

    plot = True

    NC = len(thCs)
    NV = len(thVoCs)
    if plot:
        figTH, axTH = plt.subplots(NV, NC, figsize=(NC*2, 2*NV))
        figG, axG = plt.subplots(NV, NC, figsize=(NC*2, 2*NV))
        figPL, axPL = plt.subplots(NV, NC, figsize=(NC*2, 2*NV))
    else:
        figTH = None
        figG = None
        figPL = None
        axTH = np.empty((NV, NC))
        axG = np.empty((NV, NC))
        axPL = np.empty((NV, NC))

    for i, thC in enumerate(thCs):
        tTH = []
        tG = []
        tPL = []
        print("theta_c: " + str(thC))
        for j, thVoC in enumerate(thVoCs):
            Y[0] = thVoC * thC
            Y[2] = thC
            Y[3] = 5*thC
            print("theta_v / theta_c: " + str(thVoC))
            print("  top hat")
            res = findBreaks(regime, NU, -1, Y, 3, plot=plot, ax=axTH[j, i])
            tTH.append([res[1], res[2]])
            print("  Gaussian")
            res = findBreaks(regime, NU, 0, Y, 3, plot=plot, ax=axG[j, i])
            tG.append([res[1], res[2]])
            print("  Power law")
            res = findBreaks(regime, NU, 4, Y, 3, plot=plot, ax=axPL[j, i])
            tPL.append([res[1], res[2]])
        tbTH.append(tTH)
        tbG.append(tG)
        tbPL.append(tPL)

    if plot:
        figTH.tight_layout()
        name = "breaks_aligned_lc_TH.png"
        print("Saving " + name)
        figTH.savefig(name)
        plt.close(figTH)

        figG.tight_layout()
        name = "breaks_aligned_lc_G.png"
        print("Saving " + name)
        figG.savefig(name)
        plt.close(figG)

        figPL.tight_layout()
        name = "breaks_aligned_lc_PL.png"
        print("Saving " + name)
        figPL.savefig(name)
        plt.close(figPL)

    tbTH = np.array(tbTH)
    tbG = np.array(tbG)
    tbPL = np.array(tbPL)

    print(tbTH)
    print(tbG)
    print(tbPL)

    j1 = 0
    j2 = len(thVoCs)//2
    j3 = len(thVoCs)-1

    fig, ax = plt.subplots(1, 1)
    ax.plot(thCs, tbTH[:, j1, 0]/tN, marker='o', ls='', color='C0')
    ax.plot(thCs, tbTH[:, j1, 1]/tN, marker='o', ls='', color='C0')
    ax.plot(thCs, tbTH[:, j2, 0]/tN, marker='v', ls='', color='C0')
    ax.plot(thCs, tbTH[:, j2, 1]/tN, marker='v', ls='', color='C0')
    ax.plot(thCs, tbTH[:, j3, 0]/tN, marker='^', ls='', color='C0')
    ax.plot(thCs, tbTH[:, j3, 1]/tN, marker='^', ls='', color='C0')
    ax.plot(thCs, tbG[:, j1, 0]/tN, marker='o', ls='', color='C1')
    ax.plot(thCs, tbG[:, j1, 1]/tN, marker='o', ls='', color='C1')
    ax.plot(thCs, tbG[:, j2, 0]/tN, marker='v', ls='', color='C1')
    ax.plot(thCs, tbG[:, j2, 1]/tN, marker='v', ls='', color='C1')
    ax.plot(thCs, tbG[:, j3, 0]/tN, marker='^', ls='', color='C1')
    ax.plot(thCs, tbG[:, j3, 1]/tN, marker='^', ls='', color='C1')
    ax.plot(thCs, tbPL[:, j1, 0]/tN, marker='o', ls='', color='C2')
    ax.plot(thCs, tbPL[:, j1, 1]/tN, marker='o', ls='', color='C2')
    ax.plot(thCs, tbPL[:, j2, 0]/tN, marker='v', ls='', color='C2')
    ax.plot(thCs, tbPL[:, j2, 1]/tN, marker='v', ls='', color='C2')
    ax.plot(thCs, tbPL[:, j3, 0]/tN, marker='^', ls='', color='C2')
    ax.plot(thCs, tbPL[:, j3, 1]/tN, marker='^', ls='', color='C2')
    ax.plot(thCs, np.power(2*np.sin(0.5*thCs), 8./3.), lw=4, ls='--',
            color='grey')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\theta_C$')
    ax.set_ylabel(r'$t_b / t_N$')
    fig.tight_layout()

    name = "breaks_aligned_all.pdf"
    print("Saving " + name)
    fig.savefig(name)
    plt.close(fig)

    for i in range(len(thCs)):
        fig, ax = plt.subplots(1, 1)
        ax.plot(thVoCs, tbTH[i, :, 0]/tN, ls='--', color='C0')
        ax.plot(thVoCs, tbTH[i, :, 1]/tN, ls='-', color='C0')
        ax.plot(thVoCs, tbG[i, :, 0]/tN, ls='--', color='C1')
        ax.plot(thVoCs, tbG[i, :, 1]/tN, ls='-', color='C1')
        ax.plot(thVoCs, tbPL[i, :, 0]/tN, ls='--', color='C2')
        ax.plot(thVoCs, tbPL[i, :, 1]/tN, ls='-', color='C2')
        ax.plot(thVoCs, np.power(2*np.sin(0.5*thCs[i]*(1-thVoCs)), 8./3.),
                ls='--', color='grey')
        ax.plot(thVoCs, np.power(2*np.sin(0.5*thCs[i]*(1+thVoCs)), 8./3.),
                ls='-', color='grey')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\theta_V / \theta_C$')
        ax.set_ylabel(r'$t_b / t_N$')
        fig.tight_layout()

        name = "breaks_aligned_thC_{0:.02f}.png".format(thCs[i])
        print("Saving " + name)
        fig.savefig(name)
        plt.close(fig)


def makeVeryOffAPlot():

    thV = 0.0
    E0 = 1.0e52
    thC = 0.03
    thW = 0.4
    b = 5.0
    n0 = 1.0e-3
    p = 2.2
    epse = 0.1
    epsB = 1.0e-4
    xiN = 1.0

    dL = 3.0e26

    Y = np.array([thV, E0, thC, thW, b, 0.0, 0.0, 0.0, n0, p, epse, epsB,
                  xiN, dL])
    tN = math.pow(9 * E0 / (16.0*np.pi*n0*grb.mp*grb.c**5), 1.0/3.0)

    NU = 1.0e17
    regime = 'G'

    tbTH = []
    tbG = []
    tbPL = []

    thCs = np.array([0.02, 0.04, 0.08, 0.16])
    thVoCs = np.array([1.5, 2.0, 4.0, 6.0])
    # thCs = np.array([0.02])
    # thVoCs = np.array([0.25, 0.5, 0.75])

    plot = True

    NC = len(thCs)
    NV = len(thVoCs)
    if plot:
        figTH, axTH = plt.subplots(NV, NC, figsize=(NC*2, 2*NV))
        figG, axG = plt.subplots(NV, NC, figsize=(NC*2, 2*NV))
        figPL, axPL = plt.subplots(NV, NC, figsize=(NC*2, 2*NV))
    else:
        figTH = None
        figG = None
        figPL = None
        axTH = np.empty((NV, NC))
        axG = np.empty((NV, NC))
        axPL = np.empty((NV, NC))

    for i, thC in enumerate(thCs):
        tTH = []
        tG = []
        tPL = []
        print("theta_c: " + str(thC))
        for j, thVoC in enumerate(thVoCs):
            Y[0] = thVoC * thC
            Y[2] = thC
            Y[3] = 5*thC
            print("theta_v / theta_c: " + str(thVoC))
            print("  top hat")
            res = findBreaks(regime, NU, -1, Y, 3, plot=plot, ax=axTH[j, i])
            tTH.append([res[1], res[2]])
            print("  Gaussian")
            res = findBreaks(regime, NU, 0, Y, 3, plot=plot, ax=axG[j, i])
            tG.append([res[1], res[2]])
            print("  Power law")
            res = findBreaks(regime, NU, 4, Y, 3, plot=plot, ax=axPL[j, i])
            tPL.append([res[1], res[2]])
        tbTH.append(tTH)
        tbG.append(tG)
        tbPL.append(tPL)

    if plot:
        figTH.tight_layout()
        name = "breaks_misaligned_lc_TH.png"
        print("Saving " + name)
        figTH.savefig(name)
        plt.close(figTH)

        figG.tight_layout()
        name = "breaks_misaligned_lc_G.png"
        print("Saving " + name)
        figG.savefig(name)
        plt.close(figG)

        figPL.tight_layout()
        name = "breaks_misaligned_lc_PL.png"
        print("Saving " + name)
        figPL.savefig(name)
        plt.close(figPL)

    tbTH = np.array(tbTH)
    tbG = np.array(tbG)
    tbPL = np.array(tbPL)

    print(tbTH)
    print(tbG)
    print(tbPL)

    j1 = 0
    j2 = len(thVoCs)//2
    j3 = len(thVoCs)-1

    fig, ax = plt.subplots(1, 1)
    ax.plot(thCs, tbTH[:, j1, 0]/tN, marker='o', ls='', color='C0')
    ax.plot(thCs, tbTH[:, j1, 1]/tN, marker='o', ls='', color='C0')
    ax.plot(thCs, tbTH[:, j2, 0]/tN, marker='v', ls='', color='C0')
    ax.plot(thCs, tbTH[:, j2, 1]/tN, marker='v', ls='', color='C0')
    ax.plot(thCs, tbTH[:, j3, 0]/tN, marker='^', ls='', color='C0')
    ax.plot(thCs, tbTH[:, j3, 1]/tN, marker='^', ls='', color='C0')
    ax.plot(thCs, tbG[:, j1, 0]/tN, marker='o', ls='', color='C1')
    ax.plot(thCs, tbG[:, j1, 1]/tN, marker='o', ls='', color='C1')
    ax.plot(thCs, tbG[:, j2, 0]/tN, marker='v', ls='', color='C1')
    ax.plot(thCs, tbG[:, j2, 1]/tN, marker='v', ls='', color='C1')
    ax.plot(thCs, tbG[:, j3, 0]/tN, marker='^', ls='', color='C1')
    ax.plot(thCs, tbG[:, j3, 1]/tN, marker='^', ls='', color='C1')
    ax.plot(thCs, tbPL[:, j1, 0]/tN, marker='o', ls='', color='C2')
    ax.plot(thCs, tbPL[:, j1, 1]/tN, marker='o', ls='', color='C2')
    ax.plot(thCs, tbPL[:, j2, 0]/tN, marker='v', ls='', color='C2')
    ax.plot(thCs, tbPL[:, j2, 1]/tN, marker='v', ls='', color='C2')
    ax.plot(thCs, tbPL[:, j3, 0]/tN, marker='^', ls='', color='C2')
    ax.plot(thCs, tbPL[:, j3, 1]/tN, marker='^', ls='', color='C2')
    # ax.plot(thCs, np.power(2*np.sin(0.5*thCs*thVoCs), 8./3.), lw=4, ls='--',
    #         color='grey')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\theta_C$')
    ax.set_ylabel(r'$t_b / t_N$')
    fig.tight_layout()

    name = "breaks_misaligned_all.pdf"
    print("Saving " + name)
    fig.savefig(name)
    plt.close(fig)

    for i in range(len(thCs)):
        fig, ax = plt.subplots(1, 1)
        ax.plot(thVoCs, tbTH[i, :, 0]/tN, ls='--', color='C0')
        ax.plot(thVoCs, tbTH[i, :, 1]/tN, ls='-', color='C0')
        ax.plot(thVoCs, tbG[i, :, 0]/tN, ls='--', color='C1')
        ax.plot(thVoCs, tbG[i, :, 1]/tN, ls='-', color='C1')
        ax.plot(thVoCs, tbPL[i, :, 0]/tN, ls='--', color='C2')
        ax.plot(thVoCs, tbPL[i, :, 1]/tN, ls='-', color='C2')
        ax.plot(thVoCs, np.power(2*np.sin(0.5*thCs[i]*thVoCs), 8./3.),
                ls='-', color='grey')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\theta_V / \theta_C$')
        ax.set_ylabel(r'$t_b / t_N$')
        fig.tight_layout()

        name = "breaks_misaligned_thC_{0:.02f}.pdf".format(thCs[i])
        print("Saving " + name)
        fig.savefig(name)
        plt.close(fig)


if __name__ == "__main__":

    makeOnAPlot()
    makeOffAPlot()
    makeVeryOffAPlot()
    # makeSlopesPlot()

    plt.show()
