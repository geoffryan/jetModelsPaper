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


def Etheta(jetType, th, Y):
    E0 = Y[1]
    thC = Y[2]
    thW = Y[3]
    b = Y[4]
    if jetType == 0:
        if th > thW:
            return 0
        return E0*math.exp(-0.5*th*th/(thC*thC))
    elif jetType == 4:
        if th > thW:
            return 0
        TH = math.sqrt(1.0 + th*th/(thC*thC))
        return E0*math.pow(TH, -b)
    elif jetType == -2:
        if th > thW or th < thC:
            return 0
        return E0
    else:
        if th > thC:
            return 0
        return E0


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


def al_pl(t, a, b):
    al = np.empty(t.shape)
    al[:] = b
    return al


def al_pl2(t, F0, t0, b0, b1):
    al = np.empty(t.shape)
    al[t < t0] = b0
    al[t >= t0] = b1
    return al


def al_pl3(t, F0, t0, t1, b0, b1, b2):
    al = np.empty(t.shape)
    al[t < t0] = b0
    al[t >= t0] = b1
    al[t >= t1] = b2
    return al


def chi2(x, t, F, f):
    y = x.copy()
    if len(x) is 4:
        y[0:2] = np.power(10.0, x[0:2])
    elif len(x) is 6:
        y[0:3] = np.power(10.0, x[0:3])
    Fm = f(t, *y)
    dF = (F-Fm)/(0.1*F)
    return (dF*dF).sum()


def fit_pl(t, F, printMode=None):

    b0 = np.log(F[-1]/F[0]) / np.log(t[-1]/t[0])
    a0 = (np.log(F[0])*np.log(t[-1])
          - np.log(F[-1])*np.log(t[0])) / np.log(t[-1]/t[0])

    bounds = [(0.0, None), (-4, 4)]

    res = opt.minimize(chi2, [a0, b0], (t, F, f_pl), bounds=bounds,
                       tol=1.0e-16)
    if printMode is 'all':
        print(res)
    elif printMode is 'summary':
        print("Success: " + str(res.success) + " chi2={0:.2e}".format(
              res.fun))

    return res.x, res.fun, 2


def fit_pl2(t, F, tN, thV, thC, printMode=None):

    tbp = tN * np.power(2*np.sin(0.5*np.fabs(thV+thC)), 8.0/3.0)

    try:
        ip = np.argwhere(t > tbp)[0]
    except IndexError:
        ip = len(t)//2
    b0 = np.log(F[ip]/F[0]) / np.log(t[ip]/t[0])
    b1 = np.log(F[-1]/F[ip]) / np.log(t[-1]/t[ip])
    t0 = t[ip]
    F0 = F[ip]

    bounds = [(math.log10(F.min())-1, math.log10(F.max())+1),
              (math.log10(t.min()), math.log10(t.max())),
              (-4, 4), (-4, 4)]

    lF0 = math.log10(F0)
    lt0 = math.log10(t0)

    if printMode is 'all':
        print("chi2(x0) = {0:.3e}".format(
              chi2([lF0, lt0, b0, b1], t, F, f_pl2)))

    res = opt.minimize(chi2, [lF0, lt0, b0, b0], (t, F, f_pl2),
                       bounds=bounds, method='TNC',
                       options={'maxiter': 8000})
    if printMode is 'all':
        print("chi2(x1) = {0:.3e}".format(
              chi2(res.x, t, F, f_pl2)))
    if printMode is 'all':
        print(res)
    elif printMode is 'summary':
        print("Success: " + str(res.success) + " chi2={0:.2e}".format(
              res.fun))

    y = res.x.copy()
    y[0:2] = np.power(10.0, res.x[0:2])

    return y, res.fun, 4


def fit_pl3(t, F, tN, thV, thC, printMode=None):

    tbm = tN * np.power(2*np.sin(0.5*np.fabs(thV-thC)), 8.0/3.0)
    tbp = tN * np.power(2*np.sin(0.5*np.fabs(thV+thC)), 8.0/3.0)

    """
    try:
        im = np.argwhere(t > tbm)[0]
        ip = np.argwhere(t > tbp)[0]
    except IndexError:
        im = len(t)//3
        ip = 2*im
    """

    im = len(t)//3
    ip = 2*im

    if printMode is 'all':
        print(tbm, tbp, im, ip)
    if im == ip:
        if ip < len(t)-1:
            ip += 1
        else:
            im -= 1

    lt0 = math.log10(t[im])
    lt1 = math.log10(t[ip])
    lF0 = math.log10(F[im])

    b0 = np.log(F[im]/F[0]) / np.log(t[im]/t[0])
    b1 = np.log(F[ip]/F[im]) / np.log(t[ip]/t[im])
    b2 = np.log(F[-1]/F[ip]) / np.log(t[-1]/t[ip])

    bounds = [(math.log10(F.min())-1, math.log10(F.max())+1),
              (math.log10(t.min()), math.log10(t.max())),
              (math.log10(t.min()), math.log10(t.max())),
              (-4, 4), (-4, 4), (-4, 4)]

    if printMode is 'all':
        print("chi2(x0) = {0:.3e}".format(
              chi2([lF0, lt0, lt1, b0, b0, b0], t, F, f_pl3)))
    res = opt.minimize(chi2, [lF0, lt0, lt1, b0, b1, b2], (t, F, f_pl3),
                       bounds=bounds, method='TNC',
                       options={'maxiter': 8000})
    if printMode is 'all':
        print("chi2(x1) = {0:.3e}".format(chi2(res.x, t, F, f_pl3)))
    if printMode is 'all':
        print(res)
    elif printMode is 'summary':
        print("Success: " + str(res.success) + " chi2={0:.2e}".format(
              res.fun))

    y = res.x.copy()
    y[0:3] = np.power(10.0, res.x[0:3])

    return y, res.fun, 6


def findBreaks(regime, NU, jetModel, Y, n=None, spread=False,
               plot=False, ax=None, fig=None, printMode=None):

    thV = Y[0]
    E0 = Y[1]
    thC = Y[2]
    thW = Y[3]
    n0 = Y[8]
    p = Y[9]

    tN = math.pow(9 * E0 / (16.0*np.pi*n0*grb.mp*grb.c**5), 1.0/3.0)
    chim = 2*np.sin(0.5*np.fabs(thC-thV))
    chip = 2*np.sin(0.5*(thC+thV))

    EW = Etheta(jetModel, thW, Y)
    tNW = math.pow(9 * EW / (16.0*np.pi*n0*grb.mp*grb.c**5), 1.0/3.0)
    chiW = 2*np.sin(0.5*np.fabs(thV - thW))

    t0_OnA = tN * math.pow(chim, 8.0/3.0)
    t0_OfA = tNW * math.pow(chiW, 8.0/3.0)

    """
    t0 = t0_OnA
    if thV > thW:
        t0 = min(t0_OnA, t0_OfA)
    """

    t0 = min(t0_OnA, t0_OfA)
    t = np.geomspace(1.0e-1 * t0, min(0.8*tN, 3*tN*math.pow(chip, 8.0/3.0)),
                     300)
    # t = np.geomspace(0.1 * tN * math.pow(chim, 8.0/3.0),
    #                  min(0.8*tN, 3*tN*math.pow(chip, 8.0/3.0)), 300)
    nu = np.empty(t.shape)
    nu[:] = NU

    tRes = 1000

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

    agood = []
    reading = False
    i1 = -1
    i2 = -1
    for i in range(len(good)):
        if good[i] and not reading:
            i1 = i
            reading = True
        if not good[i] and reading:
            i2 = i
            reading = False
            agood.append([i1, i2])
    if reading:
        agood.append([i1, len(good)])
    agood = np.array(agood)
    ngood = agood[:, 1] - agood[:, 0]
    j = np.argmax(ngood)
    i1 = agood[j, 0]
    i2 = agood[j, 1]

    if printMode is 'all':
        print(i2-i1)

    if plot:
        if ax is None:
            fig = plt.figure(figsize=(6, 4))
            gs = fig.add_gridspec(4, 1, hspace=0)
            ax0 = fig.add_subplot(gs[:-1])
            ax0.get_xaxis().set_visible(False)
            ax1 = fig.add_subplot(gs[-1])
            ax = [ax0, ax1]
        ax[0].plot(t, Fnu, lw=4)
        ax[0].plot(t[i1:i2], Fnu[i1:i2])
        al = np.log(Fnu[2:]/Fnu[:-2]) / np.log(t[2:]/t[:-2])
        N = len(t)
        ax[1].plot(t[1:-1], al, lw=4)
        ax[1].plot(t[max(1, i1):min(i2, N-1)], al[max(i1-1, 0):min(i2-1, N-2)])

        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[1].set_xscale('log')

    t = np.geomspace(t[i1], t[i2-1], 100)
    nu = np.empty(t.shape)
    nu[:] = NU
    Fnu = grb.fluxDensity(t, nu, jetModel, 0, *Y, tRes=tRes, spread=spread)

    if n is 1:
        y, chi2, n = fit_pl(t, Fnu, plot, printMode)
        if plot:
            ax[0].plot(t, f_pl(t, *y))
            ax[1].plot(t, al_pl(t, *y))
        return 1, y[1]
    elif n is 2:
        y, chi2, n = fit_pl2(t, Fnu, tN, thV, thC, printMode)
        if plot:
            ax[0].plot(t, f_pl2(t, *y))
            ax[1].plot(t, al_pl2(t, *y))
            ax[0].axvline(y[1], color='grey', ls='--', alpha=0.5)
            ax[1].axvline(y[1], color='grey', ls='--', alpha=0.5)
        return 2, y[1], y[2], y[3]
    elif n is 3:
        y, chi2, n = fit_pl3(t, Fnu, tN, thV, thC, printMode)
        if plot:
            ax[0].plot(t, f_pl3(t, *y))
            ax[1].plot(t, al_pl3(t, *y))
            ax[0].axvline(y[1], color='grey', ls='--', alpha=0.5)
            ax[0].axvline(y[2], color='grey', ls='--', alpha=0.5)
            ax[1].axvline(y[1], color='grey', ls='--', alpha=0.5)
            ax[1].axvline(y[2], color='grey', ls='--', alpha=0.5)
        return 3, y[1], y[2], y[3], y[4], y[5]
    else:
        y1, chi21, n1 = fit_pl(t, Fnu, printMode)
        y2, chi22, n2 = fit_pl2(t, Fnu, tN, thV, thC, printMode)
        y3, chi23, n3 = fit_pl3(t, Fnu, tN, thV, thC, printMode)

        rc1 = chi21/(n1-1)
        rc2 = chi22/(n2-1)
        rc3 = chi23/(n3-1)

        if plot:
            ax[0].plot(t, f_pl(t, *y1), color='C2')
            ax[0].plot(t, f_pl2(t, *y2), color='C3')
            ax[0].plot(t, f_pl3(t, *y3), color='C4')
            ax[1].plot(t, al_pl(t, *y1), color='C2')
            ax[1].plot(t, al_pl2(t, *y2), color='C3')
            ax[1].plot(t, al_pl3(t, *y3), color='C4')
            ax[0].axvline(y2[1], color='C3', ls='--', alpha=0.5)
            ax[0].axvline(y3[1], color='C4', ls='--', alpha=0.5)
            ax[0].axvline(y3[2], color='C4', ls='--', alpha=0.5)
            ax[1].axvline(y2[1], color='C3', ls='--', alpha=0.5)
            ax[1].axvline(y3[1], color='C4', ls='--', alpha=0.5)
            ax[1].axvline(y3[2], color='C4', ls='--', alpha=0.5)
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


def makeVeryOffAPlot(nfit=2):

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

    thCs = np.array([0.02, 0.04, 0.08, 0.12, 0.15])
    thVoCs = np.array([1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
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
            if nfit is 3:
                print("  top hat")
                res = findBreaks(regime, NU, -1, Y, 3, plot=plot,
                                 ax=axTH[j, i])
                tTH.append([res[1], res[2]])
                print("  Gaussian")
                res = findBreaks(regime, NU, 0, Y, 3, plot=plot,
                                 ax=axG[j, i])
                tG.append([res[1], res[2]])
                print("  Power law")
                res = findBreaks(regime, NU, 4, Y, 3, plot=plot,
                                 ax=axPL[j, i])
                tPL.append([res[1], res[2]])
            else:
                print("  top hat")
                res = findBreaks(regime, NU, -1, Y, 2, plot=plot,
                                 ax=axTH[j, i])
                tTH.append([res[1]])
                print("  Gaussian")
                res = findBreaks(regime, NU, 0, Y, 2, plot=plot,
                                 ax=axG[j, i])
                tG.append([res[1]])
                print("  Power law")
                res = findBreaks(regime, NU, 4, Y, 2, plot=plot,
                                 ax=axPL[j, i])
                tPL.append([res[1]])
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
    ax.plot(thCs, tbTH[:, j2, 0]/tN, marker='v', ls='', color='C0')
    ax.plot(thCs, tbTH[:, j3, 0]/tN, marker='^', ls='', color='C0')
    ax.plot(thCs, tbG[:, j1, 0]/tN, marker='o', ls='', color='C1')
    ax.plot(thCs, tbG[:, j2, 0]/tN, marker='v', ls='', color='C1')
    ax.plot(thCs, tbG[:, j3, 0]/tN, marker='^', ls='', color='C1')
    ax.plot(thCs, tbPL[:, j1, 0]/tN, marker='o', ls='', color='C2')
    ax.plot(thCs, tbPL[:, j2, 0]/tN, marker='v', ls='', color='C2')
    ax.plot(thCs, tbPL[:, j3, 0]/tN, marker='^', ls='', color='C2')
    if nfit is 3:
        ax.plot(thCs, tbTH[:, j1, 1]/tN, marker='o', ls='', color='C0')
        ax.plot(thCs, tbTH[:, j2, 1]/tN, marker='v', ls='', color='C0')
        ax.plot(thCs, tbTH[:, j3, 1]/tN, marker='^', ls='', color='C0')
        ax.plot(thCs, tbG[:, j1, 1]/tN, marker='o', ls='', color='C1')
        ax.plot(thCs, tbG[:, j2, 1]/tN, marker='v', ls='', color='C1')
        ax.plot(thCs, tbG[:, j3, 1]/tN, marker='^', ls='', color='C1')
        ax.plot(thCs, tbPL[:, j1, 1]/tN, marker='o', ls='', color='C2')
        ax.plot(thCs, tbPL[:, j2, 1]/tN, marker='v', ls='', color='C2')
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
        ax.plot(thVoCs, tbG[i, :, 0]/tN, ls='--', color='C1')
        ax.plot(thVoCs, tbPL[i, :, 0]/tN, ls='--', color='C2')
        if nfit is 3:
            ax.plot(thVoCs, tbTH[i, :, 1]/tN, ls='-', color='C0')
            ax.plot(thVoCs, tbG[i, :, 1]/tN, ls='-', color='C1')
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


def makeGeneralPlot(nfit=None):

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

    alTH = []
    alG = []
    alPL = []

    thCs = np.array([0.02, 0.04, 0.08, 0.12, 0.15])
    thVs = np.array([0.0, 0.01, 0.03, 0.06, 0.1, 0.135, 0.2, 0.3, 0.4,
                     0.6, 0.8, 1.0, 1.2])
    # thCs = np.array([0.02])
    # thVoCs = np.array([0.25, 0.5, 0.75])

    plot = True

    NC = len(thCs)
    NV = len(thVs)
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
        aTH = []
        aG = []
        aPL = []
        print("theta_c: " + str(thC))
        for j, thV in enumerate(thVs):
            Y[0] = thV
            Y[2] = thC
            Y[3] = 5*thC
            print("theta_v: " + str(thV))
            print("  top hat")
            res = findBreaks(regime, NU, -1, Y, n=nfit, plot=plot,
                             ax=axTH[j, i])
            tTH.append([res[k+1] for k in range(res[0]-1)])
            aTH.append([res[k+res[0]] for k in range(res[0])])
            print("  Gaussian")
            res = findBreaks(regime, NU, 0, Y, n=nfit, plot=plot,
                             ax=axG[j, i])
            tG.append([res[k+1] for k in range(res[0]-1)])
            aG.append([res[k+res[0]] for k in range(res[0])])
            print("  Power law")
            res = findBreaks(regime, NU, 4, Y, n=nfit, plot=plot,
                             ax=axPL[j, i])
            tPL.append([res[k+1] for k in range(res[0]-1)])
            aPL.append([res[k+res[0]] for k in range(res[0])])
        tbTH.append(tTH)
        tbG.append(tG)
        tbPL.append(tPL)
        alTH.append(aTH)
        alG.append(aG)
        alPL.append(aPL)

    if plot:
        figTH.tight_layout()
        name = "breaks_general_lc_TH.png"
        print("Saving " + name)
        figTH.savefig(name)
        plt.close(figTH)

        figG.tight_layout()
        name = "breaks_general_lc_G.png"
        print("Saving " + name)
        figG.savefig(name)
        plt.close(figG)

        figPL.tight_layout()
        name = "breaks_general_lc_PL.png"
        print("Saving " + name)
        figPL.savefig(name)
        plt.close(figPL)

    tbTH = np.array(tbTH)
    tbG = np.array(tbG)
    tbPL = np.array(tbPL)

    print(tbTH)
    print(tbG)
    print(tbPL)

    """
    j1 = 0
    j2 = len(thVs)//2
    j3 = len(thVs)-1

    fig, ax = plt.subplots(1, 1)
    ax.plot(thCs, tbTH[:, j1, 0]/tN, marker='o', ls='', color='C0')
    ax.plot(thCs, tbTH[:, j2, 0]/tN, marker='v', ls='', color='C0')
    ax.plot(thCs, tbTH[:, j3, 0]/tN, marker='^', ls='', color='C0')
    ax.plot(thCs, tbG[:, j1, 0]/tN, marker='o', ls='', color='C1')
    ax.plot(thCs, tbG[:, j2, 0]/tN, marker='v', ls='', color='C1')
    ax.plot(thCs, tbG[:, j3, 0]/tN, marker='^', ls='', color='C1')
    ax.plot(thCs, tbPL[:, j1, 0]/tN, marker='o', ls='', color='C2')
    ax.plot(thCs, tbPL[:, j2, 0]/tN, marker='v', ls='', color='C2')
    ax.plot(thCs, tbPL[:, j3, 0]/tN, marker='^', ls='', color='C2')
    if nfit is 3:
        ax.plot(thCs, tbTH[:, j1, 1]/tN, marker='o', ls='', color='C0')
        ax.plot(thCs, tbTH[:, j2, 1]/tN, marker='v', ls='', color='C0')
        ax.plot(thCs, tbTH[:, j3, 1]/tN, marker='^', ls='', color='C0')
        ax.plot(thCs, tbG[:, j1, 1]/tN, marker='o', ls='', color='C1')
        ax.plot(thCs, tbG[:, j2, 1]/tN, marker='v', ls='', color='C1')
        ax.plot(thCs, tbG[:, j3, 1]/tN, marker='^', ls='', color='C1')
        ax.plot(thCs, tbPL[:, j1, 1]/tN, marker='o', ls='', color='C2')
        ax.plot(thCs, tbPL[:, j2, 1]/tN, marker='v', ls='', color='C2')
        ax.plot(thCs, tbPL[:, j3, 1]/tN, marker='^', ls='', color='C2')
    # ax.plot(thCs, np.power(2*np.sin(0.5*thCs*thVoCs), 8./3.), lw=4, ls='--',
    #         color='grey')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\theta_C$')
    ax.set_ylabel(r'$t_b / t_N$')
    fig.tight_layout()

    name = "breaks_general_all.pdf"
    print("Saving " + name)
    fig.savefig(name)
    plt.close(fig)
    """

    for i in range(len(thCs)):
        fig, ax = plt.subplots(1, 1)
        if nfit is 2:
            TB = np.array([tbTH[i][k][0] for k in range(NV)])
            ax.plot(thVs, TB/tN, ls='-', color='C0')
            TB = np.array([tbG[i][k][0] for k in range(NV)])
            ax.plot(thVs, TB/tN, ls='-', color='C1')
            TB = np.array([tbPL[i][k][0] for k in range(NV)])
            ax.plot(thVs, TB/tN, ls='-', color='C2')
        elif nfit is 3:
            TB = np.array([tbTH[i][k][0] for k in range(NV)])
            ax.plot(thVs, TB/tN, ls='--', color='C0')
            TB = np.array([tbG[i][k][0] for k in range(NV)])
            ax.plot(thVs, TB/tN, ls='--', color='C1')
            TB = np.array([tbPL[i][k][0] for k in range(NV)])
            ax.plot(thVs, TB/tN, ls='--', color='C2')
            TB = np.array([tbTH[i][k][1] for k in range(NV)])
            ax.plot(thVs, TB/tN, ls='-', color='C0')
            TB = np.array([tbG[i][k][1] for k in range(NV)])
            ax.plot(thVs, TB/tN, ls='-', color='C1')
            TB = np.array([tbPL[i][k][1] for k in range(NV)])
            ax.plot(thVs, TB/tN, ls='-', color='C2')
        else:
            TB = np.array([tbTH[i][k][0] for k in range(NV)])
            ax.plot(thVs, TB/tN, ls='--', color='C0')
            TB = np.array([tbG[i][k][0] for k in range(NV)])
            ax.plot(thVs, TB/tN, ls='--', color='C1')
            TB = np.array([tbPL[i][k][0] for k in range(NV)])
            ax.plot(thVs, TB/tN, ls='--', color='C2')
            TB = np.array([tbTH[i][k][-1] for k in range(NV)])
            ax.plot(thVs, TB/tN, ls='-', color='C0')
            TB = np.array([tbG[i][k][-1] for k in range(NV)])
            ax.plot(thVs, TB/tN, ls='-', color='C1')
            TB = np.array([tbPL[i][k][-1] for k in range(NV)])
            ax.plot(thVs, TB/tN, ls='-', color='C2')

        ax.plot(thVs, np.power(2*np.sin(0.5*np.fabs(thVs-thCs[i])), 8./3.),
                ls='--', color='grey')
        ax.plot(thVs, np.power(2*np.sin(0.5*np.fabs(thVs+thCs[i])), 8./3.),
                ls=':', color='grey')
        ax.plot(thVs, np.power(2*np.sin(0.5*thVs), 8./3.),
                ls='-', color='grey')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\theta_V$')
        ax.set_ylabel(r'$t_b / t_N$')
        fig.tight_layout()

        name = "breaks_general_thC_{0:.02f}.pdf".format(thCs[i])
        print("Saving " + name)
        fig.savefig(name)
        plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    for i in range(NC):
        al = [alTH[i][k][0] for k in range(NV)]
        ax.plot(thVs/thCs[i], al, color='C0', ls='', marker='.')
        al = [alTH[i][k][1] for k in range(NV)]
        ax.plot(thVs/thCs[i], al, color='C0', ls='', marker='+')
        if nfit is 3:
            al = [alTH[i][k][2] for k in range(NV)]
            ax.plot(thVs/thCs[i], al, color='C0', ls='', marker='x')
        al = [alG[i][k][0] for k in range(NV)]
        ax.plot(thVs/thCs[i], al, color='C1', ls='', marker='.')
        al = [alG[i][k][1] for k in range(NV)]
        ax.plot(thVs/thCs[i], al, color='C1', ls='', marker='+')
        if nfit is 3:
            al = [alG[i][k][2] for k in range(NV)]
            ax.plot(thVs/thCs[i], al, color='C1', ls='', marker='x')
        al = [alPL[i][k][0] for k in range(NV)]
        ax.plot(thVs/thCs[i], al, color='C2', ls='', marker='.')
        al = [alPL[i][k][1] for k in range(NV)]
        ax.plot(thVs/thCs[i], al, color='C2', ls='', marker='+')
        if nfit is 3:
            al = [alPL[i][k][2] for k in range(NV)]
            ax.plot(thVs/thCs[i], al, color='C2', ls='', marker='x')

    ax.set_xscale('log')
    ax.set_xlabel(r'$\theta_V / \theta_C$')
    ax.set_ylabel(r'$\alpha$')
    fig.tight_layout()

    name = "breaks_general_alpha.pdf"
    print("Saving " + name)
    fig.savefig(name)
    plt.close(fig)


if __name__ == "__main__":

    # makeOnAPlot()
    # makeOffAPlot()
    # makeVeryOffAPlot(2)
    makeGeneralPlot(3)
    # makeSlopesPlot()

    plt.show()
