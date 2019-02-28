import math
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as collections
import h5py as h5
import grbpy as grb

sec2year = grb.sec2day / 365.25
year2sec = grb.day2sec * 365.25

labelsize = 12
ticksize = 8
legendsize = 12


def makeThVPlots():

    E0 = 1.0e52
    thC = 0.05
    thV = 0.6
    thW = 0.5*np.pi
    b = 2
    n0 = 1.0e-3
    p = 2.2
    epse = 0.1
    epsB = 0.01
    xiN = 1.0
    dL = 1.23e26

    Y = np.array([thV, E0, thC, thW, b, 0.0, 0.0, 0.0, n0, p, epse, epsB,
                  xiN, dL])

    parstr = r"$n_0$={0:.01f}x$10^{{-3}}cm^{{-3}}$".format(n0/1.0e-3)
    parstr += r"  $p$={0:.01f}".format(p)
    parstr += r"  $\epsilon_e$={0:g}".format(epse)
    parstr += r"  $\epsilon_B$={0:g}".format(epsB)
    parstr += r"  $d_L$={0:.02f}x$10^{{26}}$cm".format(dL/1.0e26)

    N = 300
    nuO = (grb.c / 6.0e-5) * np.ones(N)

    t = grb.day2sec * np.geomspace(0.1, 300, N)

    thV = 0.0
    Y[0] = thV
    FnuA = grb.fluxDensity(t, nuO, 0, 0, *Y)
    thV = 2*thC
    Y[0] = thV
    FnuB = grb.fluxDensity(t, nuO, 0, 0, *Y)
    thV = 4*thC
    Y[0] = thV
    FnuC = grb.fluxDensity(t, nuO, 0, 0, *Y)
    thV = 6*thC
    Y[0] = thV
    FnuD = grb.fluxDensity(t, nuO, 0, 0, *Y)

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    ax.plot(t * grb.sec2day, FnuA, label=r'$\theta_V = 0$', color='k')
    ax.plot(t * grb.sec2day, FnuB, label=r'$\theta_V = 2\theta_C$',
            color='tab:purple')
    ax.plot(t * grb.sec2day, FnuC, label=r'$\theta_V = 4\theta_C$',
            color='tab:blue')
    ax.plot(t * grb.sec2day, FnuD, label=r'$\theta_V = 6\theta_C$',
            color='tab:green')
    ax.legend()
    ax.set_xlabel(r'$t$ (d)')
    ax.set_ylabel(r'$F_\nu$ (mJy)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(parstr)

    save(fig, 'GaussianThV.pdf')
    plt.close(fig)

    thC = 0.05
    thW = 0.3
    Y[2] = thC
    Y[3] = thW

    thV = 0.0
    Y[0] = thV
    FnuA = grb.fluxDensity(t, nuO, 4, 0, *Y)
    thV = 4*thC
    Y[0] = thV
    FnuB = grb.fluxDensity(t, nuO, 4, 0, *Y)
    thV = 8*thC
    Y[0] = thV
    FnuC = grb.fluxDensity(t, nuO, 4, 0, *Y)
    thV = 12*thC
    Y[0] = thV
    FnuD = grb.fluxDensity(t, nuO, 4, 0, *Y)

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    ax.plot(t * grb.sec2day, FnuA, label=r'$\theta_V = 0$', color='k')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ylim = ax.get_ylim()
    ax.plot(t * grb.sec2day, FnuB, label=r'$\theta_V = 2\theta_C$',
            color='tab:purple')
    ax.plot(t * grb.sec2day, FnuC, label=r'$\theta_V = 4\theta_C$',
            color='tab:blue')
    ax.plot(t * grb.sec2day, FnuD, label=r'$\theta_V = 6\theta_C$',
            color='tab:green')
    ax.legend()
    print(ylim)
    ax.set_xlabel(r'$t$ (d)')
    ax.set_ylabel(r'$F_\nu$ (mJy)')
    ax.set_ylim(*ylim)
    ax.set_title(parstr)

    save(fig, 'PowerlawThV.pdf')
    plt.close(fig)


def makeCentralPlots(modelPlots=True):

    thC = 0.1
    thW = 0.4
    b = 2
    E0 = 1.0e52
    n0 = 1.0e-2
    p = 2.2
    epse = 0.1
    epsB = 0.001
    xiN = 1.0
    dL = 1.23e26

    Y = np.array([0.0, E0, thC, thW, b, 0.0, 0.0, 0.0, n0, p, epse, epsB, xiN,
                  dL])
    latResTH = 10
    latResG = 10
    latResPL = 10

    N = 1000
    tday = np.geomspace(0.1, 1000, N)
    t = grb.day2sec * tday
    nu = 1.0e14 * np.ones(N)
    Flim = (1.0e-6, 1.0e1)

    if modelPlots:
        thV = 0.5 * thC
        Y[0] = thV
        FnuTH = grb.fluxDensity(t, nu, -1, 0, *Y, latRes=latResTH)
        FnuG = grb.fluxDensity(t, nu, 0, 0, *Y, latRes=latResG)
        FnuPL = grb.fluxDensity(t, nu, 4, 0, *Y, latRes=latResPL)

        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.plot(tday, FnuTH, label='Top-Hat', lw=2)
        ax.plot(tday, FnuG, label='Gaussian', lw=2)
        ax.plot(tday, FnuPL, label='Power-Law', lw=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(*Flim)
        ax.set_xlabel(r'$t$ (days)', fontsize=labelsize)
        ax.set_ylabel(r'$F_\nu$ (mJy)', fontsize=labelsize)
        ax.tick_params(labelsize=ticksize)
        ax.legend(fontsize=legendsize)
        fig.tight_layout()
        figname = "lc_on.pdf"
        save(fig, figname)
        plt.close(fig)

        thV = 2*thC
        Y[0] = thV
        FnuTH = grb.fluxDensity(t, nu, -1, 0, *Y, latRes=latResTH)
        FnuG = grb.fluxDensity(t, nu, 0, 0, *Y, latRes=latResG)
        FnuPL = grb.fluxDensity(t, nu, 4, 0, *Y, latRes=latResPL)

        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.plot(tday, FnuTH, label='Top-Hat', lw=2)
        ax.plot(tday, FnuG, label='Gaussian', lw=2)
        ax.plot(tday, FnuPL, label='Power-Law', lw=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(*Flim)
        ax.set_xlabel(r'$t$ (days)', fontsize=labelsize)
        ax.set_ylabel(r'$F_\nu$ (mJy)', fontsize=labelsize)
        ax.tick_params(labelsize=ticksize)
        ax.legend(fontsize=legendsize, loc='lower left')
        fig.tight_layout()
        figname = "lc_off1.pdf"
        save(fig, figname)
        plt.close(fig)

        thV = 6*thC
        Y[0] = thV
        FnuTH = grb.fluxDensity(t, nu, -1, 0, *Y, latRes=latResTH)
        FnuG = grb.fluxDensity(t, nu, 0, 0, *Y, latRes=latResG)
        FnuPL = grb.fluxDensity(t, nu, 4, 0, *Y, latRes=latResPL)

        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.plot(tday, FnuTH, label='Top-Hat', lw=2)
        ax.plot(tday, FnuG, label='Gaussian', lw=2)
        ax.plot(tday, FnuPL, label='Power-Law', lw=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(*Flim)
        ax.set_xlabel(r'$t$ (days)', fontsize=labelsize)
        ax.set_ylabel(r'$F_\nu$ (mJy)', fontsize=labelsize)
        ax.tick_params(labelsize=ticksize)
        ax.legend(fontsize=legendsize)
        fig.tight_layout()
        figname = "lc_off2.pdf"
        save(fig, figname)
        plt.close(fig)

    fig = plt.figure(figsize=(4, 4))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    nsegs = 12
    width = 0.05
    cmap = cm.inferno

    thWd = 180.0*thW / np.pi
    dth = thWd / nsegs
    shockFront = []
    shockBody = []
    cols = []
    for i in range(nsegs):
        th1 = i*dth
        th2 = (i+1)*dth
        th = 0.5*(th1+th2)
        # col = np.exp(-0.5*th*th / (thCd*thCd))
        col = 1.0 - th*th / (thWd*thWd)

        p1 = patches.Wedge((0., 0.), 1., 90-th2, 90-th1, width=width)
        p2 = patches.Wedge((0., 0.), 1., 90+th1, 90+th2, width=width)
        shockFront.append(p1)
        shockFront.append(p2)
        p1 = patches.Wedge((0., 0.), 1.-width, 90-th2, 90-th1)
        p2 = patches.Wedge((0., 0.), 1.-width, 90+th1, 90+th2)
        shockBody.append(p1)
        shockBody.append(p2)
        cols.append(col)
        cols.append(col)

    p = collections.PatchCollection(shockFront, cmap=cmap)
    p.set_array(np.array(cols))
    ax.add_collection(p)

    p = collections.PatchCollection(shockBody, alpha=0.3, cmap=cmap)
    p.set_array(np.array(cols))
    ax.add_collection(p)

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    figname = "jet.pdf"
    save(fig, figname)
    plt.close(fig)

    fig = plt.figure(figsize=(4, 3))
    lbuff = 0.16
    bbuff = 0.16
    rbuff = 0.15
    tbuff = 0.05
    space = 0.01
    width2 = 0.03
    width1 = 1.0 - lbuff-rbuff-space-width2
    height = 1.0-bbuff-tbuff

    ax = fig.add_axes([lbuff, bbuff, width1, height])
    ax2 = fig.add_axes([lbuff+width1+space, bbuff, width2, height])

    tday = np.geomspace(0.01, 1000, N)
    t = grb.day2sec * tday
    thV = 1.25*thW
    Y[0] = thV
    dth = thW / nsegs

    colList = []

    for i in range(nsegs):

        th1 = i*dth
        th2 = (i+1)*dth
        x = float(i) / (nsegs-1.0)
        th = x * th2 + (1-x) * th1
        th_h = 0.5*(th1+th2)
        E = E0 * np.exp(-0.5*th*th / (thC*thC))
        col = 1.0 - th_h*th_h / (thW*thW)

        Y[1] = E
        Y[2] = th1
        Y[3] = th2
        FnuC = grb.fluxDensity(t, nu, -2, 0, *Y)

        ax.plot(tday, FnuC, color=cmap(col))
        colList.append(cmap(col))

    # myCmap = cmap.from_list('Custom cmap', colList, nsegs)
    myCmap = mpl.colors.ListedColormap(colList)
    thBounds = dth*np.arange(nsegs+1)
    norm = mpl.colors.BoundaryNorm(thBounds, nsegs)
    cb = mpl.colorbar.ColorbarBase(ax2, cmap=myCmap, norm=norm,
                                   spacing='proportional', ticks=thBounds,
                                   format='%.2f')
    cb.set_label(r'$\theta$ (rad)', fontsize=labelsize)

    Y[1] = E0
    Y[2] = thC
    Y[3] = thW
    FnuG = grb.fluxDensity(t, nu, 0, 0, *Y)
    ax.plot(tday, FnuG, color='grey', lw=2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1.0e-2, 1.0e3)
    ax.set_ylim(1.0e-7, 1.0e-2)
    ax.set_xlabel(r'$t$ (days)', fontsize=labelsize)
    ax.set_ylabel(r'$F_\nu$ (mJy)', fontsize=labelsize)
    ax.tick_params(labelsize=ticksize)
    ax2.tick_params(labelsize=ticksize)
    # fig.tight_layout()
    figname = "lc_decomp.pdf"
    save(fig, figname)
    plt.close(fig)


def makeConvergencePlots():

    lrs = [1, 3, 5, 10, 20, 40]
    trs = [250, 500, 1000, 2000, 4000, 8000]
    lrRef = 80
    trRef = 16000

    # lss = ['-', '--', '-.', ':', '-']
    # cols = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

    E0 = 1.0e52
    thC = 0.05
    thV = 0.3
    thW = 0.5
    b = 2
    n0 = 1.0
    p = 2.2
    epse = 0.1
    epsB = 0.01
    xiN = 1.0
    dL = 1.23e26

    Y = np.array([thV, E0, thC, thW, b, 0.0, 0.0, 0.0, n0, p, epse, epsB, xiN,
                  dL])
    jetType = 0

    t = np.geomspace(1.0e3, 1.0e8, 100)
    nu = np.empty(t.shape)
    nu[:] = 1.0e18

    FnuRef = grb.fluxDensity(t, nu, jetType, 0, *Y, latRes=lrRef, tRes=trRef)
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(t, FnuRef)

    ax.set_xlabel(r'$t$ (s)')
    ax.set_ylabel('')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.tight_layout()
    save(fig, "conv_ref.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    fig2, ax2 = plt.subplots(1, 1, figsize=(4, 3))

    for i, lr in enumerate(lrs):
        Fnu = grb.fluxDensity(t, nu, jetType, 0, *Y, latRes=lr, tRes=trRef)
        err = np.fabs(Fnu-FnuRef)/FnuRef
        ax.plot(t, err)
        ax2.plot(t, Fnu)
    ax2.plot(t, FnuRef, lw=0.5, color='k')

    ax.set_xlabel(r'$t$ (s)')
    ax.set_ylabel('Error')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.tight_layout()
    save(fig, "conv_latRes.pdf")
    plt.close(fig)

    ax2.set_xlabel(r'$t$ (s)')
    ax2.set_ylabel('Flux')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    fig2.tight_layout()
    save(fig2, "conv_flux_latRes.pdf")
    plt.close(fig2)

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    fig2, ax2 = plt.subplots(1, 1, figsize=(4, 3))

    for i, tr in enumerate(trs):
        Fnu = grb.fluxDensity(t, nu, jetType, 0, *Y, latRes=lrRef, tRes=tr)
        err = np.fabs(Fnu-FnuRef)/FnuRef
        ax.plot(t, err)
        ax2.plot(t, Fnu)

    ax.set_xlabel(r'$t$ (s)')
    ax.set_ylabel('Error')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.tight_layout()
    save(fig, "conv_tRes.pdf")
    plt.close(fig)

    ax2.set_xlabel(r'$t$ (s)')
    ax2.set_ylabel('Flux')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    fig2.tight_layout()
    save(fig2, "conv_flux_tRes.pdf")
    plt.close(fig2)


def get_powers(regime='D', p=2.2):
    if regime is 'D':
        beta = 1.0/3.0
        s1 = 1.0
        s2 = 0.0
    elif regime is 'E':
        beta = 1.0/3.0
        s1 = 7.0/3.0
        s2 = 2.0/3.0
    elif regime is 'F':
        beta = -0.5
        s1 = 1.5
        s2 = -1.0
    elif regime is 'G':
        beta = 0.5*(1-p)
        s1 = 0.5*(3*p+1)
        s2 = 0.0
    elif regime is 'H':
        beta = -0.5*p
        s1 = 1.5*p
        s2 = -1.0

    al1 = s1 - 2.0*s2/3.0 - beta
    D = 12 - 1.5*al1 - 4*beta

    return beta, al1, D, s1, s2, p


def f_cone(th, thC, thW):
    ath = np.atleast_1d(th)
    f = np.empty(ath.shape)
    valid = ath >= thC & ath <= thW
    f[valid] = 1.0
    f[~valid] = 0.0
    return f


def f_tophat(th, thC, thW):
    ath = np.atleast_1d(th)
    f = np.empty(ath.shape)
    valid = ath <= thC
    f[valid] = 1.0
    f[~valid] = 0.0
    return f


def f_gaussian(th, thC, thW):
    ath = np.atleast_1d(th)
    f = np.empty(ath.shape)
    valid = ath <= thW
    f[valid] = np.exp(-0.5*(ath*ath)[valid] / (thC*thC))
    f[~valid] = 0.0
    return f


def f_powerlaw(th, thC, thW, b=2):
    ath = np.atleast_1d(th)
    f = np.empty(ath.shape)
    valid = ath <= thW
    f[valid] = 1.0 / np.power(np.sqrt(1.0 + (ath[valid] / thC)**2), b)
    f[~valid] = 0.0
    return f


def get_f(jetModel, thC, thW, b=2):
    if jetModel == -2:
        return lambda th: f_cone(th, thC, thW)
    elif jetModel == -1:
        return lambda th: f_tophat(th, thC, thW)
    elif jetModel == 0:
        return lambda th: f_gaussian(th, thC, thW)
    elif jetModel == 4:
        return lambda th: f_powerlaw(th, thC, thW, b)
    else:
        return None


def calc_g(th, thV, f):

    dth = 1.0e-4

    fR = f(th+dth)
    fL = f(th-dth)
    dlfdth = np.log(fR/fL) / (2*dth)

    g = -2*np.tan(0.5*np.fabs(thV-th)) * dlfdth

    return g


def calc_tp(th, tN0, thV, f, regime, p):

    chim = 2*np.sin(0.5*np.fabs(thV-th))
    chip = 2*np.sin(0.5*(thV+th))
    beta, al1, D, s1, s2, p = get_powers(regime, p)

    fac2 = math.pow(2.0, -(8-3*beta) / (2.0*D))
    facD = math.pow(4.0*D, 1.0/(4.0*D))
    facChi = np.power(chim, 1.0-1.0/(2.0*D)) * np.power(chip, 1.0/(2.0*D))
    facChi = chim * np.power(chip/chim, 1.0/(2.0*D))
    facChi = chim

    tp = tN0 * np.power(f(th), 1.0/3.0) * np.power(fac2*facD*facChi, 8.0/3.0)
    return tp


def calc_dOm(th, tN0, thV, f, regime='G', n0=1.0e-3, p=2.2, epse=0.1,
             epsB=0.01, dL=1.0e26):

    chi = 2*np.sin(0.5*(thV-th))
    beta, al1, D, s1, s2, p = get_powers(regime, p)

    # dth = 1.0e-4
    # thp = th+dth
    # thm = th-dth
    # fc = f(th)
    # fp = f(thp)
    # fm = f(thm)
    # d2lfdth2 = np.log(fp*fm/(fc*fc)) / (dth*dth)
    # thCn = 1.0/np.sqrt(np.fabs(d2lfdth2))
    # dOmth = thCn
    dOmth = 0.1

    return chi * dOmth


def calc_Ip(th, tN0, thV, f, nu, regime='G', n0=1.0e-3, p=2.2, epse=0.1,
            epsB=0.01, dL=1.0e26):

    chi = 2*np.sin(0.5*(thV-th))
    beta, al1, D, s1, s2, p = get_powers(regime, p)

    n = 4*n0
    eth = n * grb.mp * grb.c**2
    B = math.sqrt(8*np.pi*epsB * eth)
    eP = 0.5*(p-1)*math.sqrt(3)*(grb.ee**3) / (grb.me * grb.c*grb.c) * n*B
    gm = (p-2.)/(p-1.) * epse * eth / (n * grb.me * grb.c**2)
    gc = 6*np.pi * grb.me*grb.c / (grb.sigmaT * B*B * tN0)
    num = 3.0/(4*np.pi) * grb.ee * B * gm*gm / (grb.me * grb.c)
    nuc = 3.0/(4*np.pi) * grb.ee * B * gc*gc / (grb.me * grb.c)

    if regime is 'D':
        nufac = np.power(nu/num, 1.0/3.0)
    elif regime is 'E':
        nufac = np.power(nu/nuc, 1.0/3.0)
    elif regime is 'F':
        nufac = np.power(nu/nuc, -0.5)
    elif regime is 'G':
        nufac = np.power(nu/num, 0.5*(1-p))
    elif regime is 'H':
        nufac = np.power(nuc/num, 0.5*(1-p)) * np.power(nu/nuc, -0.5*p)

    Rfac = (grb.c * tN0) ** 3 / (4*np.pi*dL*dL) * math.sqrt(2)/12.0

    I0 = eP * nufac * Rfac

    Ip = np.power(f(th), 1.0+s2/3.0) * np.power(chi, -s1+2*s2/3.0+beta)

    return I0 * Ip * grb.cgs2mJy


def calc_g_from_al_som(al, som, regime, p):

    beta, al1, D, s1, s2, p = get_powers(regime, p)
    al2 = 3+s2

    return (-3*al1 - 8*al + 3*som) / (al - al2)


def calc_som_from_al_g(al, g, regime, p):

    beta, al1, D, s1, s2, p = get_powers(regime, p)
    al2 = 3+s2

    return (8*al + (al-al2)*g + 3*al1) / 3.0


def makeNumAnaCompPlots(regime='G', jetModel=0):

    fig, ax = plt.subplots(5, 2, figsize=(10, 15))

    N = 100

    NU, tN0, Y, beta = getRegimePars(regime)

    nu = np.empty(N)
    nu[:] = NU
    # thVs = [0.1, 0.3, 0.5, 0.7, 0.1, 0.3, 0.5, 0.7, 0.1, 0.3, 0.5, 0.7]
    # thCs = [0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2]
    # thWs = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    thVs = [0.2, 0.5, 0.8]
    thCs = [0.05, 0.05, 0.05]
    thWs = [0.4, 0.4, 0.4]

    b = Y[4]
    n0 = Y[8]
    p = Y[9]
    epse = Y[10]
    epsB = Y[11]
    dL = Y[13]

    nufac = 1.1

    ax[2, 0].axhline(beta, lw=2, color='grey', alpha=0.5)
    ax[2, 1].axhline(beta, lw=2, color='grey', alpha=0.5)

    for thV, thC, thW in zip(thVs, thCs, thWs):

        chi = np.geomspace(max(2*math.sin(0.5*(thV-thW)), 1.0e-2),
                           2*math.sin(0.5*(thV)),
                           N)
        th = thV - 2*np.arcsin(0.5*chi)

        f = get_f(jetModel, thC, thW, b)
        tp = calc_tp(th, tN0, thV, f, regime, p)
        Ip = calc_Ip(th, tN0, thV, f, nu, regime, n0, p, epse, epsB, dL)
        dOm = calc_dOm(th, tN0, thV, f, regime, p)

        Y[0] = thV
        Y[2] = thC
        Y[3] = thW
        Fnu = grb.fluxDensity(tp, nu, jetModel, 0, *Y, latRes=10)
        FnuR = grb.fluxDensity(tp, nu*nufac, jetModel, 0, *Y, latRes=10)
        FnuL = grb.fluxDensity(tp, nu/nufac, jetModel, 0, *Y, latRes=10)
        FnuA = Ip*dOm

        alA = np.log(FnuA[1:]/FnuA[:-1]) / np.log(tp[1:]/tp[:-1])
        al = np.log(Fnu[1:]/Fnu[:-1]) / np.log(tp[1:]/tp[:-1])

        betaN = np.log(FnuR/FnuL) / np.log(nufac*nufac)

        chiC = np.sqrt(chi[1:]*chi[:-1])
        thCC = thV - 2*np.arcsin(0.5*chiC)

        gA = calc_g(thCC, thV, f)
        somA = 1.0 * np.ones(thCC.shape)
        gN = calc_g_from_al_som(al, somA, regime, p)
        somN = calc_som_from_al_g(al, gA, regime, p)

        tpC = np.sqrt(tp[1:]*tp[:-1])
        label = r'$\theta_V =$ {0:.1f}'.format(thV)
        label += r', $\theta_C = $ {0:.1f}'.format(thC)
        label += r', $\theta_W = $ {0:.1f}'.format(thW)
        ax[0, 0].plot(chi, Fnu, label=label)
        ax[0, 0].plot(chi, FnuA)
        ax[1, 0].plot(chiC, al)
        ax[1, 0].plot(chiC, alA)
        ax[2, 0].plot(chi, betaN)
        ax[3, 0].plot(chiC, gN)
        ax[3, 0].plot(chiC, gA)
        ax[4, 0].plot(chiC, somN)
        ax[4, 0].plot(chiC, somA)
        ax[0, 1].plot(tp, Fnu)
        ax[0, 1].plot(tp, FnuA)
        ax[1, 1].plot(tpC, al)
        ax[1, 1].plot(tpC, alA)
        ax[2, 1].plot(tp, betaN)
        ax[3, 1].plot(tpC, gN)
        ax[3, 1].plot(tpC, gA)
        ax[4, 1].plot(tpC, somN)
        ax[4, 1].plot(tpC, somA)

    ax[0, 0].legend()
    ax[0, 0].set_xscale('log')
    ax[0, 0].set_yscale('log')
    ax[1, 0].set_xscale('log')
    ax[1, 0].set_ylim(-3.0, 3.0)
    ax[2, 0].set_xscale('log')
    ax[3, 0].set_xscale('log')
    ax[3, 0].set_yscale('log')
    ax[4, 0].set_xscale('log')
    ax[0, 1].set_xscale('log')
    ax[0, 1].set_yscale('log')
    ax[1, 1].set_xscale('log')
    ax[1, 1].set_ylim(-3.0, 3.0)
    ax[2, 1].set_xscale('log')
    ax[3, 1].set_xscale('log')
    ax[3, 1].set_yscale('log')
    ax[4, 1].set_xscale('log')
    ax[0, 0].set_ylabel(r'$F_\nu$')
    ax[1, 0].set_ylabel(r'$\alpha$')
    ax[2, 0].set_ylabel(r'$\beta$')
    ax[3, 0].set_ylabel(r'$g$')
    ax[4, 0].set_ylabel(r'$s_{{\Omega}}$')
    ax[4, 0].set_xlabel(r'$\chi$')
    ax[4, 1].set_xlabel(r'$t_{{obs}}$')
    fig.tight_layout()

    plotname = 'numanacomp.pdf'
    save(fig, plotname)
    plt.close(fig)


def makeNormalizationPlots(regime='G', jetModel=0):

    fig, ax = plt.subplots(4, 2, figsize=(12, 15))

    NU, tN0, Y, beta = getRegimePars(regime)
    N = 100

    nu = np.empty(N)
    nu[:] = NU
    # thVs = [0.1, 0.3, 0.5, 0.7, 0.1, 0.3, 0.5, 0.7, 0.1, 0.3, 0.5, 0.7]
    # thCs = [0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2]
    # thWs = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    thVs = [0.1, 0.3, 0.5, 0.7, 0.1, 0.3, 0.5, 0.7]
    thCs = [0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2]
    thWs = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]

    b = Y[4]
    n0 = Y[8]
    p = Y[9]
    epse = Y[10]
    epsB = Y[11]
    dL = Y[13]

    nufac = 1.1

    ax[3, 0].axhline(beta, lw=2, color='grey', alpha=0.5)
    ax[3, 1].axhline(beta, lw=2, color='grey', alpha=0.5)

    for thV, thC, thW in zip(thVs, thCs, thWs):

        chi = np.geomspace(max(2*math.sin(0.5*(thV-thW)), 1.0e-2),
                           2*math.sin(0.5*(thV)),
                           N)
        th = thV - 2*np.arcsin(0.5*chi)

        f = get_f(jetModel, thC, thW, b)
        tp = calc_tp(th, tN0, thV, f, regime, p)
        Ip = calc_Ip(th, tN0, thV, f, nu, regime, n0, p, epse, epsB, dL)
        # dth = 1.0e-4
        # fl = f(th-dth)
        # fc = f(th)
        # fr = f(th+dth)

        # dlfdth = np.log(fr/fl)/(2*dth)
        # d2lfdth2 = np.log(fr*fl/(fc*fc))/(dth*dth)
        # dth1 = np.power(np.fabs(dlfdth), -1.0)
        # dth2 = np.power(np.fabs(d2lfdth2), -0.5)
        dth2 = thC  # 0.1

        Y[0] = thV
        Y[2] = thC
        Y[3] = thW
        Fnu = grb.fluxDensity(tp, nu, jetModel, 0, *Y, latRes=20)
        FnuR = grb.fluxDensity(tp, nu*nufac, jetModel, 0, *Y, latRes=10)
        FnuL = grb.fluxDensity(tp, nu/nufac, jetModel, 0, *Y, latRes=10)
        betaN = np.log(FnuR/FnuL) / np.log(nufac*nufac)

        norm = Fnu/Ip
        som = np.log(norm[1:]/norm[:-1]) / np.log(chi[1:]/chi[:-1])
        chiC = np.sqrt(chi[1:]*chi[:-1])
        tpC = np.sqrt(tp[1:]*tp[:-1])
        label = r'$\theta_V =$ {0:.1f}'.format(thV)
        label += r', $\theta_C = $ {0:.1f}'.format(thC)
        label += r', $\theta_W = $ {0:.1f}'.format(thW)
        linelist = ax[0, 0].plot(chi, Fnu, label=label)
        ax[1, 0].plot(chi, norm)
        # ax[1, 0].plot(chi, chi*dth1, color=linelist[0].get_color(), ls='--')
        ax[1, 0].plot(chi, chi*dth2, color=linelist[0].get_color(), ls=':')
        ax[2, 0].plot(chiC, som)
        ax[3, 0].plot(chi, betaN)
        ax[0, 1].plot(tp, Fnu)
        ax[1, 1].plot(tp, norm)
        ax[2, 1].plot(tpC, som)
        ax[3, 1].plot(tp, betaN)

    ax[0, 0].legend()
    ax[0, 0].set_xscale('log')
    ax[0, 0].set_yscale('log')
    ax[1, 0].set_xscale('log')
    ax[1, 0].set_yscale('log')
    ax[1, 0].set_ylim(1.0e-6, 1.0)
    ax[2, 0].set_xscale('log')
    ax[2, 0].set_ylim(-2.0, 6.0)
    ax[3, 0].set_xscale('log')
    ax[0, 1].set_xscale('log')
    ax[0, 1].set_yscale('log')
    ax[1, 1].set_xscale('log')
    ax[1, 1].set_yscale('log')
    ax[2, 1].set_xscale('log')
    ax[2, 1].set_ylim(-2.0, 6.0)
    ax[3, 1].set_xscale('log')
    ax[0, 0].set_ylabel(r'$F_\nu$')
    ax[1, 0].set_ylabel(r'$\Delta \Omega = F_\nu / I_\nu^*$')
    ax[2, 0].set_ylabel(r'$s_\Omega$')
    ax[3, 0].set_ylabel(r'$\beta$')
    ax[2, 0].set_xlabel(r'$\chi$')
    ax[2, 1].set_xlabel(r'$t_{{obs}}$')
    fig.tight_layout()

    plotname = 'som.pdf'
    save(fig, plotname)
    plt.close(fig)


def getRegimePars(regime='D'):

    n0_slow = 1.0
    n0_fast = 1.0e6
    tN0 = 400 * 24 * 3600.
    E0_slow = 16*np.pi/9.0 * n0_slow * grb.mp * grb.c**5 * tN0**3
    E0_fast = 16*np.pi/9.0 * n0_fast * grb.mp * grb.c**5 * tN0**3

    thC = 0.1
    thW = 0.4
    thV = 0.0
    b = 8.0

    p = 2.2
    epse_slow = 0.1
    epse_fast = 0.3
    epsB_slow = 1.0e-3
    epsB_fast = 0.3
    dL = 1.23e26

    Yslow = np.array([thV, E0_slow, thC, thW, b, 0.0, 0.0, 0.0, n0_slow, p,
                      epse_slow, epsB_slow, 1.0, dL])
    Yfast = np.array([thV, E0_fast, thC, thW, b, 0.0, 0.0, 0.0, n0_fast, p,
                      epse_fast, epsB_fast, 1.0, dL])

    betaD = 1.0/3.0
    betaE = 1.0/3.0
    betaF = -0.5
    betaG = 0.5*(1-p)
    betaH = -0.5*p

    nuD = 1.0e-6
    nuE = 1.0e-6
    nuF = 1.0e7
    nuG = 3.0e15
    nuH = 1.0e30

    if regime is 'D':
        return nuD, tN0, Yslow, betaD
    elif regime is 'E':
        return nuE, tN0, Yfast, betaE
    elif regime is 'F':
        return nuF, tN0, Yfast, betaF
    elif regime is 'G':
        return nuG, tN0, Yslow, betaG
    elif regime is 'H':
        return nuH, tN0, Yslow, betaH

    return None


def makeRegimePlots():

    nuD, tND, YD, betaD = getRegimePars('D')
    nuE, tNE, YE, betaE = getRegimePars('E')
    nuF, tNF, YF, betaF = getRegimePars('F')
    nuG, tNG, YG, betaG = getRegimePars('G')
    nuH, tNH, YH, betaH = getRegimePars('H')

    N = 100
    tN0 = tNG

    jetModel = 4
    nu = np.geomspace(1.0e0, 1.0e20, N)
    t = np.empty(N)
    Ts = np.logspace(-5, 0, base=10.0, num=6) * tN0

    fig, ax = plt.subplots(1, 1)
    figF, axF = plt.subplots(1, 1)
    figS, axS = plt.subplots(1, 1)

    axF.axvline(nuE, color='grey')
    axF.axvline(nuF, color='grey')
    axF.axvline(nuH, color='grey')

    axS.axvline(nuD, color='grey')
    axS.axvline(nuG, color='grey')
    axS.axvline(nuH, color='grey')
    for T in Ts:
        t[:] = T
        FnuF = grb.fluxDensity(t, nu, jetModel, 0, *YE)
        FnuS = grb.fluxDensity(t, nu, jetModel, 0, *YG)
        axF.plot(nu, FnuF)
        axS.plot(nu, FnuS)

    axF.set_xscale('log')
    axF.set_yscale('log')
    axS.set_xscale('log')
    axS.set_yscale('log')

    plotname = 'specFast.pdf'
    save(figF, plotname)
    plt.close(figF)

    plotname = 'specSlow.pdf'
    save(figS, plotname)
    plt.close(figS)

    t = np.geomspace(1.0e-6*tN0, 1.0*tN0, N)
    nu = np.empty(N)

    nu[:] = nuD
    FnuD = grb.fluxDensity(t, nu, jetModel, 0, *YD)
    nu[:] = nuE
    FnuE = grb.fluxDensity(t, nu, jetModel, 0, *YE)
    nu[:] = nuF
    FnuF = grb.fluxDensity(t, nu, jetModel, 0, *YF)
    nu[:] = nuG
    FnuG = grb.fluxDensity(t, nu, jetModel, 0, *YG)
    nu[:] = nuH
    FnuH = grb.fluxDensity(t, nu, jetModel, 0, *YH)

    ax.plot(t, FnuD)
    ax.plot(t, FnuE)
    ax.plot(t, FnuF)
    ax.plot(t, FnuG)
    ax.plot(t, FnuH)

    ax.set_xscale('log')
    ax.set_yscale('log')

    plotname = 'lc_regimes.pdf'
    save(fig, plotname)
    plt.close(fig)


def makeCharEvolPlots():

    nuD, tND, YD, betaD = getRegimePars('D')
    nuE, tNE, YE, betaE = getRegimePars('E')
    nuF, tNF, YF, betaF = getRegimePars('F')
    nuG, tNG, YG, betaG = getRegimePars('G')
    nuH, tNH, YH, betaH = getRegimePars('H')

    N = 100
    tN0 = tNG

    t = np.geomspace(1.0e-5 * tN0, tN0, N)
    nu = np.empty(N)

    jetModel = 0

    pD = betaD
    pE = betaE
    pF = betaF
    pG = betaG
    pHs = betaH
    pHf = betaH

    figs, axs = plt.subplots(2, 2)
    figf, axf = plt.subplots(2, 2)

    thVs = [0.0, 0.25, 0.5]

    for thV in thVs:
        YD[0] = thV
        YE[0] = thV
        YF[0] = thV
        YG[0] = thV
        YH[0] = thV

        nu[:] = nuD
        FD = grb.fluxDensity(t, nu, jetModel, 0, *YD)
        nu[:] = nuE
        FE = grb.fluxDensity(t, nu, jetModel, 0, *YE)
        nu[:] = nuF
        FF = grb.fluxDensity(t, nu, jetModel, 0, *YF)
        nu[:] = nuG
        FG = grb.fluxDensity(t, nu, jetModel, 0, *YG)
        nu[:] = nuH
        FHs = grb.fluxDensity(t, nu, jetModel, 0, *YH)
        FHf = grb.fluxDensity(t, nu, jetModel, 0, *YF)

        # Slow Cooling

        nums = np.power(FG/FD * np.power(nuD, pD) * np.power(nuG, -pG),
                        1.0/(pD-pG))
        nucs = np.power(FHs/FG * np.power(nuG, pG) * np.power(nuH, -pHs),
                        1.0/(pG-pHs))
        Fms = np.power(np.power(FD, pG) * np.power(FG, -pD)
                       * np.power(nuG/nuD, pD*pG), 1.0/(pG-pD))
        Fcs = np.power(np.power(FG, pHs) * np.power(FHs, -pG)
                       * np.power(nuH/nuG, pG*pHs), 1.0/(pHs-pG))

        # Fast Cooling

        nucf = np.power(FF/FE * np.power(nuE, pE) * np.power(nuF, -pF),
                        1.0/(pE-pF))
        numf = np.power(FHf/FF * np.power(nuF, pF) * np.power(nuH, -pHf),
                        1.0/(pF-pHf))
        Fcf = np.power(np.power(FE, pF) * np.power(FF, -pE)
                       * np.power(nuF/nuE, pE*pF), 1.0/(pF-pE))
        Fmf = np.power(np.power(FF, pHf) * np.power(FHf, -pF)
                       * np.power(nuH/nuF, pF*pHf), 1.0/(pHf-pF))

        axs[0, 0].plot(t/tN0, Fms,
                       label=r'$\theta_V = {0:.2}$ rad'.format(thV))
        axs[0, 1].plot(t/tN0, Fcs)
        axs[1, 0].plot(t/tN0, nums)
        axs[1, 1].plot(t/tN0, nucs)
        axf[0, 0].plot(t/tN0, Fmf,
                       label=r'$\theta_V = {0:.2}$ rad'.format(thV))
        axf[0, 1].plot(t/tN0, Fcf)
        axf[1, 0].plot(t/tN0, numf)
        axf[1, 1].plot(t/tN0, nucf)

    axs[0, 0].legend()
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_ylabel(r'$F_m$')
    axs[0, 1].set_xscale('log')
    axs[0, 1].set_yscale('log')
    axs[0, 1].set_ylabel(r'$F_c$')
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_ylabel(r'$\nu_m$')
    axs[1, 1].set_xscale('log')
    axs[1, 1].set_yscale('log')
    axs[1, 1].set_ylabel(r'$\nu_c$')
    axs[1, 0].set_xlabel(r'$\tau$')
    axs[1, 1].set_xlabel(r'$\tau$')
    figs.tight_layout()

    axf[0, 0].legend()
    axf[0, 0].set_xscale('log')
    axf[0, 0].set_yscale('log')
    axf[0, 0].set_ylabel(r'$F_m$')
    axf[0, 1].set_xscale('log')
    axf[0, 1].set_yscale('log')
    axf[0, 1].set_ylabel(r'$F_c$')
    axf[1, 0].set_xscale('log')
    axf[1, 0].set_yscale('log')
    axf[1, 0].set_ylabel(r'$\nu_m$')
    axf[1, 1].set_xscale('log')
    axf[1, 1].set_yscale('log')
    axf[1, 1].set_ylabel(r'$\nu_c$')
    axf[1, 0].set_xlabel(r't')
    axf[1, 1].set_xlabel(r't')
    figf.tight_layout()

    plotname = "charEvolSlow.pdf"
    save(figs, plotname)
    plt.close(figs)

    plotname = "charEvolFast.pdf"
    save(figf, plotname)
    plt.close(figf)


def calcTN0(Y, aE0=1.0, an0=1.0):

    if Y is not None:
        E0 = Y[1]
        n0 = Y[8]
    else:
        E0 = aE0
        n0 = an0
    tN0 = np.power(9.0 * E0 / (16*np.pi * n0 * grb.mp * grb.c**5), 1.0/3.0)

    return tN0


def calcTW(jetModel, Y, regime):

    thV = Y[0]
    thC = Y[2]
    thW = Y[3]
    b = Y[4]
    p = Y[9]

    if thW >= thV:
        return 0.0

    f = get_f(jetModel, thC, thW, b)
    tN0 = calcTN0(Y)

    tW = np.asscalar(calc_tp(thW, tN0, thV, f, regime, p))

    return tW


def calcTB(jetModel, Y, regime):

    thV = Y[0]
    thC = Y[2]
    thW = Y[3]
    b = Y[4]
    p = Y[9]

    f = get_f(jetModel, thC, thW, b)
    tN0 = calcTN0(Y)

    tB = np.asscalar(calc_tp(0.0, tN0, thV, f, regime, p))

    return tB


def calcTBpm(jetModel, Y, regime):

    thV = Y[0]
    thC = Y[2]
    thW = Y[3]
    b = Y[4]
    p = Y[9]

    f = get_f(jetModel, thC, thW, b)
    tN0 = calcTN0(Y)

    tBpm = calc_tp(thC, tN0, thV, f, regime, p)

    return tBpm


def calcSlopeOffaxis(jetModel, Y, regime):

    p = Y[9]

    if regime is 'D':
        alpha = 7.0
    elif regime is 'E':
        alpha = 17.0/3.0
    elif regime is 'F':
        alpha = 13.0/2.0
    elif regime is 'G':
        alpha = 15.0/2.0 - 3*p/2.0
    elif regime is 'H':
        alpha = 8.0 - 3*p/2.0

    return alpha


def calcSlopeStruct(jetModel, Y, regime, g=None, som=None):

    thV = Y[0]
    thC = Y[2]
    thW = Y[3]
    b = Y[4]
    p = Y[9]

    beta, al1, D, s1, s2, p = get_powers(regime, p)
    al2 = 3+s2

    f = get_f(jetModel, thC, thW, b)

    thS = min(0.5*thV, 0.5*thW)
    psiS = thV-thS

    dth = max(0.1*thC, 1.0e-6)
    fR = f(thS+dth)
    fL = f(thS-dth)

    dlfdth = math.log(fR/fL) / (2*dth)

    if g is None:
        g = -psiS * dlfdth
    if som is None:
        som = 1.0

    alpha = (-3*al1 + 3*som + al2*g) / (8.0 + g)

    return alpha


def calcSlopePre(jetModel, Y, regime):

    p = Y[9]

    if regime is 'D':
        alpha = 0.5
    elif regime is 'E':
        alpha = 1.0/6.0
    elif regime is 'F':
        alpha = -0.25
    elif regime is 'G':
        alpha = 3*(1.0-p)/4.0
    elif regime is 'H':
        alpha = 0.25*(2-3*p)

    return alpha


def calcSlopePost(jetModel, Y, regime):

    p = Y[9]

    if regime is 'D':
        alpha = -0.25
    elif regime is 'E':
        alpha = -7.0/12.0
    elif regime is 'F':
        alpha = -1.0
    elif regime is 'G':
        alpha = -3*p/4.0
    elif regime is 'H':
        alpha = -0.25*(3*p+1)

    return alpha


def makeAnalyticTestPlots():

    # regimes = ['D', 'E', 'F', 'G', 'H']
    regimes = ['E', 'G', 'H']
    # regimes = ['D']
    jetModel = 4
    spread = False

    # thWs = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    thWs = [0.25, 0.45, 0.65, 0.85]
    # thCs = [0.02, 0.04, 0.08, 0.12, 0.24, 0.48]
    if jetModel is 4:
        thCs = [0.02, 0.04, 0.08, 0.16]
    else:
        thCs = [0.08, 0.16, 0.24, 0.32]
    # thVs = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
    thVs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    NC = len(thCs)
    NV = len(thVs)
    NW = len(thWs)

    N = 100
    nufac = 1.2
    tfac = 1.2

    t = np.empty(N)
    nu = np.empty(N)

    alOffN = np.empty((NW, NV, NC))
    # alPreN = np.empty((NW, NV, NC))
    alStructN = np.empty((NW, NV, NC))
    alStructN2 = np.empty((NW, NV, NC))
    alStructNAveLin = np.empty((NW, NV, NC))
    alStructNAveLog = np.empty((NW, NV, NC))
    alPostN = np.empty((NW, NV, NC))
    FbN = np.empty((NW, NV, NC))

    for regime in regimes:
        NU, tN, Y, betaE = getRegimePars(regime)
        # ax.set_ylim(0, 10)

        minTW = np.inf
        maxTB = -np.inf
        nu[:] = NU

        for k, thW in enumerate(thWs):

            for j, thC in enumerate(thCs):
                for i, thV in enumerate(thVs):

                    Y[0] = thV
                    Y[2] = thC
                    Y[3] = thW

                    tW = calcTW(jetModel, Y, regime)
                    tb = calcTB(jetModel, Y, regime)
                    if tW > 0.0 and tW < minTW:
                        minTW = tW
                    if tb > maxTB:
                        maxTB = tb

            t[:] = np.geomspace(0.03*minTW, 10*maxTB, num=N)

        for k, thW in enumerate(thWs):

            # fig, ax = plt.subplots(3*NV, NC, figsize=(2*NC, 6*NV))
            panelWidth = 6.0
            panelHeight = 6.0
            fig = plt.figure(figsize=(panelWidth*NV, panelHeight*NC))
            gs0 = fig.add_gridspec(NC, NV)

            for i, thC in enumerate(thCs):
                for j, thV in enumerate(thVs):

                    print(i+1, j+1)

                    Y[0] = thV
                    Y[2] = thC
                    Y[3] = thW

                    tW = calcTW(jetModel, Y, regime)
                    tb = calcTB(jetModel, Y, regime)
                    # tbpm = calcTBpm(jetModel, Y, regime)

                    Fnu = grb.fluxDensity(t, nu, jetModel, 0, *Y,
                                          spread=spread)
                    Fnutr = grb.fluxDensity(t*tfac, nu, jetModel, 0, *Y,
                                            spread=spread)
                    Fnutl = grb.fluxDensity(t/tfac, nu, jetModel, 0, *Y,
                                            spread=spread)
                    Fnunr = grb.fluxDensity(t, nu*nufac, jetModel, 0, *Y,
                                            spread=spread)
                    Fnunl = grb.fluxDensity(t, nu/nufac, jetModel, 0, *Y,
                                            spread=spread)

                    al = np.log(Fnutr/Fnutl) / np.log(tfac*tfac)
                    al2 = np.log(Fnutr*Fnutl/(Fnu*Fnu)) / np.log(tfac)**2
                    al3 = np.zeros(al.shape)
                    al3[1:-1] = (al2[2:]-al2[:-2]) / np.log(tfac*tfac)
                    beta = np.log(Fnunr/Fnunl) / np.log(nufac*nufac)

                    if tW > 0.0:
                        t1 = tW
                        tStruct = math.sqrt(tW*tb)
                    else:
                        tStruct = max(0.01*tb, math.sqrt(minTW*tb))
                        t1 = tStruct
                    t2 = tb
                    tPost = np.power(tb*tb*tb*tN, 1.0/4.0)
                    tOff = 0.1*tW

                    nStruct = np.searchsorted(t, tStruct)
                    alStructN[k, j, i] = al[nStruct]

                    nPost = np.searchsorted(t, tPost)
                    alPostN[k, j, i] = al[nPost]
                    if tOff > 0.0:
                        nOff = np.searchsorted(t, tOff)
                        alOffN[k, j, i] = al[nOff]
                    else:
                        alOffN[k, j, i] = np.inf
                    nb = np.searchsorted(t, tb)
                    FbN[k, j, i] = Fnu[nb]

                    indS = (t > t1) & (t < t2)
                    alNlog = al[indS].mean()
                    dtS = t[indS][1:] - t[indS][:-1]
                    TS = t[indS][-1] - t[indS][0]

                    alNlin = (0.5*(al[indS][1:]+al[indS][:-1]) * dtS / TS
                              ).sum()
                    alStructNAveLin[k, j, i] = alNlin
                    alStructNAveLog[k, j, i] = alNlog

                    indStruct = np.argwhere((t > tW) & (t < tb) & (al2 > 0))
                    if len(indStruct) > 0:
                        nStruct2 = indStruct[-1]
                    else:
                        indStruct = np.argwhere((t > tW) & (t < tb)
                                                & (al3 > 0))
                        nStruct2 = indStruct[-1]
                    tStruct2 = np.sqrt(t[nStruct2]*t[nStruct2+1])
                    alStructN2[k, j, i] = 0.5*(al[nStruct2]+al[nStruct2+1])

                    gs = gs0[i, j].subgridspec(4, 1, hspace=0.0)
                    axF = fig.add_subplot(gs[0:-2, 0])
                    axa = fig.add_subplot(gs[-2, 0])
                    axb = fig.add_subplot(gs[-1, 0])

                    if tW > 0.0:
                        axF.axvline(tW, ls='--', color='grey')
                        axa.axvline(tW, ls='--', color='grey')
                        axb.axvline(tW, ls='--', color='grey')
                    if tb > 0.0:
                        axF.axvline(tb, ls='-', color='grey')
                        axa.axvline(tb, ls='-', color='grey')
                        axb.axvline(tb, ls='-', color='grey')
                    # if tbpm > 0.0:
                    #    axF.axvline(tbpm, ls=':', color='grey')
                    #    axa.axvline(tbpm, ls=':', color='grey')
                    #    axb.axvline(tbpm, ls=':', color='grey')
                    axF.axvline(tN, ls='-', lw=4, color='lightgrey')
                    axa.axvline(tN, ls='-', lw=4, color='lightgrey')
                    axb.axvline(tN, ls='-', lw=4, color='lightgrey')
                    axF.axvline(tStruct, ls=':', lw=4, color='lightgrey')
                    axa.axvline(tStruct, ls=':', lw=4, color='lightgrey')
                    axb.axvline(tStruct, ls=':', lw=4, color='lightgrey')
                    axF.axvline(tPost, ls=':', lw=4, color='lightgrey')
                    axa.axvline(tPost, ls=':', lw=4, color='lightgrey')
                    axb.axvline(tPost, ls=':', lw=4, color='lightgrey')
                    axF.axvline(tOff, ls=':', lw=4, color='lightgrey')
                    axa.axvline(tOff, ls=':', lw=4, color='lightgrey')
                    axb.axvline(tOff, ls=':', lw=4, color='lightgrey')
                    axF.axvline(tStruct2, ls='-.', lw=4, color='lightgrey')
                    axa.axvline(tStruct2, ls='-.', lw=4, color='lightgrey')
                    axb.axvline(tStruct2, ls='-.', lw=4, color='lightgrey')

                    alOff = calcSlopeOffaxis(jetModel, Y, regime)
                    alStruct = calcSlopeStruct(jetModel, Y, regime)
                    alPre = calcSlopePre(jetModel, Y, regime)
                    alPost = calcSlopePost(jetModel, Y, regime)

                    axa.axhline(alOff, color='grey', ls='--')
                    axa.axhline(alStruct, color='grey', ls='-')
                    axa.axhline(alPre, color='grey', ls='-.')
                    axa.axhline(alNlin, color='orange', ls='-')
                    axa.axhline(alNlog, color='orange', ls='--')
                    axa.axhline(alPost, color='grey', ls=':')
                    axb.axhline(betaE, color='grey')

                    axF.plot(t, Fnu)
                    axa.plot(t, al)
                    axb.plot(t, beta)

                    axF.set_xscale('log')
                    axF.set_yscale('log')
                    axa.set_xscale('log')
                    axb.set_xscale('log')

                    lim = axF.get_ylim()
                    if lim[0] < 1.0e-12*lim[1]:

                        F0 = FbN[k, j, i]
                        if F0/lim[0] < lim[1]/F0 and F0/lim[0] < 1.0e12:
                            axF.set_ylim(lim[0], 1.0e12*lim[0])
                        elif F0/lim[0] >= lim[1]/F0 and lim[1]/F0 < 1.0e12:
                            axF.set_ylim(1.0e-12*lim[1], lim[1])
                        else:
                            axF.set_ylim(1.0e-6*F0, 1.0e6*F0)

                    if j == 0:
                        axF.set_ylabel(r'$F_\nu$')
                        axa.set_ylabel(r'$\alpha$')
                        axb.set_ylabel(r'$\beta$')
                    if i == NV-1:
                        axb.set_xlabel(r'$t$')

            fig.tight_layout()
            plotname = "lcGrid_{0:s}_thW_{1:d}.png".format(regime, k)
            save(fig, plotname)
            plt.close(fig)

        filename = 'slopes_{0:s}.h5'.format(regime)
        print("Creating archive " + filename)
        fi = h5.File(filename, "w")
        fi.create_dataset("alOffN", data=alOffN)
        fi.create_dataset("alStructN", data=alStructN)
        fi.create_dataset("alStructN2", data=alStructN2)
        fi.create_dataset("alStructNAveLin", data=alStructNAveLin)
        fi.create_dataset("alStructNAveLog", data=alStructNAveLog)
        fi.create_dataset("alPostN", data=alPostN)
        fi.create_dataset("FbN", data=np.array(FbN))
        fi.create_dataset("thW", data=np.array(thWs))
        fi.create_dataset("thV", data=np.array(thVs))
        fi.create_dataset("thC", data=np.array(thCs))
        fi.create_dataset("Y", data=Y)
        fi.close()

        color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                 'tab:purple', 'tab:brown', 'tab:pink', 'tab:grey',
                 'tab:olive', 'tab:cyan']
        ls = ['-', '--', '-.', ':']

        fig, ax = plt.subplots(1, 1)
        for k, thW in enumerate(thWs):
            for i, thC in enumerate(thCs):
                ax.plot(thVs, alOffN[k, :, i], color=color[k], ls=ls[i])
        plotname = "alOff_{0:s}_thW.png".format(regime)
        save(fig, plotname)
        plt.close(fig)

        fig, ax = plt.subplots(1, 1)
        for k, thW in enumerate(thWs):
            for i, thC in enumerate(thCs):
                ax.plot(thVs, alStructN[k, :, i], color=color[k], ls=ls[i])
        plotname = "alStruct_{0:s}_thW.png".format(regime)
        save(fig, plotname)
        plt.close(fig)

        fig, ax = plt.subplots(1, 1)
        for k, thW in enumerate(thWs):
            for i, thC in enumerate(thCs):
                ax.plot(thVs, alPostN[k, :, i], color=color[k], ls=ls[i])
        plotname = "alPost_{0:s}_thW.png".format(regime)
        save(fig, plotname)
        plt.close(fig)

    return 0


def save(fig, name):

    print("Saving " + name)
    fig.savefig(name)


if __name__ == "__main__":

    # makeThVPlots()
    # makeCentralPlots(modelPlots=True)
    # makeConvergencePlots()
    # makeRegimePlots()
    # makeNormalizationPlots('E', 4)
    # makeNumAnaCompPlots('E', 4)
    # makeCharEvolPlots()
    makeAnalyticTestPlots()
