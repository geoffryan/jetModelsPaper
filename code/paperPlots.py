import math
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as collections
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
    n0 = 1.0e-3
    p = 2.2
    epse = 0.1
    epsB = 0.01
    xiN = 1.0
    dL = 1.23e26

    Y = np.array([thV, E0, thC, thW, 0.0, 0.0, 0.0, n0, p, epse, epsB, xiN,
                  dL])

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

    print("Saving GaussianThV.pdf")
    fig.savefig('GaussianThV.pdf')
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

    print('Saving PowerlawThV.pdf')
    fig.savefig('PowerlawThV.pdf')
    plt.close(fig)


def makeCentralPlots(modelPlots=True):

    thC = 0.1
    thW = 0.4
    E0 = 1.0e52
    n0 = 1.0e-2
    p = 2.2
    epse = 0.1
    epsB = 0.001
    xiN = 1.0
    dL = 1.23e26

    Y = np.array([0.0, E0, thC, thW, 0.0, 0.0, 0.0, n0, p, epse, epsB, xiN,
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
        print("Saving " + figname)
        fig.savefig(figname)
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
        print("Saving " + figname)
        fig.savefig(figname)
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
        print("Saving " + figname)
        fig.savefig(figname)
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
    print("Saving " + figname)
    fig.savefig(figname, transparent=True)
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
    print("Saving " + figname)
    fig.savefig(figname)
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
    n0 = 1.0
    p = 2.2
    epse = 0.1
    epsB = 0.01
    xiN = 1.0
    dL = 1.23e26

    Y = np.array([thV, E0, thC, thW, 0.0, 0.0, 0.0, n0, p, epse, epsB, xiN,
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
    print("Saving conv_ref.pdf")
    fig.savefig("conv_ref.pdf")
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
    print("Saving conv_latRes.pdf")
    fig.savefig("conv_latRes.pdf")
    plt.close(fig)

    ax2.set_xlabel(r'$t$ (s)')
    ax2.set_ylabel('Flux')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    fig2.tight_layout()
    print("Saving conv_flux_latRes.pdf")
    fig2.savefig("conv_flux_latRes.pdf")
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
    print("Saving conv_tRes.pdf")
    fig.savefig("conv_tRes.pdf")
    plt.close(fig)

    ax2.set_xlabel(r'$t$ (s)')
    ax2.set_ylabel('Flux')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    fig2.tight_layout()
    print("Saving conv_flux_tRes.pdf")
    fig2.savefig("conv_flux_tRes.pdf")
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


def f_powerlaw(th, thC, thW):
    ath = np.atleast_1d(th)
    f = np.empty(ath.shape)
    valid = ath <= thW
    f[valid] = 1.0 / (1.0 + (ath*ath)[valid] / (thC*thC))
    f[~valid] = 0.0
    return f


def get_f(jetModel, thC, thW):
    if jetModel == -2:
        return lambda th: f_cone(th, thC, thW)
    elif jetModel == -1:
        return lambda th: f_tophat(th, thC, thW)
    elif jetModel == 0:
        return lambda th: f_gaussian(th, thC, thW)
    elif jetModel == 4:
        return lambda th: f_powerlaw(th, thC, thW)
    else:
        return None


def calc_tp(th, tN0, thV, f, regime, p):

    chim = 2*np.sin(0.5*(thV-th))
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

    dth = 1.0e-4
    thp = th+dth
    thm = th-dth
    fc = f(th)
    fp = f(thp)
    fm = f(thm)
    d2lfdth2 = np.log(fp*fm/(fc*fc)) / (dth*dth)
    thCn = 1.0/np.sqrt(np.fabs(d2lfdth2))
    dOmth = thCn
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


def makeNumAnaCompPlots():

    n0 = 1.0
    tN0 = 400 * 24 * 3600.
    E0 = 16*np.pi/9.0 * n0 * grb.mp * grb.c**5 * tN0**3

    thC = 0.1
    thW = 0.4
    thV = 0.5

    p = 2.2
    epse = 0.1
    epsB = 1.0e-3
    dL = 1.23e26
    N = 100

    Y = np.array([thV, E0, thC, thW, 0.0, 0.0, 0.0, n0, p, epse, epsB, 1.0,
                  dL])
    fig, ax = plt.subplots(3, 2, figsize=(12, 13))

    nu = np.empty(N)
    nu[:] = 1.0e14

    jetModel = 4
    regime = 'G'
    # thVs = [0.1, 0.3, 0.5, 0.7, 0.1, 0.3, 0.5, 0.7, 0.1, 0.3, 0.5, 0.7]
    # thCs = [0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2]
    # thWs = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    thVs = [0.2, 0.5, 0.8]
    thCs = [0.1, 0.1, 0.1]
    thWs = [0.4, 0.4, 0.4]

    for thV, thC, thW in zip(thVs, thCs, thWs):

        chi = np.geomspace(max(2*math.sin(0.5*(thV-thW)), 1.0e-2),
                           2*math.sin(0.5*(thV)),
                           N)
        th = thV - 2*np.arcsin(0.5*chi)

        f = get_f(jetModel, thC, thW)
        tp = calc_tp(th, tN0, thV, f, regime, p)
        Ip = calc_Ip(th, tN0, thV, f, nu, regime, n0, p, epse, epsB, dL)
        dOm = calc_dOm(th, tN0, thV, f, regime, p)

        Y[0] = thV
        Y[2] = thC
        Y[3] = thW
        Fnu = grb.fluxDensity(tp, nu, jetModel, 0, *Y, latRes=10)
        FnuA = Ip*dOm

        alA = np.log(FnuA[1:]/FnuA[:-1]) / np.log(tp[1:]/tp[:-1])
        al = np.log(Fnu[1:]/Fnu[:-1]) / np.log(tp[1:]/tp[:-1])

        chiC = np.sqrt(chi[1:]*chi[:-1])
        tpC = np.sqrt(tp[1:]*tp[:-1])
        label = r'$\theta_V =$ {0:.1f}'.format(thV)
        label += r', $\theta_C = $ {0:.1f}'.format(thC)
        label += r', $\theta_W = $ {0:.1f}'.format(thW)
        ax[0, 0].plot(chi, Fnu, label=label)
        ax[0, 0].plot(chi, FnuA)
        ax[1, 0].plot(chiC, al)
        ax[1, 0].plot(chiC, alA)
        ax[2, 0].plot(chiC, tpC)
        ax[0, 1].plot(tp, Fnu)
        ax[0, 1].plot(tp, FnuA)
        ax[1, 1].plot(tpC, al)
        ax[1, 1].plot(tpC, alA)
        ax[2, 1].plot(tpC, chiC)

    ax[0, 0].legend()
    ax[0, 0].set_xscale('log')
    ax[0, 0].set_yscale('log')
    ax[1, 0].set_xscale('log')
    ax[1, 0].set_ylim(-2.0, 2.0)
    ax[2, 0].set_xscale('log')
    ax[2, 0].set_yscale('log')
    ax[0, 1].set_xscale('log')
    ax[0, 1].set_yscale('log')
    ax[1, 1].set_xscale('log')
    ax[1, 1].set_ylim(-2.0, 2.0)
    ax[2, 1].set_xscale('log')
    ax[2, 1].set_yscale('log')
    fig.tight_layout()

    plotname = 'numanacomp.pdf'
    print('Saving ' + plotname)
    fig.savefig(plotname)
    plt.close(fig)


def makeNormalizationPlots():

    n0 = 1.0
    tN0 = 400 * 24 * 3600.
    E0 = 16*np.pi/9.0 * n0 * grb.mp * grb.c**5 * tN0**3

    thC = 0.1
    thW = 0.4
    thV = 0.5

    p = 2.2
    epse = 0.1
    epsB = 1.0e-3
    dL = 1.23e26

    N = 100

    nu = np.empty(N)
    nu[:] = 1.0e14

    Y = np.array([thV, E0, thC, thW, 0.0, 0.0, 0.0, n0, p, epse, epsB, 1.0,
                  dL])
    fig, ax = plt.subplots(3, 2, figsize=(12, 13))

    jetModel = 4
    regime = 'G'
    # thVs = [0.1, 0.3, 0.5, 0.7, 0.1, 0.3, 0.5, 0.7, 0.1, 0.3, 0.5, 0.7]
    # thCs = [0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2]
    # thWs = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    thVs = [0.1, 0.3, 0.5, 0.7, 0.1, 0.3, 0.5, 0.7]
    thCs = [0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2]
    thWs = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]

    for thV, thC, thW in zip(thVs, thCs, thWs):

        chi = np.geomspace(max(2*math.sin(0.5*(thV-thW)), 1.0e-2),
                           2*math.sin(0.5*(thV)),
                           N)
        th = thV - 2*np.arcsin(0.5*chi)

        f = get_f(jetModel, thC, thW)
        tp = calc_tp(th, tN0, thV, f, regime, p)
        Ip = calc_Ip(th, tN0, thV, f, nu, regime, n0, p, epse, epsB, dL)
        dth = 1.0e-4
        fl = f(th-dth)
        fc = f(th)
        fr = f(th+dth)

        dlfdth = np.log(fr/fl)/(2*dth)
        d2lfdth2 = np.log(fr*fl/(fc*fc))/(dth*dth)
        dth1 = np.power(np.fabs(dlfdth), -1.0)
        dth2 = np.power(np.fabs(d2lfdth2), -0.5)
        dth2 = 0.1

        Y[0] = thV
        Y[2] = thC
        Y[3] = thW
        Fnu = grb.fluxDensity(tp, nu, jetModel, 0, *Y, latRes=20)

        norm = Fnu/Ip
        som = np.log(norm[1:]/norm[:-1]) / np.log(chi[1:]/chi[:-1])
        chiC = np.sqrt(chi[1:]*chi[:-1])
        tpC = np.sqrt(tp[1:]*tp[:-1])
        label = r'$\theta_V =$ {0:.1f}'.format(thV)
        label += r', $\theta_C = $ {0:.1f}'.format(thC)
        label += r', $\theta_W = $ {0:.1f}'.format(thW)
        linelist = ax[0, 0].plot(chi, Fnu, label=label)
        ax[1, 0].plot(chi, norm)
        ax[1, 0].plot(chi, chi*dth1, color=linelist[0].get_color(), ls='--')
        ax[1, 0].plot(chi, chi*dth2, color=linelist[0].get_color(), ls=':')
        ax[2, 0].plot(chiC, som)
        ax[0, 1].plot(tp, Fnu)
        ax[1, 1].plot(tp, norm)
        ax[2, 1].plot(tpC, som)

    ax[0, 0].legend()
    ax[0, 0].set_xscale('log')
    ax[0, 0].set_yscale('log')
    ax[1, 0].set_xscale('log')
    ax[1, 0].set_yscale('log')
    ax[1, 0].set_ylim(1.0e-6, 1.0)
    ax[2, 0].set_xscale('log')
    ax[2, 0].set_ylim(-2.0, 6.0)
    ax[0, 1].set_xscale('log')
    ax[0, 1].set_yscale('log')
    ax[1, 1].set_xscale('log')
    ax[1, 1].set_yscale('log')
    ax[2, 1].set_xscale('log')
    ax[2, 1].set_ylim(-2.0, 6.0)
    fig.tight_layout()

    plotname = 'som.pdf'
    print('Saving ' + plotname)
    fig.savefig(plotname)
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

    p = 2.2
    epse_slow = 0.1
    epse_fast = 0.3
    epsB_slow = 1.0e-3
    epsB_fast = 0.3
    dL = 1.23e26

    Yslow = np.array([thV, E0_slow, thC, thW, 0.0, 0.0, 0.0, n0_slow, p,
                      epse_slow, epsB_slow, 1.0, dL])
    Yfast = np.array([thV, E0_fast, thC, thW, 0.0, 0.0, 0.0, n0_fast, p,
                      epse_fast, epsB_fast, 1.0, dL])

    nuD = 1.0e2
    nuE = 1.0
    nuF = 1.0e7
    nuG = 3.0e15
    nuH = 1.0e20

    if regime is 'D':
        return nuD, tN0, Yslow
    elif regime is 'E':
        return nuE, tN0, Yfast
    elif regime is 'F':
        return nuF, tN0, Yfast
    elif regime is 'G':
        return nuG, tN0, Yslow
    elif regime is 'H':
        return nuH, tN0, Yslow

    return None


def makeRegimePlots():

    nuD, tND, YD = getRegimePars('D')
    nuE, tNE, YE = getRegimePars('E')
    nuF, tNF, YF = getRegimePars('F')
    nuG, tNG, YG = getRegimePars('G')
    nuH, tNH, YH = getRegimePars('H')

    N = 100
    tN0 = tNG

    jetModel = 0
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
    print('Saving ' + plotname)
    figF.savefig(plotname)

    plotname = 'specSlow.pdf'
    print('Saving ' + plotname)
    figS.savefig(plotname)

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
    print('Saving ' + plotname)
    fig.savefig(plotname)


if __name__ == "__main__":

    # makeThVPlots()
    # makeCentralPlots(modelPlots=True)
    # makeConvergencePlots()
    makeRegimePlots()
    # makeNormalizationPlots()
    # makeNumAnaCompPlots()
