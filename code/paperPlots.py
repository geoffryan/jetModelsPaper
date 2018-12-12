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


if __name__ == "__main__":

    # makeThVPlots()
    makeCentralPlots(modelPlots=True)
