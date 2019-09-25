import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import afterglowpy as grb
import h5py as h5
import corner


def E_Gaussian(th, thC, thW, b):
    E = np.exp(-0.5*th*th/(thC*thC))
    E[th > thW] = 0.0
    return E


def E_powerlaw1(th, thC, thW, b):
    TH = np.sqrt(1 + th*th/(thC*thC))
    E = np.power(TH, -b)
    E[th > thW] = 0.0
    return E


def E_powerlaw2(th, thC, thW, b):
    TH = np.sqrt(1 + th*th/(b*thC*thC))
    E = np.power(TH, -b)
    E[th > thW] = 0.0
    return E


def weightNull(flatchain):

    return np.ones(flatchain.shape[0])


def weightPlanck(flatchain):

    thv = flatchain[:, 0]
    cthv = np.cos(thv)

    # From Fitting Figure 3 of LIGO Standard Siren Paper
    cv0 = 0.985
    sig = 0.070

    x = (cthv-cv0)/sig
    w = np.exp(-0.5*x*x)

    return w


def weightSHoES(flatchain):

    thv = flatchain[:, 0]
    cthv = np.cos(thv)

    cv0 = 0.909
    sig = 0.068

    x = (cthv-cv0)/sig
    w = np.exp(-0.5*x*x)

    return w


def weightLIGO(flatchain):

    thv = flatchain[:, 0]
    cthv = np.cos(thv)

    cv0 = 0.958
    sig = 0.266

    x = (cthv-cv0)/sig
    w = np.exp(-0.5*x*x)

    return w


def percentile_1d(a, q, w=None):

    if w is None:
        return np.percentile(a, q)

    return corner.quantile(a, np.array(q)/100.0, w)


def percentile_2d(a, q, w=None, axis=None):

    if w is None:
        return np.percentile(a, q, axis=axis)

    if axis is None:
        return corner.quantile(a.flat, np.array(q)/100.0, w)

    if axis == 0:
        ret = np.empty((len(q), a.shape[1]))
        for i in range(a.shape[1]):
            ret[:, i] = corner.quantile(a[:, i], np.array(q)/100.0, w)
    elif axis == 1:
        ret = np.empty((len(q), a.shape[0]))
        for i in range(a.shape[0]):
            ret[:, i] = corner.quantile(a[i, :], np.array(q)/100.0, w)
    else:
        raise ValueError("axis must be 0 or 1 for 2d array")

    return ret


def getParsFromSamples(filename, thin=1, f_weight=None):

    print("Loading file: " + filename)

    f = h5.File(filename, "r")
    steps_taken = f['steps_taken'][0]
    chain = f['chain'][...][:, thin-1:steps_taken:thin, :]
    # lnprobability = f['lnprobability'][...][:, :steps_taken:thin]
    X0 = f['X0'][...]
    jetType = f['jetType'][0]
    fitVars = f['fitVars'][...]
    # labels = np.array(f['labels'][...], dtype='U32')
    try:
        Z = {}
        Zg = f['Z']
        for key in Zg:
            Z[key] = Zg[key][0]
    except KeyError:
        Z = {}
    f.close()

    # nwalkers = chain.shape[0]
    nsteps = chain.shape[1]
    ndim = chain.shape[2]
    nburn = nsteps//4  # 36000/thin #nsteps/4

    chainBurned = chain[:, nburn:, :]
    # lnprobabilityBurned = lnprobability[:, nburn:]

    flatchainBurned = chainBurned.reshape((-1, ndim))
    # flatlnprobabilityBurned = lnprobabilityBurned.reshape((-1,))

    N = flatchainBurned.shape[0]

    print("Steps considered: {0:d}".format(chainBurned.shape[1]))
    print("Total draws: {0:d}".format(N))

    thV = np.empty(N)
    lE0 = np.empty(N)
    thC = np.empty(N)
    thW = np.empty(N)
    b = np.empty(N)

    if 0 in fitVars:
        i = np.where(fitVars == 0)[0][0]
        thV[:] = flatchainBurned[:, i]
    else:
        thV[:] = X0[0]
    if 1 in fitVars:
        i = np.where(fitVars == 1)[0][0]
        lE0[:] = flatchainBurned[:, i]
    else:
        lE0[:] = X0[1]
    if 2 in fitVars:
        i = np.where(fitVars == 2)[0][0]
        thC[:] = flatchainBurned[:, i]
    else:
        thC[:] = X0[2]
    if 3 in fitVars:
        i = np.where(fitVars == 3)[0][0]
        thW[:] = flatchainBurned[:, i]
    else:
        thW[:] = X0[3]
    if 4 in fitVars:
        i = np.where(fitVars == 4)[0][0]
        b[:] = flatchainBurned[:, i]
    else:
        b[:] = X0[4]

    if f_weight is None:
        w = None
    else:
        w = f_weight(flatchainBurned)

    return jetType, thV, lE0, thC, thW, b, w


def plotModelFromSamples(ax, thd, thr, filename, thin, plotKwargs, scheme=0,
                         f_weight=None):

    jetType, thV, lE0, thC, thW, b, w = getParsFromSamples(filename, thin,
                                                           f_weight)

    if jetType == 4:
        label = "Power Law"
    elif jetType == 0:
        label = "Gaussian"
    else:
        label = "Top Hat"

    E0 = np.power(10.0, lE0)

    if jetType == 4:
        f_E = E_powerlaw2
    else:
        f_E = E_Gaussian

    if scheme == 1:
        N = len(thV)
        th_c = 0.5*(thr[:-1] + thr[1:])
        Nth = len(th_c)
        E_edges = np.geomspace(1.0e46, 1.0e56, 201)
        Ec = np.zeros((Nth, len(E_edges)-1))
        ith = np.arange(Nth)

        for i in range(N):
            E = E0[i] * f_E(th_c, thC[i], thW[i], b[i])
            dig = np.digitize(E, E_edges)
            valid = (dig > 0) & (dig < len(E_edges))
            if w is None:
                Ec[ith[valid], dig[valid]] += 1
            else:
                Ec[ith[valid], dig[valid]] += w[i]

        pcol = ax.pcolormesh(thd, E_edges, Ec.T, **plotKwargs)
        pcol.set_edgecolor('face')
    elif scheme == 2:
        N = len(thV)
        Nth = len(thr)
        E = np.empty((Nth, N))

        for i in range(N):
            E[:, i] = E0[i] * f_E(thr, thC[i], thW[i], b[i])

        quantiles = percentile_2d(E, [2.5, 16, 50, 84, 97.5], w,
                                  axis=1)
        # thVquant = percentile_1d(180/np.pi*thV, [2.5, 16, 50, 84, 97.5], w)

        alphaFac = plotKwargs.pop('alphaFactor')

        ax.plot(thd, quantiles[2], alpha=1.0, lw=2, label=label, **plotKwargs)
        ls = plotKwargs.pop('ls')
        ax.fill_between(thd, quantiles[1], quantiles[3], alpha=0.7*alphaFac,
                        **plotKwargs)
        ax.fill_between(thd, quantiles[0], quantiles[4], alpha=0.4*alphaFac,
                        **plotKwargs)

        # ax.fill_betweenx(np.array([1.0e46, 1.0e56]),
        #                  thVquant[1], thVquant[3],
        #                  alpha=0.7*alphaFac, **plotKwargs)
        # ax.fill_betweenx(np.array([1.0e46, 1.0e56]),
        #                  thVquant[0], thVquant[4],
        #                  alpha=0.4*alphaFac, **plotKwargs)
        plotKwargs['ls'] = ls
    elif scheme == 3:
        N = len(thV)
        Nth = len(thr)
        E = np.empty((Nth, N))

        for i in range(N):
            E[:, i] = E0[i] * f_E(thr, thC[i], thW[i], b[i])

        cut = [0, 25, 50, 75, 100]

        # quantiles = percentile_1d(thV, [0, 20, 40, 60, 80, 100], w)
        quantiles = percentile_1d(thV, cut, w)
        # quantiles = percentile_1d(thV, [2.5, 16, 50, 84, 97.5], w)
        # quantiles = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

        print("thV quantiles (rad)")
        print(quantiles)
        print("thV quantiles (deg)")
        print(np.array(quantiles) * 180/np.pi)

        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
        alphaFac = 1.0

        for i in range(len(quantiles)-1):
            inds = (thV >= quantiles[i]) & (thV <= quantiles[i+1])

            bandLabel = (r'$\theta_{{\mathrm{{obs}}, {0:d}}}'
                         r'< \theta_{{\mathrm{{obs}}}}  < '
                         r'\theta_{{\mathrm{{obs}}, {1:d}}}$'
                         ).format(cut[i], cut[i+1])

            Equantiles = percentile_2d(E[:, inds], [2.5, 16, 50, 84, 97.5],
                                       w[inds], axis=1)
            # ax.fill_between(thd, E[:, inds].min(axis=1),
            #                 E[:, inds].max(axis=1), color=colors[i],
            #                 **plotKwargs)
            alpha = plotKwargs.pop('alpha')
            ax.fill_between(thd, Equantiles[0], Equantiles[4],
                            color=colors[3-i], alpha=alphaFac*alpha,
                            label=bandLabel, lw=0,
                            **plotKwargs)
            # ax.fill_between(thd, Equantiles[1], Equantiles[3],
            #                 color=colors[i], alpha=alpha,
            #                 **plotKwargs)
            plotKwargs['alpha'] = alpha

            # ax.fill_betweenx(np.array([1.0e46, 1.0e56]),
            #                 180/np.pi*quantiles[i], 180/np.pi*quantiles[i+1],
            #                  color=colors[i],
            #                  **plotKwargs)
    elif scheme == 4:
        N = len(thV)
        Nth = len(thr)
        E = np.empty((Nth, N))

        for i in range(N):
            E[:, i] = E0[i] * f_E(thr, thC[i], thW[i], b[i])

        lE = np.empty(E.shape)
        lE[E > 0] = np.log(E[E > 0])
        lE[E <= 0] = -np.inf

        th_c = 0.5*(thr[1:] + thr[:-1])
        thd_c = 0.5*(thd[1:] + thd[:-1])
        dlEdth = (lE[1:, :]-lE[:-1, :]) / (thr[1:] - thr[:-1])[:, None]
        dlEdth[~np.isfinite(dlEdth)] = 0.0

        dlEdth *= -(thV[None, :] - th_c[:, None])

        quantiles = percentile_1d(thV, [0, 20, 40, 60, 80, 100], w)
        quantiles = percentile_1d(thV, [2.5, 16, 50, 84, 97.5], w)

        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']

        for i in range(len(quantiles)-1):
            inds = (thV >= quantiles[i]) & (thV <= quantiles[i+1])
            ax.fill_between(thd_c, dlEdth[:, inds].min(axis=1),
                            dlEdth[:, inds].max(axis=1), color=colors[i],
                            **plotKwargs)
    else:
        for i in range(thV.shape[0]):
            E = E0[i] * f_E(thr, thC[i], thW[i], b[i])
            ax.plot(thd, E, **plotKwargs)


def makeModelSamplesPlot(filenames, thin, scheme=0, f_weight=None):

    alpha = 0.03
    colors = ['C0', 'C1', 'C2', 'C3']
    cmaps = [mpl.cm.Reds, mpl.cm.Blues, mpl.cm.Greens, mpl.cm.Greys]
    ls = ['-', '--', '.-', ':']

    thd = np.linspace(0.0, 30.0, 300)
    thr = thd * np.pi/180.0

    fig, ax = plt.subplots(1, 1)

    if scheme == 1:
        for i, filename in enumerate(filenames):
            fig, axi = plt.subplots(1, 1, figsize=(9, 6))

            kwargs = {'cmap': cmaps[i], 'alpha': 0.5}
            kwargsi = {'cmap': mpl.cm.viridis, 'alpha': 1.0}

            plotModelFromSamples(ax, thd, thr, filename, thin, kwargs,
                                 scheme, f_weight)
            plotModelFromSamples(axi, thd, thr, filename, thin, kwargsi,
                                 scheme, f_weight)

            axi.set_xscale('linear')
            axi.set_yscale('log')
            axi.set_xlim(th.min(), th.max())
            axi.set_ylim(1.0e-4, 1.0e2)
    elif scheme == 2:
        for i, filename in enumerate(filenames):
            kwargs = {'color': colors[i], 'alphaFactor': 0.5, 'ls': ls[i]}
            plotModelFromSamples(ax, thd, thr, filename, thin, kwargs, scheme,
                                 f_weight)

            figi, axi = plt.subplots(1, 1, figsize=(9, 6))
            kwargsi = {'cmap': mpl.cm.Greys, 'alpha': 1.0}
            kwargs = {'color': colors[i], 'alphaFactor': 0.5, 'ls': ls[i]}
            thd2 = np.linspace(0.0, 30.0, 201)
            thr2 = thd2 * np.pi/180.0
            plotModelFromSamples(axi, thd2, thr2, filename, thin, kwargsi, 1,
                                 f_weight)
            plotModelFromSamples(axi, thd, thr, filename, thin, kwargs, scheme,
                                 f_weight)

            axi.set_xscale('linear')
            axi.set_yscale('log')
            axi.set_xlim(thd.min(), thd.max())
            axi.set_ylim(1.0e49, 1.0e55)
            axi.set_xlabel(r'$\theta$ (degrees)')
            axi.set_ylabel(r'$E(\theta)$ (erg)')
            axi.legend()

            figname = "fit_E_theta_model_{0:d}.pdf".format(i)
            print("Saving " + figname)
            figi.savefig(figname)
    elif scheme == 3:
        kwargs = {'alpha': 0.5}

        for i, filename in enumerate(filenames):
            plotModelFromSamples(ax, thd, thr, filename, thin, kwargs, scheme,
                                 f_weight)

            figi, axi = plt.subplots(1, 1, figsize=(3.5, 2.5))
            plotModelFromSamples(axi, thd, thr, filename, thin, kwargs, scheme,
                                 f_weight)

            axi.legend()
            axi.set_xscale('linear')
            axi.set_yscale('log')
            axi.set_xlim(thd.min(), thd.max())
            axi.set_ylim(2.0e49, 5.0e55)
            axi.set_xlabel(r'$\theta$ (degrees)')
            axi.set_ylabel(r'$E_{\mathrm{iso}}(\theta)$ (erg)')

            figi.tight_layout()
            figname = "fit_E_theta_model_{0:d}.pdf".format(i)
            print("Saving " + figname)
            figi.savefig(figname)
    elif scheme == 4:
        kwargs = {'alpha': 0.5}

        for i, filename in enumerate(filenames):
            plotModelFromSamples(ax, thd, thr, filename, thin, kwargs, scheme,
                                 f_weight)

            figi, axi = plt.subplots(1, 1, figsize=(9, 6))
            plotModelFromSamples(axi, thd, thr, filename, thin, kwargs, scheme,
                                 f_weight)

            axi.set_xscale('linear')
            # axi.set_yscale('log')
            axi.set_xlim(th.min(), th.max())
            # axi.set_ylim(1.0e48, 1.0e54)
    else:
        for i, filename in enumerate(filenames):
            kwargs = {'color': colors[i], 'alpha': alpha}
            plotModelFromSamples(ax, thd, thr, filename, thin, kwargs, 0,
                                 f_weight)

    ax.set_xscale('linear')
    ax.set_xlim(thd.min(), thd.max())
    if scheme != 4:
        ax.set_yscale('log')
        ax.set_ylim(1.0e49, 1.0e55)

    ax.set_xlabel(r'$\theta$ (degrees)')
    ax.set_ylabel(r'$E(\theta)$ (erg)')
    ax.legend()
    fig.tight_layout()

    figname = "fit_E_theta.pdf"
    print("Saving " + figname)
    fig.savefig(figname)


thMax = 0.6

th = np.linspace(0, 0.5*np.pi, 300)

EG_1 = E_Gaussian(th, 0.1, thMax, 0)
EP1_2 = E_powerlaw1(th, 0.1, 0.2, 2)
EP1_4 = E_powerlaw1(th, 0.1, 0.2, 4)
EP1_6 = E_powerlaw1(th, 0.1, 0.2, 6)
EP1_8 = E_powerlaw1(th, 0.1, 0.2, 8)
EP2_2 = E_powerlaw2(th, 0.1, 0.2, 2)
EP2_4 = E_powerlaw2(th, 0.1, 0.2, 4)
EP2_6 = E_powerlaw2(th, 0.1, 0.2, 6)
EP2_8 = E_powerlaw2(th, 0.1, 0.2, 8)

fig, ax = plt.subplots(1, 1, figsize=(12, 9))

ax.plot(th, EG_1, color='C0')
ax.plot(th, EP1_2, color='C1')
ax.plot(th, EP1_4, color='C2')
ax.plot(th, EP1_6, color='C3')
ax.plot(th, EP1_8, color='C4')
ax.plot(th, EP2_2, color='C1', alpha=0.5)
ax.plot(th, EP2_4, color='C2', alpha=0.5)
ax.plot(th, EP2_6, color='C3', alpha=0.5)
ax.plot(th, EP2_8, color='C4', alpha=0.5)

ax.set_yscale('log')

ax.set_xlim(0.0, thMax)
ax.set_ylim(1.0e-4, 1.0)

ax.set_xlabel(r'$\theta$ (radians)')
ax.set_ylabel(r'$E_{\mathrm{iso}}$ (arbitrary units)')


fig2, ax2 = plt.subplots(2, 2, figsize=(12, 9))
Y = np.array([0.5, 1.0e53, 0.1, 0.4, 6.0, 0.0, 0.0, 0.0, 1.0e-3, 2.2, 0.1,
              0.01, 1.0, 1.0e26])

t = np.geomspace(1.0e4, 1.0e10, 150)
nu = np.empty(t.shape)
nu[:] = 1.0e14

thC0 = 0.1
bs = [2, 4, 6, 8]
thVs = [0.0, 0.2, 0.4, 0.6]
cs = ['C1', 'C2', 'C3', 'C4']
ls = ['-', '--', '-.', ':']

spread = True

for j, thV in enumerate(thVs):
    Y[0] = thV

    Y[2] = thC0
    Fnu = grb.fluxDensity(t, nu, -1, 0, *Y, spread=spread)
    ax2.flat[j].plot(t, Fnu, color='k', ls=ls[j])
    Y[2] = thC0 / np.sqrt(1.0)
    Fnu = grb.fluxDensity(t, nu, 0, 0, *Y, spread=spread)
    ax2.flat[j].plot(t, Fnu, color='C0', ls=ls[j])

    for i, b in enumerate(bs):
        Y[2] = thC0
        Y[4] = b
        # Fnu = grb.fluxDensity(t, nu, 4, 0, *Y, spread=False)
        # ax2.plot(t, Fnu, color=cs[i])
        Y[2] = thC0 * np.sqrt(b/2)
        # Y[2] = thC0 * np.sqrt(b)
        # if b == 2:
        #    Y[2] = thC0 / np.sqrt(2.0)
        Y[4] = b
        Fnu = grb.fluxDensity(t, nu, 4, 0, *Y, spread=spread)
        ax2.flat[j].plot(t, Fnu, color=cs[i], ls=ls[j], alpha=0.5)

    ax2.flat[j].set_xscale('log')
    ax2.flat[j].set_yscale('log')

    ax2.flat[j].set_ylim(1.0e-7, 1.0e3)

figname = "tooManyModels.pdf"
print("Saving " + figname)

fig2.savefig(figname)

makeModelSamplesPlot(sys.argv[1:], 100, 3, weightPlanck)

plt.show()
