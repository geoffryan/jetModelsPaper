import math
import numpy as np
import matplotlib.pyplot as plt
import grbpy as grb
import paperPlots as pp


def analyzeBreakTimes(regime, jetModel):

    NU, tN, Y, betaE = pp.getRegimePars(regime)

    tfac = 1.1
    nufac = 1.1

    if jetModel is 4:
        bs = np.array([2.0, 3, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    else:
        bs = np.array([1.0])
    thCs = np.array([0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16])
    thW = 0.4

    Nb = len(bs)
    Nc = len(thCs)

    panelWidth = 6.0
    panelHeight = 6.0
    fig = plt.figure(figsize=(panelWidth*Nb, panelHeight*Nc))
    gs0 = fig.add_gridspec(Nc, Nb)

    # minTb = math.pow(min(thCs), 8.0/3.0)

    t = np.geomspace(1.0e-6*tN, 0.03*tN, 100)
    nu = np.empty(t.shape)
    nu[:] = NU

    print("Calculating")

    tb = np.empty((Nc, Nb))

    for i, thC in enumerate(thCs):
        for j, b in enumerate(bs):

            Y[0] = 0.0
            Y[2] = thC
            Y[3] = thW
            Y[4] = b

            Fnu = grb.fluxDensity(t, nu, jetModel, 0, *Y, latRes=10)
            Fnutr = grb.fluxDensity(t*tfac, nu, jetModel, 0, *Y, latRes=10)
            Fnutl = grb.fluxDensity(t/tfac, nu, jetModel, 0, *Y, latRes=10)
            Fnunr = grb.fluxDensity(t, nu*nufac, jetModel, 0, *Y, latRes=10)
            Fnunl = grb.fluxDensity(t, nu/nufac, jetModel, 0, *Y, latRes=10)

            alpha = np.log(Fnutr/Fnutl) / np.log(tfac*tfac)
            beta = np.log(Fnunr/Fnunl) / np.log(nufac*nufac)

            alPre = pp.calcSlopePre(jetModel, Y, regime)
            alPost = pp.calcSlopePost(jetModel, Y, regime)

            # ib = np.searchsorted(alpha, 0.5*(alPre+alPost))
            ib = np.argwhere(alpha < 0.5*(alPre+alPost))[0]
            tb[i, j] = t[ib]

            gs = gs0[i, j].subgridspec(4, 1, hspace=0)
            axF = fig.add_subplot(gs[0:-2, 0])
            axa = fig.add_subplot(gs[-2, 0])
            axb = fig.add_subplot(gs[-1, 0])

            axF.plot(t, Fnu)
            axa.plot(t, alpha)
            axb.plot(t, beta)

            axa.axhline(alPre, color='grey', ls='--')
            axa.axhline(alPost, color='grey', ls='-')

            axF.set_xscale('log')
            axa.set_xscale('log')
            axb.set_xscale('log')
            axF.set_yscale('log')

    plotname = "jb_lc_jet_{0:d}_regime_{1:s}.png".format(jetModel, regime)
    print("Saving " + plotname)
    fig.savefig(plotname)

    p = Y[9]
    betaE2, al1, D, s1, s2, p = pp.get_powers(regime, p)

    numfac = math.pow(math.pow(2.0, 8+3*betaE) / math.sqrt(4*D), -1.0/(2*D))
    tbTH = 0.5 * np.power(thCs * numfac, 8.0/3.0) * tN

    fig, ax = plt.subplots(1, 1)
    ax.plot(thCs, tbTH, ls='--', lw=3, color='grey')
    for j, b in enumerate(bs):
        ax.plot(thCs, tb[:, j], marker='.')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plotname = "jb_tb_jet_{0:d}_regime_{1:s}.png".format(jetModel, regime)
    print("Saving " + plotname)
    fig.savefig(plotname)

    if len(bs) > 1.0:
        fig, ax = plt.subplots(1, 1)
        ax.plot(bs, np.power(bs/2, -1.5), ls='--', lw=3, color='grey')
        ax.plot(bs, np.power(bs/2, -1.65), ls='--', lw=3, color='grey')
        ax.plot(bs, np.power(bs/2, -2), ls='--', lw=3, color='grey')
        for i, thC in enumerate(thCs):
            ax.plot(bs, tb[i, :] / tbTH[i], marker='.')
        ax.set_xscale('log')
        ax.set_yscale('log')
        plotname = "jb_corr_jet_{0:d}_regime_{1:s}.png".format(
            jetModel, regime)
        print("Saving " + plotname)
        fig.savefig(plotname)

        res = np.polyfit(np.log(bs)[1:],
                         np.log(tb[:, :]/tbTH[:, None]).mean(axis=0)[1:],
                         deg=1)
        print(res)


if __name__ == "__main__":

    analyzeBreakTimes('D', -1)
    analyzeBreakTimes('E', -1)
    analyzeBreakTimes('F', -1)
    analyzeBreakTimes('G', -1)
    analyzeBreakTimes('H', -1)
    analyzeBreakTimes('H', 0)
    analyzeBreakTimes('H', 4)
