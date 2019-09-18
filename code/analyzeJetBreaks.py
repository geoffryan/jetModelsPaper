import sys
import math
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import afterglowpy as grb


def loadData(filename):
    filename = sys.argv[1]
    f = h5.File(filename, "r")
    Y = f['Y'][...]
    tj_th = f['tj_tophat'][...]
    tj_g = f['tj_gaussian'][...]
    tj_pl2 = f['tj_powerlaw2'][...]
    tj_pl6 = f['tj_powerlaw6'][...]
    try:
        thV = f['thV'][...]
        thC = f['thC'][...]
    except KeyError:
        thV = np.linspace(0.0, 1.0, 11)
        thC = np.linspace(0.04, 0.4, 11)
    f.close()

    E0 = Y[1]
    n0 = Y[8]

    tN = np.power(9*E0/(16*np.pi*grb.mp*n0*grb.c**5), 1.0/3.0)

    return tN, thV, thC, tj_th, tj_g, tj_pl2, tj_pl6


if __name__ == "__main__":

    tN, thVs, thCs, tj_th, tj_g, tj_pl2, tj_pl6 = loadData(sys.argv[1])

    tau_g = tj_g / tN
    tau_pl2 = tj_pl2 / tN
    tau_pl6 = tj_pl6 / tN

    thCpVs = thCs[:, None] + thVs[None, :]
    x = np.linspace(thCpVs.min(), thCpVs.max(), 100)

    num = math.pow(0.15, 3.0/8.0)

    chi = np.maximum(1.18*thCs[:, None], 0.53*(thCs[:, None] + thVs[None, :]))
    color = ['C{0:d}'.format(i) for i in range(10)]

    fig, ax = plt.subplots(3, 2, figsize=(8, 9))
    ax[0, 0].plot(thCs, thCs, color='grey', ls='--')
    ax[0, 0].plot(thCs, num*thCs, color='grey', ls=':')
    ax[1, 0].plot(thVs, thVs, color='grey', ls='--')
    ax[1, 0].plot(thVs, num*thVs, color='grey', ls=':')
    for j in range(len(thVs)):
        ax[0, 0].plot(thCs, chi[:, j], alpha=0.5, color=color[j % 10])
        ax[0, 0].plot(thCs, np.power(tau_g[:, j], 0.375))
        ax[0, 1].plot(thCs, chi[:, j] / (thCs+thVs[j]),
                      alpha=0.5, color=color[j % 10])
        ax[0, 1].plot(thCs, np.power(tau_g[:, j], 0.375) / (thCs+thVs[j]))
    for i in range(len(thCs)):
        ax[1, 0].plot(thVs, chi[i, :],
                      alpha=0.5, color=color[i % 10])
        ax[1, 0].plot(thVs, np.power(tau_g[i, :], 0.375))
        ax[1, 1].plot(thVs, chi[i, :] / (thCs[i]+thVs),
                      alpha=0.5, color=color[i % 10])
        ax[1, 1].plot(thVs, np.power(tau_g[i, :], 0.375) / (thCs[i]+thVs))
    ax[2, 0].plot(thCpVs.flat, np.power(tau_g.flat, 0.375), ls='', marker='.')
    ax[2, 0].plot(x, 0.55*x, ls='-', color='grey')

    nz = thVs > 0.0

    figll, axll = plt.subplots(3, 2, figsize=(8, 9))
    axll[0, 0].plot(thCs, thCs, color='grey', ls='--')
    axll[0, 0].plot(thCs, num*thCs, color='grey', ls=':')
    axll[1, 0].plot(thVs[nz], thVs[nz], color='grey', ls='--')
    axll[1, 0].plot(thVs[nz], num*thVs[nz], color='grey', ls=':')
    for j in range(len(thVs)):
        axll[0, 0].plot(thCs, chi[:, j], alpha=0.5, color=color[j % 10])
        axll[0, 0].plot(thCs, np.power(tau_g[:, j], 0.375))
        axll[0, 1].plot(thCs, chi[:, j] / (thCs+thVs[j]),
                        alpha=0.5, color=color[j % 10])
        axll[0, 1].plot(thCs, np.power(tau_g[:, j], 0.375) / (thCs+thVs[j]))
    for i in range(len(thCs)):
        axll[1, 0].plot(thVs[nz], chi[i, :][nz],
                        alpha=0.5, color=color[i % 10])
        axll[1, 0].plot(thVs[nz], np.power(tau_g[i][nz], 0.375))
        axll[1, 1].plot(thVs[nz], chi[i, :][nz] / (thCs[i]+thVs[nz]),
                        alpha=0.5, color=color[i % 10])
        axll[1, 1].plot(thVs[nz],
                        np.power(tau_g[i][nz], 0.375) / (thCs[i]+thVs[nz]))
    axll[2, 0].plot(thCpVs.flat, np.power(tau_g.flat, 0.375),
                    ls='', marker='.')
    axll[2, 0].plot(x, 0.55*x, ls='-', color='grey')

    axll[0, 0].set_xscale('log')
    axll[0, 0].set_yscale('log')
    axll[1, 0].set_xscale('log')
    axll[1, 0].set_yscale('log')
    axll[2, 0].set_xscale('log')
    axll[2, 0].set_yscale('log')
    axll[0, 1].set_xscale('log')
    axll[0, 1].set_yscale('log')
    axll[1, 1].set_xscale('log')
    axll[1, 1].set_yscale('log')

    print(((tau_g[:, 0]+tau_pl2[:, 0]+tau_pl6[:, 0])
           * np.power(thCs, -8.0/3.0)).mean() / 3.0)

    figOnA, axOnA = plt.subplots(1, 1)
    axOnA.plot(thCs, np.power(chi[:, 0], 8.0/3.0),
               ls='--', color='grey', lw=4,
               label=r'Analytic')
    axOnA.plot(thCs, tau_g[:, 0], ls='-', color='tab:green',
               label=r'Gaussian Jet')
    axOnA.plot(thCs, tau_pl2[:, 0], ls='-.', color='tab:orange',
               label=r'Power Law Jet $b=2$')
    axOnA.plot(thCs, tau_pl6[:, 0], ls=':', color='tab:red',
               label=r'Power Law Jet $b=6$')
    axOnA.text(0.05, 0.95, r'$\theta_{\mathrm{obs}} = 0.0$',
               horizontalalignment='left', verticalalignment='top',
               transform=axOnA.transAxes)
    axOnA.legend(loc='lower right')

    axOnA.set_xscale('log')
    axOnA.set_yscale('log')

    axOnA.set_xlabel(r'$\theta_{\mathrm{c}}$ (rad)')
    axOnA.set_ylabel(r'$t_b / t_{\mathrm{N}}$')

    figOnA.tight_layout()
    name = "jetbreak_OnAxis.pdf"
    print("Saving " + name)
    figOnA.savefig(name)

    figOffA, axOffA = plt.subplots(1, 1)
    for i in range(len(thCs))[::4]:
        thC = thCs[i]
        print("Off-Axis thetaC = {0:.4f}".format(thC))
        if i == 0:
            labelA = r'Analytic'
            labelG = r'Gaussian Jet'
            labelPL2 = r'Power Law Jet $b=2$'
            labelPL6 = r'Power Law Jet $b=6$'
        else:
            labelA = None
            labelG = None
            labelPL2 = None
            labelPL6 = None
        axOffA.plot(thVs, np.power(chi[i, :], 8.0/3.0),
                    ls='--', color='grey', lw=4,
                    label=labelA)
        axOffA.plot(thVs, tau_g[i, :], ls='-', color='tab:green',
                    label=labelG)
        axOffA.plot(thVs, tau_pl2[i, :], ls='-.', color='tab:orange',
                    label=labelPL2)
        axOffA.plot(thVs, tau_pl6[i, :], ls=':', color='tab:red',
                    label=labelPL6)

    axOffA.legend()

    # axOffA.set_xscale('log')
    axOffA.set_yscale('log')

    axOffA.set_xlabel(r'$\theta_{\mathrm{obs}}$ (rad)')
    axOffA.set_ylabel(r'$t_b / t_{\mathrm{N}}$')

    figOffA.tight_layout()
    name = "jetbreak_OffAxis.pdf"
    print("Saving " + name)
    figOffA.savefig(name)

    plt.show()
