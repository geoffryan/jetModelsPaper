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

    chiOnCoeff = 1.18
    chiOffCoeff = 0.53

    chi = np.maximum(1.18*thCs[:, None], 0.53*(thCs[:, None] + thVs[None, :]))
    color = ['C{0:d}'.format(i) for i in range(10)]

    E53 = 1.0e53
    E50 = 1.0e50
    n00 = 1.0
    nm3 = 1.0e-3
    th0p1 = 0.1
    th0p2 = 0.2
    th0p3 = 0.3

    coeff_On_53_00_0p1_s = math.pow(9*E53/(16*math.pi*grb.mp*n00*grb.c**5),
                                    1.0/3.0) * math.pow(chiOnCoeff * th0p1,
                                                        8.0/3.0)
    coeff_On_53_00_0p1_d = coeff_On_53_00_0p1_s * grb.sec2day
    coeff_Off_53_00_0p1_s = math.pow(9*E53/(16*math.pi*grb.mp*n00*grb.c**5),
                                     1.0/3.0) * math.pow(chiOffCoeff * th0p1,
                                                         8.0/3.0)
    coeff_Off_53_00_0p1_d = coeff_Off_53_00_0p1_s * grb.sec2day
    coeff_Off_53_00_0p2_s = math.pow(9*E53/(16*math.pi*grb.mp*n00*grb.c**5),
                                     1.0/3.0) * math.pow(chiOffCoeff * th0p2,
                                                         8.0/3.0)
    coeff_Off_53_00_0p2_d = coeff_Off_53_00_0p2_s * grb.sec2day
    coeff_Off_53_00_0p3_s = math.pow(9*E53/(16*math.pi*grb.mp*n00*grb.c**5),
                                     1.0/3.0) * math.pow(chiOffCoeff * th0p3,
                                                         8.0/3.0)
    coeff_Off_53_00_0p3_d = coeff_Off_53_00_0p3_s * grb.sec2day

    coeff_On_50_m3_0p1_s = math.pow(9*E50/(16*math.pi*grb.mp*nm3*grb.c**5),
                                    1.0/3.0) * math.pow(chiOnCoeff * th0p1,
                                                        8.0/3.0)
    coeff_On_50_m3_0p1_d = coeff_On_50_m3_0p1_s * grb.sec2day
    coeff_Off_50_m3_0p1_s = math.pow(9*E50/(16*math.pi*grb.mp*nm3*grb.c**5),
                                     1.0/3.0) * math.pow(chiOffCoeff * th0p1,
                                                         8.0/3.0)
    coeff_Off_50_m3_0p1_d = coeff_Off_50_m3_0p1_s * grb.sec2day
    coeff_Off_50_m3_0p2_s = math.pow(9*E50/(16*math.pi*grb.mp*nm3*grb.c**5),
                                     1.0/3.0) * math.pow(chiOffCoeff * th0p2,
                                                         8.0/3.0)
    coeff_Off_50_m3_0p2_d = coeff_Off_50_m3_0p2_s * grb.sec2day
    coeff_Off_50_m3_0p3_s = math.pow(9*E50/(16*math.pi*grb.mp*nm3*grb.c**5),
                                     1.0/3.0) * math.pow(chiOffCoeff * th0p3,
                                                         8.0/3.0)
    coeff_Off_50_m3_0p3_d = coeff_Off_50_m3_0p3_s * grb.sec2day

    print("tj_On_E53_n00_th0p1 = {0:.4e} s = {1:.4e} days".format(
          coeff_On_53_00_0p1_s, coeff_On_53_00_0p1_d))
    print("tj_Off_E53_n00_th0p1 = {0:.4e} s = {1:.4e} days".format(
          coeff_Off_53_00_0p1_s, coeff_Off_53_00_0p1_d))
    print("tj_Off_E53_n00_th0p2 = {0:.4e} s = {1:.4e} days".format(
          coeff_Off_53_00_0p2_s, coeff_Off_53_00_0p2_d))
    print("tj_Off_E53_n00_th0p3 = {0:.4e} s = {1:.4e} days".format(
          coeff_Off_53_00_0p3_s, coeff_Off_53_00_0p3_d))
    print("tj_On_E50_nm3_th0p1 = {0:.4e} s = {1:.4e} days".format(
          coeff_On_50_m3_0p1_s, coeff_On_50_m3_0p1_d))
    print("tj_Off_E50_nm3_th0p1 = {0:.4e} s = {1:.4e} days".format(
          coeff_Off_50_m3_0p1_s, coeff_Off_50_m3_0p1_d))
    print("tj_Off_E50_nm3_th0p2 = {0:.4e} s = {1:.4e} days".format(
          coeff_Off_50_m3_0p2_s, coeff_Off_50_m3_0p2_d))
    print("tj_Off_E50_nm3_th0p3 = {0:.4e} s = {1:.4e} days".format(
          coeff_Off_50_m3_0p3_s, coeff_Off_50_m3_0p3_d))

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

    tau_theo = np.power(chi, 8.0/3.0)
    tb_lim_min = 1.0e-4
    tb_lim_max = 1.0

    figOnA, axOnA = plt.subplots(1, 1)
    axOnA.plot(thCs, tau_theo[:, 0],
               ls='--', color='grey', alpha=0.8, lw=4,
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
    axOnA.set_ylabel(r'$t_{\mathrm{b}} / t_{\mathrm{NR}}$')

    axOnA.set_xticks([0.04, 0.1, 0.4])
    axOnA.set_xticklabels([r'$4\times 10^{-2}$', r'$10^{-1}$',
                           r'$4\times 10^{-1}$'])

    axOnA.set_ylim(tb_lim_min, tb_lim_max)

    figOnA.tight_layout()
    name = "jetbreak_OnAxis.pdf"
    print("Saving " + name)
    figOnA.savefig(name)

    textProps = {'horizontalalignment': 'left',
                 'verticalalignment': 'bottom',
                 'fontsize': 8}

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

        axOffA.plot(thVs, tau_theo[i, :],
                    ls='--', color='grey', alpha=0.8, lw=4,
                    label=labelA)
        axOffA.plot(thVs, tau_g[i, :], ls='-', color='tab:green',
                    label=labelG)
        axOffA.plot(thVs, tau_pl2[i, :], ls='-.', color='tab:orange',
                    label=labelPL2)
        axOffA.plot(thVs, tau_pl6[i, :], ls=':', color='tab:red',
                    label=labelPL6)

        xloc = -0.04
        yloc = tau_theo[i, 0]

        if i == 0:
            # xloc = 0.07
            yloc /= 1.2
            textProps['horizontalalignment'] = 'left'
            textProps['verticalalignment'] = 'top'
        elif i == 4:
            yloc /= 1.4
            textProps['horizontalalignment'] = 'left'
            textProps['verticalalignment'] = 'top'
        elif i == 8:
            yloc *= 1.1
            textProps['horizontalalignment'] = 'left'
            textProps['verticalalignment'] = 'bottom'

        axOffA.text(xloc, yloc,
                    r"$\theta_{{\mathrm{{c}}}} = {0:.2f}$ rad".format(thC),
                    **textProps)

    fracDiff_g = np.fabs((tau_g - tau_theo) / tau_theo)
    fracDiff_pl2 = np.fabs((tau_pl2 - tau_theo) / tau_theo)
    fracDiff_pl6 = np.fabs((tau_pl6 - tau_theo) / tau_theo)

    print("Gaussian frac diff(min/mean/max): OnAxis  {0:.2e}/{1:.2e}/{2:.2e}"
          .format(fracDiff_g[:, 0].min(), fracDiff_g[:, 0].mean(),
                  fracDiff_g[:, 0].max()))
    print("                                  OffAxis {0:.2e}/{1:.2e}/{2:.2e}"
          .format(fracDiff_g.min(), fracDiff_g.mean(), fracDiff_g.max()))

    print("PowerLaw2 frac diff(min/mean/max): OnAxis  {0:.2e}/{1:.2e}/{2:.2e}"
          .format(fracDiff_pl2[:, 0].min(), fracDiff_pl2[:, 0].mean(),
                  fracDiff_pl2[:, 0].max()))
    print("                                   OffAxis {0:.2e}/{1:.2e}/{2:.2e}"
          .format(fracDiff_pl2.min(), fracDiff_pl2.mean(), fracDiff_pl2.max()))

    print("PowerLaw6 frac diff(min/mean/max): OnAxis  {0:.2e}/{1:.2e}/{2:.2e}"
          .format(fracDiff_pl6[:, 0].min(), fracDiff_pl6[:, 0].mean(),
                  fracDiff_pl6[:, 0].max()))
    print("                                   OffAxis {0:.2e}/{1:.2e}/{2:.2e}"
          .format(fracDiff_pl6.min(), fracDiff_pl6.mean(), fracDiff_pl6.max()))

    print("AllModels frac diff(min/mean/max): OnAxis  {0:.2e}/{1:.2e}/{2:.2e}"
          .format(min(fracDiff_g[:, 0].min(), fracDiff_pl2[:, 0].min(),
                      fracDiff_pl6[:, 0].min()),
                  (fracDiff_g[:, 0].mean() + fracDiff_pl2[:, 0].mean()
                   + fracDiff_pl6[:, 0].mean())/3.0,
                  max(fracDiff_g[:, 0].max(), fracDiff_pl2[:, 0].max(),
                      fracDiff_pl6[:, 0].max())))
    print("                                   OffAxis {0:.2e}/{1:.2e}/{2:.2e}"
          .format(min(fracDiff_g.min(), fracDiff_pl2.min(),
                      fracDiff_pl6.min()),
                  (fracDiff_g.mean() + fracDiff_pl2.mean()
                   + fracDiff_pl6.mean())/3.0,
                  max(fracDiff_g.max(), fracDiff_pl2.max(),
                      fracDiff_pl6.max())))

    # axOffA.legend()

    # axOffA.set_xscale('log')
    axOffA.set_yscale('log')

    axOffA.set_xlabel(r'$\theta_{\mathrm{obs}}$ (rad)')
    axOffA.set_ylabel(r'$t_{\mathrm{b}} / t_{\mathrm{NR}}$')

    axOffA.set_ylim(tb_lim_min, tb_lim_max)

    figOffA.tight_layout()
    name = "jetbreak_OffAxis.pdf"
    print("Saving " + name)
    figOffA.savefig(name)

    # plt.show()
