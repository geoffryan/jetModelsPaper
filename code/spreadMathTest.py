import math
import numpy as np
import matplotlib.pyplot as plt
import grbpy as grb


def f_tophat(th, **kwargs):
    return np.ones(th.shape)


def f_Gaussian(th, **kwargs):
    thC = kwargs['thC']
    return np.exp(-0.5 * th*th/(thC*thC))


def f_powerlaw(th, **kwargs):
    thC = kwargs['thC']
    b = kwargs['b']
    return np.power(1 + th*th/(b*thC*thC), -0.5*b)


def frexp10(x):

    exp = int(math.floor(math.log10(abs(x))))
    man = x / 10**exp
    return man, exp


def get_t(th, f, f_kwargs, ta, t0):

    Ith = np.zeros(th.shape)

    thC = 0.5*(th[:-1]+th[1:])
    dth = np.fabs(th[1:]-th[:-1])

    dIdth = np.sqrt(f(thC, **f_kwargs))

    Ith[1:] = np.cumsum(dIdth*dth)

    t = t0 * np.power(3*math.sqrt(2) * Ith + math.pow(ta/t0, 1.5), 2.0/3.0)

    return t


def make_travelTimes_plot():

    thC = 0.08
    thW = 0.4
    b = 5
    t0 = 1.0

    kw_tophat = {'thC': thC}
    kw_Gaussian = {'thC': thC}
    kw_powerlaw = {'thC': thC*math.sqrt(b), 'b': b}

    TH = np.linspace(0.0, thW, 20)

    t_a2C_tophat = np.empty(TH.shape)
    t_C2a_tophat = np.empty(TH.shape)
    t_b2C_tophat = np.empty(TH.shape)
    t_C2b_tophat = np.empty(TH.shape)
    t_a2C_Gaussian = np.empty(TH.shape)
    t_C2a_Gaussian = np.empty(TH.shape)
    t_b2C_Gaussian = np.empty(TH.shape)
    t_C2b_Gaussian = np.empty(TH.shape)
    t_a2C_powerlaw = np.empty(TH.shape)
    t_C2a_powerlaw = np.empty(TH.shape)
    t_b2C_powerlaw = np.empty(TH.shape)
    t_C2b_powerlaw = np.empty(TH.shape)

    for i in range(len(TH)):
        if TH[i]-thC >= 0.0:
            th = np.linspace(TH[i]-thC, TH[i], 500)
            t_a2C_tophat[i] = get_t(th, f_tophat, kw_tophat, 0.0, t0)[-1]
            t_a2C_Gaussian[i] = get_t(th, f_Gaussian, kw_Gaussian, 0.0, t0)[-1]
            t_a2C_powerlaw[i] = get_t(th, f_powerlaw, kw_powerlaw, 0.0, t0)[-1]
            th = th[::-1]
            t_C2a_tophat[i] = get_t(th, f_tophat, kw_tophat, 0.0, t0)[-1]
            t_C2a_Gaussian[i] = get_t(th, f_Gaussian, kw_Gaussian, 0.0, t0)[-1]
            t_C2a_powerlaw[i] = get_t(th, f_powerlaw, kw_powerlaw, 0.0, t0)[-1]
        else:
            t_a2C_tophat[i] = 0.0
            t_a2C_Gaussian[i] = 0.0
            t_a2C_powerlaw[i] = 0.0
            t_C2a_tophat[i] = 0.0
            t_C2a_Gaussian[i] = 0.0
            t_C2a_powerlaw[i] = 0.0
        if TH[i]+thC <= thW:
            th = np.linspace(TH[i], TH[i]+thC, 500)
            t_C2b_tophat[i] = get_t(th, f_tophat, kw_tophat, 0.0, t0)[-1]
            t_C2b_Gaussian[i] = get_t(th, f_Gaussian, kw_Gaussian, 0.0, t0)[-1]
            t_C2b_powerlaw[i] = get_t(th, f_powerlaw, kw_powerlaw, 0.0, t0)[-1]
            th = th[::-1]
            t_b2C_tophat[i] = get_t(th, f_tophat, kw_tophat, 0.0, t0)[-1]
            t_b2C_Gaussian[i] = get_t(th, f_Gaussian, kw_Gaussian, 0.0, t0)[-1]
            t_b2C_powerlaw[i] = get_t(th, f_powerlaw, kw_powerlaw, 0.0, t0)[-1]
        else:
            t_C2b_tophat[i] = 0.0
            t_C2b_Gaussian[i] = 0.0
            t_C2b_powerlaw[i] = 0.0
            t_b2C_tophat[i] = 0.0
            t_b2C_Gaussian[i] = 0.0
            t_b2C_powerlaw[i] = 0.0

    fig, ax = plt.subplots(1, 1)
    ax.plot(TH[t_a2C_tophat > 0], t_a2C_tophat[t_a2C_tophat > 0],
            color='C0', ls='-')
    ax.plot(TH[t_b2C_tophat > 0], t_b2C_tophat[t_b2C_tophat > 0],
            color='C0', ls='--')
    ax.plot(TH[t_C2a_tophat > 0], t_C2a_tophat[t_C2a_tophat > 0],
            color='C0', ls='-.')
    ax.plot(TH[t_C2b_tophat > 0], t_C2b_tophat[t_C2b_tophat > 0],
            color='C0', ls=':')
    ax.plot(TH[t_a2C_Gaussian > 0], t_a2C_Gaussian[t_a2C_Gaussian > 0],
            color='C1', ls='-')
    ax.plot(TH[t_b2C_Gaussian > 0], t_b2C_Gaussian[t_b2C_Gaussian > 0],
            color='C1', ls='--')
    ax.plot(TH[t_C2a_Gaussian > 0], t_C2a_Gaussian[t_C2a_Gaussian > 0],
            color='C1', ls='-.')
    ax.plot(TH[t_C2b_Gaussian > 0], t_C2b_Gaussian[t_C2b_Gaussian > 0],
            color='C1', ls=':')
    ax.plot(TH[t_a2C_powerlaw > 0], t_a2C_powerlaw[t_a2C_powerlaw > 0],
            color='C2', ls='-')
    ax.plot(TH[t_b2C_powerlaw > 0], t_b2C_powerlaw[t_b2C_powerlaw > 0],
            color='C2', ls='--')
    ax.plot(TH[t_C2a_powerlaw > 0], t_C2a_powerlaw[t_C2a_powerlaw > 0],
            color='C2', ls='-.')
    ax.plot(TH[t_C2b_powerlaw > 0], t_C2b_powerlaw[t_C2b_powerlaw > 0],
            color='C2', ls=':')

    t_tophat_ana = t0 * np.power(np.sqrt(2.0)*3*thC, 2.0/3.0)\
        * np.power(f_tophat(TH, **kw_tophat), 1.0/3.0)
    t_Gaussian_ana = t0 * np.power(np.sqrt(2.0)*3*thC, 2.0/3.0)\
        * np.power(f_Gaussian(TH, **kw_Gaussian), 1.0/3.0)
    t_powerlaw_ana = t0 * np.power(np.sqrt(2.0)*3*thC, 2.0/3.0)\
        * np.power(f_powerlaw(TH, **kw_powerlaw), 1.0/3.0)

    ax.plot(TH, t_tophat_ana, color='C0', alpha=0.5)
    ax.plot(TH, t_Gaussian_ana, color='C1', alpha=0.5)
    ax.plot(TH, t_powerlaw_ana, color='C2', alpha=0.5)

    ax.set_yscale('log')
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'Time to go $\theta_C$')

    fig.tight_layout()
    print("Saving travelTimes.pdf")
    fig.savefig("travelTimes.pdf")
    plt.close(fig)


def makeSpreadStructPlot(TH, f, f_kwargs, spread, name):

    t0 = 1.0e8

    thC = f_kwargs['thC']
    rho0 = 1.0 * grb.mp

    E0 = 16*np.pi/9.0 * rho0 * t0**3 * grb.c**5
    print("E0: {0:.3e} erg".format(E0))

    # t = t0 * np.array([3.0e-2, 1.0e-1, 2.0e-1, 3.0e-1, 5.0e-1, 1.0,
    #                    3.0, 1.0e1, 30.0])
    t = t0 * np.geomspace(0.001, 100.0, 10)
    tth = np.geomspace(min(t.min(), 1.0e-6*t0), t.max(), 10000)
    tind = np.searchsorted(tth, t)

    R = np.empty((len(t), len(TH)))
    u = np.empty((len(t), len(TH)))
    thj = np.empty((len(t), len(TH)))

    for i in range(1, len(TH)):
        thi = 0.5*(TH[i-1]+TH[i])

        Eth = E0 * f(thi, **f_kwargs)
        u0 = math.sqrt(9.0 * Eth / (16*np.pi * rho0 * grb.c**5 * tth[0]**3))
        R0 = grb.c*(1 - 1.0 / (16*u0*u0)) * tth[0]
        th0 = TH[i]
        Rth, uth, thth = grb.shock.shockEvolSpreadRK4(tth, R0, u0, th0, 0.0,
                                                      rho0,
                                                      0.0, 0.0, 0.0, 0.0, 0.0,
                                                      0.0, thC, spread)
        for j in range(len(t)):
            R[j, i] = Rth[tind[j]]
            u[j, i] = uth[tind[j]]
            thj[j, i] = thth[tind[j]]

    R[:, 0] = R[:, 1]
    u[:, 0] = u[:, 1]
    thj[:, 0] = 0.0

    g = np.sqrt(u*u+1)
    be = u/g

    E = 4*np.pi/9.0 * rho0 * grb.c*grb.c * R*R*R * (4*u*u+3) * be*be

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    for i in range(len(t)):
        label = r"$t = {0:.1f}\times 10^{{{1:d}}}$s".format(*frexp10(t[i]))
        R0 = grb.c * t[i]
        ax[0, 0].plot(R[i]*np.sin(thj[i])/R0, R[i]*np.cos(thj[i])/R0,
                      label=label)
        ax[0, 1].plot(thj[i], u[i], label=label)
        ax[1, 1].plot(thj[i], E[i], label=label)
    for i in range(len(TH)):
        ax[1, 0].plot(R[:, i]*np.sin(thj[:, i]),
                      R[:, i]*np.cos(thj[:, i]), color='k', lw=0.5)

    ax[0, 0].set_aspect('equal')
    ax[0, 0].set_xlim(0.0, 1.0)
    ax[0, 0].set_ylim(0.0, 1.0)
    ax[0, 0].set_xlabel(r'$x/ct$')
    ax[0, 0].set_ylabel(r'$y/ct$')

    ax[1, 0].set_aspect('equal')
    ax[1, 0].set_xlim(0.0, 1.05*R.max())
    ax[1, 0].set_ylim(0.0, 1.05*R.max())
    ax[1, 0].set_xlabel(r'$x$ (cm)')
    ax[1, 0].set_ylabel(r'$y$ (cm)')

    ax[0, 1].legend()
    ax[0, 1].set_yscale('log')
    ax[0, 1].set_xlabel(r'$\theta$ (rad)')
    ax[0, 1].set_ylabel(r'$u$')
    ax[0, 1].set_ylim(1.0e-3, 1.0e5)

    ax[1, 1].set_yscale('log')
    ax[1, 1].set_xlabel(r'$\theta$ (rad)')
    ax[1, 1].set_ylabel(r'$E$ (erg)')
    ax[1, 1].set_ylim(1.0e48, 5.0e53)

    fig.tight_layout()
    figname = "structure_" + name + ".pdf"
    print("Saving " + figname)
    fig.savefig(figname)
    plt.close(fig)


if __name__ == "__main__":

    thC = 0.05
    b = 10
    t0 = 1.0

    th_02e = np.linspace(0.0, 0.5*np.pi, 500)
    th_e20 = np.linspace(0.5*np.pi, 0.0, 500)
    th_c20 = np.linspace(thC, 0.0, 500)
    th_c2e = np.linspace(thC, 0.5*np.pi, 500)

    kw_tophat = {'thC': thC}
    kw_Gaussian = {'thC': thC}
    kw_powerlaw = {'thC': thC, 'b': b}

    t_02e_tophat = get_t(th_02e, f_tophat, kw_tophat, 0.0, t0)
    t_02e_Gaussian = get_t(th_02e, f_Gaussian, kw_Gaussian, 0.0, t0)
    t_02e_powerlaw = get_t(th_02e, f_powerlaw, kw_powerlaw, 0.0, t0)

    t_e20_tophat = get_t(th_e20, f_tophat, kw_tophat, 0.0, t0)
    t_e20_Gaussian = get_t(th_e20, f_Gaussian, kw_Gaussian, 0.0, t0)
    t_e20_powerlaw = get_t(th_e20, f_powerlaw, kw_powerlaw, 0.0, t0)

    t_c20_tophat = get_t(th_c20, f_tophat, kw_tophat, 0.0, t0)
    t_c20_Gaussian = get_t(th_c20, f_Gaussian, kw_Gaussian, 0.0, t0)
    t_c20_powerlaw = get_t(th_c20, f_powerlaw, kw_powerlaw, 0.0, t0)

    t_c2e_tophat = get_t(th_c2e, f_tophat, kw_tophat, 0.0, t0)
    t_c2e_Gaussian = get_t(th_c2e, f_Gaussian, kw_Gaussian, 0.0, t0)
    t_c2e_powerlaw = get_t(th_c2e, f_powerlaw, kw_powerlaw, 0.0, t0)

    t_e202e_tophat = get_t(th_02e, f_tophat, kw_tophat,
                           t_e20_tophat[-1], t0)
    t_e202e_Gaussian = get_t(th_02e, f_Gaussian, kw_Gaussian,
                             t_e20_Gaussian[-1], t0)
    t_e202e_powerlaw = get_t(th_02e, f_powerlaw, kw_powerlaw,
                             t_e20_powerlaw[-1], t0)

    t_c202e_tophat = get_t(th_02e, f_tophat, kw_tophat,
                           t_c20_tophat[-1], t0)
    t_c202e_Gaussian = get_t(th_02e, f_Gaussian, kw_Gaussian,
                             t_c20_Gaussian[-1], t0)
    t_c202e_powerlaw = get_t(th_02e, f_powerlaw, kw_powerlaw,
                             t_c20_powerlaw[-1], t0)

    rho0 = 1.0
    E0 = 16*np.pi/9.0 * rho0 * t0**3 * grb.c**5

    t = np.geomspace(1.0e-6, 1.0e1, 10000)
    u0 = math.pow(t[0]/t0, -1.5)
    R0 = grb.c*(1 - 1.0 / (16*u0*u0)) * t[0]

    RN, uN, thN = grb.shock.shockEvolSpreadRK4(t, R0, u0, thC, 0.0, rho0, 0.0,
                                               0.0, 0.0, 0.0, 0.0, 0.0, thC,
                                               True)
    RN2, uN2, thN2 = grb.shock.shockEvolSpreadRK4(t, R0, u0, thC, 0.0, rho0,
                                                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                  thC, 5)

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    ax.plot(t, thN, color='grey', lw=2, alpha=0.5)
    ax.plot(t, thN2, color='grey', lw=2, alpha=0.5, ls='--')

    ax.plot(t_02e_tophat, th_02e, color='C0', ls='-')
    ax.plot(t_02e_Gaussian, th_02e, color='C1', ls='-')
    ax.plot(t_02e_powerlaw, th_02e, color='C2', ls='-')

    ax.plot(t_e20_tophat, th_e20, color='C0', ls='--')
    ax.plot(t_e20_Gaussian, th_e20, color='C1', ls='--')
    ax.plot(t_e20_powerlaw, th_e20, color='C2', ls='--')
    ax.plot(t_e202e_tophat, th_02e, color='C0', ls='--')
    ax.plot(t_e202e_Gaussian, th_02e, color='C1', ls='--')
    ax.plot(t_e202e_powerlaw, th_02e, color='C2', ls='--')

    ax.plot(t_c20_tophat, th_c20, color='C0', ls=':')
    ax.plot(t_c20_Gaussian, th_c20, color='C1', ls=':')
    ax.plot(t_c20_powerlaw, th_c20, color='C2', ls=':')
    ax.plot(t_c202e_tophat, th_02e, color='C0', ls=':')
    ax.plot(t_c202e_Gaussian, th_02e, color='C1', ls=':')
    ax.plot(t_c202e_powerlaw, th_02e, color='C2', ls=':')

    ax.plot(t_c2e_tophat, th_c2e, color='C0', ls='-.')
    ax.plot(t_c2e_Gaussian, th_c2e, color='C1', ls='-.')
    ax.plot(t_c2e_powerlaw, th_c2e, color='C2', ls='-.')

    ax.axvline(np.power(1.0 / (np.sqrt(2)*3*thC), -2.0/3.0), color='grey')

    ax.set_xscale('log')
    ax.set_xlim(1.0e-3, 1.0e1)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\theta$')

    fig.tight_layout()
    print("Saving waveTracks.pdf")
    fig.savefig("waveTracks.pdf")
    plt.close(fig)

    make_travelTimes_plot()

    thW = 0.4
    TH = np.linspace(0.0, thW, 100)
    THtophat = np.linspace(0.0, thC, 25)
    makeSpreadStructPlot(THtophat, f_tophat, kw_tophat, 0, "tophat_NoSpread")
    makeSpreadStructPlot(TH, f_Gaussian, kw_Gaussian, 0, "Gaussian_NoSpread")
    makeSpreadStructPlot(TH, f_powerlaw, kw_powerlaw, 0, "powerlaw_NoSpread")
    makeSpreadStructPlot(THtophat, f_tophat, kw_tophat, 6, "tophat_SpreadBad")
    makeSpreadStructPlot(TH, f_Gaussian, kw_Gaussian, 6, "Gaussian_SpreadBad")
    makeSpreadStructPlot(TH, f_powerlaw, kw_powerlaw, 6, "powerlaw_SpreadBad")
    makeSpreadStructPlot(THtophat, f_tophat, kw_tophat, 8, "tophat_SpreadGood")
    makeSpreadStructPlot(TH, f_Gaussian, kw_Gaussian, 8, "Gaussian_SpreadGood")
    makeSpreadStructPlot(TH, f_powerlaw, kw_powerlaw, 8, "powerlaw_SpreadGood")

    TH = np.linspace(0.0, thC, 41)
    makeSpreadStructPlot(TH, f_tophat, kw_tophat, 6, "tophat_Spread_40")
    TH = np.linspace(0.0, thC, 21)
    makeSpreadStructPlot(TH, f_tophat, kw_tophat, 6, "tophat_Spread_20")
    TH = np.linspace(0.0, thC, 11)
    makeSpreadStructPlot(TH, f_tophat, kw_tophat, 6, "tophat_Spread_10")
    TH = np.linspace(0.0, thC, 6)
    makeSpreadStructPlot(TH, f_tophat, kw_tophat, 6, "tophat_Spread_05")
    TH = np.linspace(0.0, thC, 3)
    makeSpreadStructPlot(TH, f_tophat, kw_tophat, 6, "tophat_Spread_02")
