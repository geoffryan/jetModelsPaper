import math
import numpy as np
import matplotlib.pyplot as plt
import grbpy as grb

decompPlot = False


def f_struct(th, thC, jetType=0):

    if jetType == -1:
        return 1.0
    elif jetType == 0:
        return np.exp(-0.5*th*th/(thC*thC))
    elif jetType == 4:
        return 1.0/(1.0 + th*th/(thC*thC))

    return 1.0


def t_on(thV, thC, thW, E0, n0, p, jetType=0):

    if thW >= thV:
        return 0.0

    c5 = grb.c ** 5

    fW = f_struct(thW, thC, jetType)

    t0 = math.pow(9.0/(16.0*np.pi) * E0*fW/(n0*grb.mp*c5), 1.0/3.0)

    # return t0 * math.pow(math.sin(thV-thW), 8.0/3.0)
    return t0 * math.pow(2, -1.6) * math.pow(2*math.sin(0.5*(thV-thW)),
                                             8.0/3.0)


def t_peak(thV, thC, thW, E0, n0, p):

    if thC >= thV:
        return 0.0

    c5 = grb.c ** 5

    t0 = math.pow(9.0/(16.0*np.pi) * E0/(n0*grb.mp*c5), 1.0/3.0)

    # return t0 * math.pow(math.sin(thV-thC), 8.0/3.0)
    return t0 * math.pow(2*math.sin(0.5*thV), 8.0/3.0)


def t_newt(E0, n0):

    c5 = grb.c ** 5
    return math.pow(9.0/(16.0*np.pi) * E0/(n0*grb.mp*c5), 1.0/3.0)


def t_coast(umax, umin, Er, k, Mej_solar, n0, p):

    u = umax
    g = np.sqrt(u*u+1)
    b = u/g
    rho0 = n0*grb.mp
    gm1 = u*u/(g+1)
    Mej = Mej_solar * grb.Msun

    Rd = np.power(9 * gm1*Mej / (4*np.pi * rho0 * b*b*(4*u*u+3)), 1.0/3.0)

    vs = 4*u*g * grb.c / (4*u*u+3)

    td = Rd / vs

    td_obs = td - Rd/grb.c

    return td_obs


def t_end(umax, umin, Er, k, Mej_solar, n0, p):

    c5 = grb.c ** 5

    tN = np.power(9*Er/(16*np.pi*grb.mp*n0 * c5), 1.0/3.0)

    tmin_obs = 0.25*(2.+k)/(8.+k) * np.power(umin, -(8.+k)/3.) * tN

    return tmin_obs


def sync_slope(p, seg='G'):

    if seg == 'D':
        return 1.0/3.0
    elif seg == 'E':
        return 1.0/3.0
    elif seg == 'F':
        return -0.5
    elif seg == 'G':
        return 0.5*(1-p)
    elif seg == 'H':
        return -0.5*p

    return 0.0


def em_slope_t(p, seg='G'):
    if seg == 'D':
        return 0.0
    elif seg == 'E':
        return 2.0/3.0
    elif seg == 'F':
        return -1.0
    elif seg == 'G':
        return 0.0
    elif seg == 'H':
        return -1.0

    return 0.0


def em_slope_g(p, seg='G'):
    if seg == 'D':
        return 1.0
    elif seg == 'E':
        return 7.0/3.0
    elif seg == 'F':
        return 1.5
    elif seg == 'G':
        return 0.5*(1+3*p)
    elif seg == 'H':
        return 1.5*p

    return 0.0


def em_slope_v(p, seg='G'):
    if seg == 'D':
        return -2.0/3.0
    elif seg == 'E':
        return 2.0
    elif seg == 'F':
        return -0.5
    elif seg == 'G':
        return 0.5*(5*p-3)
    elif seg == 'H':
        return 0.5*(5*p-3)

    return 0.0


def g_struct(thV, thC, thW, jetType=0):

    th = 0.5*thV
    psi = thV - th
    chi = 2*np.tan(0.5*psi)

    if jetType == -1:
        return 0.0
    elif jetType == 0:
        return chi * th / (thC*thC)
    elif jetType == 4:
        return chi * 2*th/(thC*thC + th*th)

    return 0.0


def t_psi(psi, thV, thC, E0, n0, jetType=0):

    c5 = grb.c ** 5
    th = thV-psi
    f = f_struct(th, thC, jetType)
    t0 = np.power(9.0/(16.0*np.pi) * E0*f/(n0*grb.mp*c5), 1.0/3.0)

    return t0 * np.power(np.fabs(np.sin(psi)), 8.0/3.0)


def g_psi(psi, thV, thC, jetType=0):

    th = thV-psi

    if jetType == 0:
        return 2*np.tan(0.5*psi)*th/(thC*thC)
    elif jetType == 4:
        return 2*np.tan(0.5*psi)*2*th/(thC*thC+th*th)

    return 0.0


def slope_struct_psi(psi, thV, thC, p, jetType=0, seg='G'):

    p_em_t = em_slope_t(p, seg)
    # p_em_g = em_slope_g(p, seg)
    p_em_g = em_slope_g(p, seg)+1
    p_nu = sync_slope(p, seg)
    g = g_psi(psi, thV, thC, jetType)

    return (6-3*p_em_g+2*p_em_t+3*p_nu+(3+p_em_t)*g) / (8.0+g)


def slope_off(thV, thC, thW, p, jetType=0, seg='G'):

    p_em_t = em_slope_t(p, seg)
    p_em_g = em_slope_g(p, seg)
    p_nu = sync_slope(p, seg)

    return 9.0 - 1.5*(p_em_g+p_nu) + p_em_t


def slope_struct(thV, thC, thW, p, jetType=0, seg='G'):
    p_em_t = em_slope_t(p, seg)
    # p_em_g = em_slope_g(p, seg)
    p_em_g = em_slope_g(p, seg)+1
    p_nu = sync_slope(p, seg)
    g = g_struct(thV, thC, thW, jetType)

    return (6-3*p_em_g+2*p_em_t+3*p_nu+(3+p_em_t)*g) / (8.0+g)


def slope_prebreak(thV,  thC, thW, p, jetType=0, seg='G'):
    p_em_t = em_slope_t(p, seg)
    p_em_g = em_slope_g(p, seg)
    p_nu = sync_slope(p, seg)

    return 0.25*(3+p_em_t)+0.375*(p_nu-p_em_g)


def slope_postbreak(thV,  thC, thW, p, jetType=0, seg='G'):
    p_em_t = em_slope_t(p, seg)
    p_em_g = em_slope_g(p, seg)
    p_nu = sync_slope(p, seg)

    return 0.25*p_em_t+0.75*(p_nu-p_em_g)


def slope_coast(p, k, seg='G'):
    return 3.0


def slope_cocoon(p, k, seg='G'):
    p_em_t = em_slope_t(p, seg)
    p_em_g = em_slope_g(p, seg)
    p_nu = sync_slope(p, seg)

    return (6-3*p_em_g+2*p_em_t+3*p_nu+(3+p_em_t)*k) / (8.0+k)


def slope_newt(p, seg='G'):
    p_em_t = em_slope_t(p, seg)
    p_em_v = em_slope_v(p, seg)

    return 1.2 - 0.6*p_em_v + p_em_t


def pl_lc_jet(t, thV, thC, thW, E0, n0, p, jetType=0, seg='G'):

    to = t_on(thV, thC, thW, E0, n0, p, jetType)
    tp = t_peak(thV, thC, thW, E0, n0, p)
    tN = t_newt(E0, n0)

    s1 = slope_off(thV, thC, thW, p, jetType, seg)
    s2 = slope_struct(thV, thC, thW, p, jetType, seg)
    s3 = slope_postbreak(thV, thC, thW, p, jetType, seg)
    s4 = slope_newt(p, seg)

    phase1 = t < to
    phase2 = (t >= to) * (t < tp)
    phase3 = (t >= tp) * (t < tN)
    phase4 = t >= tN

    print("t_on: {0:.2e} s".format(to))
    print("t_peak: {0:.2e} s".format(tp))
    print("t_N: {0:.2e} s".format(tN))
    print("s1: {0:.2f}".format(s1))
    print("s2: {0:.2f}".format(s2))
    print("s3: {0:.2f}".format(s3))
    print("s4: {0:.2f}".format(s4))

    Fnu = np.empty(t.shape)
    if phase1.any():
        Fnu[phase1] = np.power(t[phase1]/to, s1) * math.pow(to/tp, s2)
    Fnu[phase2] = np.power(t[phase2]/tp, s2)
    Fnu[phase3] = np.power(t[phase3]/tp, s3)
    Fnu[phase4] = np.power(t[phase4]/tN, s4) * math.pow(tN/tp, s3)

    return Fnu


def pl_lc_cocoon(t, umax, umin, Er, k, Mej, n0, p, seg='G'):

    tc = t_coast(umax, umin, Er, k, Mej, n0, p)
    te = t_end(umax, umin, Er, k, Mej, n0, p)
    tN = t_newt(Er*np.power(umin, -k), n0)

    s1 = slope_coast(p, k, seg)
    s2 = slope_cocoon(p, k, seg)
    s3 = slope_prebreak(thV, thC, thW, p, jetType, seg)
    s4 = slope_newt(p, seg)

    phase1 = t < tc
    phase2 = (t >= tc) * (t < te)
    phase3 = (t >= te) * (t < tN)
    phase4 = t >= tN

    print("t_c: {0:.2e} s".format(tc))
    print("t_e: {0:.2e} s".format(te))
    print("t_N: {0:.2e} s".format(tN))
    print("s1: {0:.2f}".format(s1))
    print("s2: {0:.2f}".format(s2))
    print("s3: {0:.2f}".format(s3))
    print("s4: {0:.2f}".format(s4))

    Fnu = np.empty(t.shape)
    if phase1.any():
        Fnu[phase1] = np.power(t[phase1]/tc, s1) * math.pow(tc/te, s2)
    Fnu[phase2] = np.power(t[phase2]/te, s2)
    Fnu[phase3] = np.power(t[phase3]/te, s3)
    Fnu[phase4] = np.power(t[phase4]/tN, s4) * math.pow(tN/te, s3)

    return Fnu


def plot_lc_jet(t, thV, thC, thW, E0, n0, p, jetType=0, seg='G'):

    nu = np.empty(t.shape)
    nu[:] = 1.0e14
    Y = np.array([thV, E0, thC, thW, 0, 0, 0, n0, p, 0.1, 1.0e-4, 1.0, 10])
    FnuSA = grb.fluxDensity(t, nu, jetType, 0, *Y)

    to = t_on(thV, thC, thW, E0, n0, p, jetType)
    tp = t_peak(thV, thC, thW, E0, n0, p)
    # ip = np.searchsorted(t, tp)
    # F0 = FnuSA[ip]
    # FnuSA /= F0

    if seg is not None:
        Fnu = pl_lc_jet(t, thV, thC, thW, E0, n0, p, jetType, seg)
    # else:
        # FnuSAb = grb.fluxDensity(t, nu*2, jetType, 0, *Y)
        # FnuSAa = grb.fluxDensity(t, nu*0.5, jetType, 0, *Y)
        # beta = np.log(FnuSAb/FnuSAa) / np.log(2/0.5)
        # bD = 1.0/3.0
        # bE = 1.0/3.0
        # bF = -0.5
        # bG = 0.5*(1-p)
        # bH = -0.5*p
        # HHH = beta < 0.25-0.5*p
        # FFF = (beta >= 0.25-0.5*p)

    ir = np.searchsorted(t, 0.1*tp)
    F0 = FnuSA[ir]/Fnu[ir]
    FnuSA /= F0

    tc = np.sqrt(t[1:]*t[:-1])
    dF = np.log(Fnu[1:]/Fnu[:-1]) / np.log(t[1:]/t[:-1])
    dFSA = np.log(FnuSA[1:]/FnuSA[:-1]) / np.log(t[1:]/t[:-1])

    psi = np.linspace(np.abs(thV-thW), thV, 100)
    tpsi = t_psi(psi, thV, thC, E0, n0, jetType)
    dFpsi = slope_struct_psi(psi, thV, thC, p, jetType, seg)

    fig, ax = plt.subplots(2, 1)
    ax[0].axvline(to, color='grey', ls=':')
    ax[0].axvline(tp, color='grey', ls='--')
    ax[1].axvline(to, color='grey', ls=':')
    ax[1].axvline(tp, color='grey', ls='--')
    ax[0].plot(t, Fnu)
    ax[0].plot(t, FnuSA)
    ax[1].plot(tc, dF)
    ax[1].plot(tc, dFSA)
    ax[1].plot(tpsi, dFpsi)
    ax[0].set_xscale("log")
    ax[1].set_xscale("log")
    ax[0].set_yscale("log")
    ax[1].set_yscale("linear")
    ax[1].set_xlabel(r"$t$ (s)")
    ax[0].set_ylabel(r"$F_\nu/F_0$")
    ax[1].set_ylabel(r"d$\log F_\nu$/d$\log t$")
    ax[0].set_xlim(t.min(), t.max())
    ax[1].set_xlim(t.min(), t.max())

    fig2 = None

    if decompPlot:
        nth = 10

        fig2, ax2 = plt.subplots()
        ax2.plot(t, FnuSA)
        ths = np.linspace(0.0, thW, nth+1)
        Y = np.array([thV, E0, thC, thW, 0, 0, 0, n0, p, 0.1, 1.0e-4, 1.0, 10])
        for i in range(nth):
            Y[1] = E0*f_struct(0.5*(ths[i]+ths[i+1]), thC, jetType)
            Y[2] = ths[i]
            Y[3] = ths[i+1]
            FnuR = grb.fluxDensity(t, nu, -2, 0, *Y) / F0
            ax2.plot(t, FnuR, ls='--')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel(r"$t$ (s)")
        ax2.set_ylabel(r"$F_\nu / F_0$")

    return fig, fig2


def plot_lc_cocoon(t, umax, umin, Er, k, Mej, n0, p, jetType=3, seg='G'):

    nu = np.empty(t.shape)
    nu[:] = 1.0e14
    Y = np.array([umax, umin, Er, k, Mej, 0, 0, 0,
                  n0, p, 0.1, 1.0e-4, 1.0, 10])
    FnuSA = grb.fluxDensity(t, nu, jetType, 0, *Y)

    tc = t_coast(umax, umin, Er, k, Mej, n0, p)
    te = t_end(umax, umin, Er, k, Mej, n0, p)
    # ip = np.searchsorted(t, tp)
    # F0 = FnuSA[ip]
    # FnuSA /= F0

    Fnu = pl_lc_cocoon(t, umax, umin, Er, k, Mej, n0, p, seg)

    ir = np.searchsorted(t, 0.1*te)
    F0 = FnuSA[ir]/Fnu[ir]
    FnuSA /= F0

    Tc = np.sqrt(t[1:]*t[:-1])
    dF = np.log(Fnu[1:]/Fnu[:-1]) / np.log(t[1:]/t[:-1])
    dFSA = np.log(FnuSA[1:]/FnuSA[:-1]) / np.log(t[1:]/t[:-1])

    fig, ax = plt.subplots(2, 1)
    ax[0].axvline(tc, color='grey', ls=':')
    ax[0].axvline(te, color='grey', ls='--')
    ax[1].axvline(tc, color='grey', ls=':')
    ax[1].axvline(te, color='grey', ls='--')
    ax[0].plot(t, Fnu)
    ax[0].plot(t, FnuSA)
    ax[1].plot(Tc, dF)
    ax[1].plot(Tc, dFSA)
    ax[0].set_xscale("log")
    ax[1].set_xscale("log")
    ax[0].set_yscale("log")
    ax[1].set_yscale("linear")
    ax[1].set_xlabel(r"$t$ (s)")
    ax[0].set_ylabel(r"$F_\nu/F_0$")
    ax[1].set_ylabel(r"d$\log F_\nu$/d$\log t$")
    ax[0].set_xlim(t.min(), t.max())
    ax[1].set_xlim(t.min(), t.max())

    fig2 = None

    return fig, fig2


def prettyLCAxis(ax, tmin, tmax, Fmin, Fmax):

    # ax.set_title('$E_0=10^{52}\mathrm{erg}\ n_0 = 10^{-3}\mathrm{cm}^{-3}\
    #                \\theta_0=0.1\ \mathrm{rad} \ \\theta_{obs} = 0.5\
    #                \mathrm{rad}$', fontsize=16)
    # ax.set_title('$\mathrm{Off-Axis\ Gaussian\ Jet}$')
    ax.set_title(r'Off-Axis Gaussian Jet', fontsize=24)
    ax.set_xlabel('$t \ \mathrm{ (days)}$', fontsize=20)
    ax.set_ylabel('$F/F_0$', fontsize=20)
    ax.legend(loc='lower right', fontsize=18)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(tmin, tmax)
    ax.set_ylim(Fmin, Fmax)
    ax.tick_params(labelsize=18)


"""
def makeNicePlot(t, thV, thC, thW, E0, n0, p, jetType, seg):

    day = 86400
    tmin = t.min() / day
    tmax = t.max() / day
    Fmin = 1.0e-6
    Fmax = 3.0

    nu = np.empty(t.shape)
    nu[:] = 1.0e14
    Y = np.array([thV, E0, thC, thW, 0, 0, 0, n0, p, 0.1, 1.0e-4, 1.0, 10])

    to = t_on(thV, thC, thW, E0, n0, p, jetType)
    tp = t_peak(thV, thC, thW, E0, n0, p)
    FA = pl_lc(t, thV, thC, thW, E0, n0, p, jetType, seg)
    FN = grb.fluxDensity(t, nu, jetType, 0, *Y)

    i = np.searchsorted(t, 0.1*tp)
    FN /= FN[i]/FA[i]

    from matplotlib import rc
    # rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)

    phase2 = (t > to) * (t < tp)

    labelN = r'Numerical'
    labelPL = r'Power Law Approximation'
    labelSJ = r'Structured Jet Closure'

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(t/day, FN, lw=2, color='k', ls='--', label=labelN)
    ax.plot(t/day, FA, lw=4, color='k', alpha=0.5, label=labelPL)
    ax.plot(t[phase2]/day, FA[phase2], lw=4, color='r', alpha=1.0,
            label=labelSJ)

    prettyLCAxis(ax, tmin, tmax, Fmin, Fmax)
    fig.savefig("struct_lc_NUM+PL_bright.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(t/day, FN, lw=2, color='k', ls='--', label=labelN)
    ax.plot(t/day, FA, lw=4, color='k', alpha=0.5, label=labelPL)
    ax.plot(t[phase2]/day, FA[phase2], lw=4, color='r', alpha=0.5,
            label=labelSJ)

    prettyLCAxis(ax, tmin, tmax, Fmin, Fmax)
    fig.savefig("struct_lc_NUM+PL_dull.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(t/day, FN, lw=2, color='k', ls='--', label=labelN)
    ax.plot(t/day, FA, lw=4, color='k', alpha=0.5, label=labelPL)

    prettyLCAxis(ax, tmin, tmax, Fmin, Fmax)
    fig.savefig("struct_lc_NUM+PL_grey.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(t/day, FN, lw=2, color='k', ls='--', label=labelN)

    prettyLCAxis(ax, tmin, tmax, Fmin, Fmax)
    fig.savefig("struct_lc_NUM.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(t/day, FA, lw=4, color='k', alpha=0.5, label=labelPL)
    ax.plot(t[phase2]/day, FA[phase2], lw=4, color='r', alpha=0.5,
            label=labelSJ)

    prettyLCAxis(ax, tmin, tmax, Fmin, Fmax)
    fig.savefig("struct_lc_PL_dull.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(t/day, FA, lw=4, color='k', alpha=0.5, label=labelPL)
    ax.plot(t[phase2]/day, FA[phase2], lw=4, color='r', alpha=1.0,
            label=labelSJ)

    prettyLCAxis(ax, tmin, tmax, Fmin, Fmax)
    fig.savefig("struct_lc_PL_bright.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(t/day, FA, lw=4, color='k', alpha=0.5, label=labelPL)

    prettyLCAxis(ax, tmin, tmax, Fmin, Fmax)
    fig.savefig("struct_lc_PL_grey.pdf")
    plt.close(fig)
"""


def plotSlopes(p, s3, name):

    g = np.linspace(0, 10, 200)

    segs = ['D', 'E', 'F', 'G', 'H']

    fig, ax = plt.subplots(1, 1)

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

    for i, seg in enumerate(segs):
        s1_p2 = em_slope_g(2.0, seg)
        s2_p2 = em_slope_t(2.0, seg)
        be_p2 = sync_slope(2.0, seg)

        s1 = em_slope_g(p, seg)
        s2 = em_slope_t(p, seg)
        be = sync_slope(p, seg)

        s1_p3 = em_slope_g(3.0, seg)
        s2_p3 = em_slope_t(3.0, seg)
        be_p3 = sync_slope(3.0, seg)

        s = (2*(3+s2)-3*(2+s3+s1-be)+(3+s2)*g) / (8+g)
        s_p2 = (2*(3+s2_p2)-3*(2+s3+s1_p2-be_p2)+(3+s2_p2)*g) / (8+g)
        s_p3 = (2*(3+s2_p3)-3*(2+s3+s1_p3-be_p3)+(3+s2_p3)*g) / (8+g)

        ax.fill_between(g, s_p2, s_p3, alpha=0.5, color=colors[i])
        ax.plot(g, s, label=seg, alpha=1.0, color=colors[i])

    ax.set_ylim(-3, 3)

    ax.set_ylabel(r'$d \log F_\nu\ /\ d \log t_{obs}$')
    ax.set_xlabel(r'$g$')

    fig.legend()
    fig.savefig(name)


def plotG(name, jetType, mode=0):

    fig, ax = plt.subplots(1, 1)

    thCs = [0.01, 0.03, 0.1, 0.3]
    thVs = [0.2, 0.5, 0.8]

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    lss = ['-', '--', '-.', ':']

    for i, thC in enumerate(thCs):
        if mode is not 0:
            thVs = [thC, 3*thC, 6*thC]
        for j, thV in enumerate(thVs):
            psi = np.linspace(0, thV, 100)
            g = g_psi(psi, thV, thC, jetType)
            ax.plot(psi, g, ls=lss[i], color=colors[j],
                    label=r'$\theta_V={0:.2f}$ $\theta_C={1:.2f}$'.format(
                            thV, thC))

    ax.set_yscale('log')
    ax.set_ylabel('g')
    ax.set_xlabel(r'$\psi$')

    fig.legend()
    fig.savefig(name)


if __name__ == "__main__":

    t = np.logspace(2, 9, 200)
    thV = 0.5
    thC = 0.1
    thW = 0.4
    E0 = 1.0e52
    n0 = 1.0e-2
    p = 2.3
    jetType = 0
    seg = 'D'

    umax = 10.0
    umin = 3.0
    Er = 1.0e55
    k = 5
    Mej = 1.0e-8

    fig1, fig2 = plot_lc_jet(t, 0.25, thC, thW, E0, n0, p, jetType, seg)
    fig3, fig4 = plot_lc_jet(t, 0.45, thC, thW, E0, n0, p, jetType, seg)
    fig5, fig6 = plot_lc_jet(t, 0.7, thC, thW, E0, n0, p, jetType, seg)
    fig7, fig8 = plot_lc_jet(t, 1.0, thC, thW, E0, n0, p, jetType, seg)

    # fig = makeNicePlot(t, thV, thC, thW, E0, n0, p, jetType, seg)

    # fig9, fig10 = plot_lc_cocoon(t, umax, umin, Er, 1, Mej, n0, p, 3, seg)
    # fig11, fig12 = plot_lc_cocoon(t, umax, umin, Er, 2, Mej, n0, p, 3, seg)
    # fig13, fig14 = plot_lc_cocoon(t, umax, umin, Er, 5, Mej, n0, p, 3, seg)
    # fig15, fig16 = plot_lc_cocoon(t, umax, umin, Er, 10, Mej, n0, p, 3, seg)

    if fig1 is not None:
        fig1.savefig("lc_pl_thV_0p25.png")
    if fig2 is not None:
        fig2.savefig("lc_decomp_thV_0p25.png")

    if fig3 is not None:
        fig3.savefig("lc_pl_thV_0p45.png")
    if fig4 is not None:
        fig4.savefig("lc_decomp_thV_0p45.png")

    if fig5 is not None:
        fig5.savefig("lc_pl_thV_0p70.png")
    if fig6 is not None:
        fig6.savefig("lc_decomp_thV_0p70.png")

    if fig7 is not None:
        fig7.savefig("lc_pl_thV_1p00.png")
    if fig8 is not None:
        fig8.savefig("lc_decomp_thV_1p00.png")

    # if fig9 is not None:
    #     fig9.savefig("lc_pl_k_01.png")

    # if fig11 is not None:
    #     fig11.savefig("lc_pl_k_02.png")

    # if fig13 is not None:
    #     fig13.savefig("lc_pl_k_05.png")

    # if fig15 is not None:
    #     fig15.savefig("lc_pl_k_10.png")

    plotSlopes(p, -2, "slopesOn.png")
    plotSlopes(p, -1, "slopesStruct.png")
    plotG('struct_g_gaussian', 0)
    plotG('struct_g_powerlaw', 4)
    plotG('struct_g_gaussian_scaled', 0, 1)
    plotG('struct_g_powerlaw_scaled', 4, 1)

    plt.show()
