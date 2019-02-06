import math
import numpy as np
import matplotlib.pyplot as plt

c = 2.99792458e10
me = 9.1093897e-28
mp = 1.6726231e-24


def gaussian(theta, thC=0.1, thW=0.4, E0=1.0e53, n0=1.0, t0=1.0e7):
    atheta = np.atleast_1d(theta)
    onaxis = theta < thW
    offaxis = theta >= thW
    th = atheta[onaxis]

    E = np.empty(atheta.shape)
    g = np.empty(atheta.shape)

    E[onaxis] = E0/(4*np.pi) * np.exp(-0.5*(th*th)/(thC*thC))
    g[onaxis] = np.sqrt(17*E[onaxis]/(2*n0*mp*c*c*c*c*c)) * math.pow(t0, -1.5)

    if offaxis.any():
        E[offaxis] = 0.0
        g[offaxis] = 1.0

    E[g < 1.0] = 0.0
    g[g < 1.0] = 1.0

    return g, E


def powerlaw(theta, thC=0.1, thW=0.4, E0=3.0e53, n0=1.0, t0=1.0e7, b=3):

    atheta = np.atleast_1d(theta)
    onaxis = theta < thW
    offaxis = theta >= thW
    th = atheta[onaxis]

    E = np.empty(atheta.shape)
    g = np.empty(atheta.shape)

    Th = np.sqrt(1.0 + th*th/(thC*thC))

    E[onaxis] = E0/(4*np.pi) * np.power(Th, -b)
    g[onaxis] = np.sqrt(17*E[onaxis]/(2*n0*mp*c*c*c*c*c)) * math.pow(t0, -1.5)

    if offaxis.any():
        E[offaxis] = 0.0
        g[offaxis] = 1.0

    # E[g < 1.0] = 0.0
    g[g < 1.0] = 1.0

    return g, E


def boostedFireball(theta, gb=3.0, eta0=3.0, E0=1.0e53):

    # returns dE/dOmega scaled to Eiso(0) = E0

    ub = math.sqrt(1.0 + gb*gb)
    vb = ub/gb

    u0 = math.sqrt(1.0 + eta0*eta0)
    v0 = u0 / eta0

    stMax = u0/ub
    if stMax < 1.0:
        thMax = math.asin(stMax)
    else:
        thMax = np.inf

    atheta = np.atleast_1d(theta)

    onaxis = atheta < thMax
    offaxis = atheta >= thMax

    th = atheta[onaxis]
    ct = np.cos(th)
    st = np.sin(th)

    g = np.empty(atheta.shape)
    g[onaxis] = gb * (eta0 + vb*ct*np.sqrt(u0*u0-ub*ub*st*st)
                      ) / (1 + ub*ub*st*st)
    if offaxis.any():
        g[offaxis] = 1.0
    gM0 = gb*eta0*(1+vb*v0)

    E = E0/(4*np.pi) * np.power(g/gM0, 3)
    if offaxis.any():
        E[offaxis] = 0.0

    return g, E


def loadTxt(filename, cols, skiprows, delimiter=None):

    n = len(cols)
    dat = []

    f = open(filename, "r")
    for i, line in enumerate(f):
        if i < skiprows:
            continue
        if delimiter is not None:
            words = line.split(delimiter)
        else:
            words = line.split()
        try:
            x = [float(words[j]) for j in cols]
        except ValueError:
            break
        dat.append(x)
    f.close()

    dat = np.array(dat)

    datT = tuple([dat[:, i].copy() for i in range(n)])

    return datT


def duffell2015Oblate(theta, Etot=1.0e49):

    # data is dE/dOmega, scaled as Etot/M_0 c^2 = 0.024

    filename = "StructureData/duffell2015_data.csv"

    thg, gp = loadTxt(filename, cols=[6, 7], skiprows=2, delimiter=',')
    thE, Ep = loadTxt(filename, cols=[4, 5], skiprows=2, delimiter=',')

    # Msun = 1.989e33
    # c = 2.99792458e10
    # M0 = 1.0e-4 * Msun
    Ep *= Etot/0.024

    g = np.interp(theta, thg, gp, right=1.0)
    E = np.interp(theta, thE, Ep, right=0.0)

    return g, E


def duffell2015Spherical(theta, Etot=1.0e49):

    # data is dE/dOmega

    filename = "StructureData/duffell2015_data.csv"

    thg, gp = loadTxt(filename, cols=[2, 3], skiprows=2, delimiter=',')
    thE, Ep = loadTxt(filename, cols=[0, 1], skiprows=2, delimiter=',')
    Ep *= Etot/0.024

    g = np.interp(theta, thg, gp, right=1.0)
    E = np.interp(theta, thE, Ep, right=0.0)

    return g, E


def morsony2007realg5(theta, Etot=3.0e51):

    # data is E_iso, total energy is 2.66e52 erg

    filename = "StructureData/morsony2007realg5E.csv"
    thp, Ep = np.loadtxt(filename, skiprows=2, unpack=True, delimiter=',')

    thp *= np.pi/180.0
    Ep *= Etot / (2.66e52 * 4*np.pi)

    E = np.interp(theta, thp, Ep, right=0.0)
    g = np.zeros(E.shape)

    return g, E


def morsony2007realg2(theta, Etot=3.0e51):

    # data is E_iso, total energy is 2.66e52 erg

    filename = "StructureData/morsony2007realg2E.csv"
    thp, Ep = np.loadtxt(filename, skiprows=2, unpack=True, delimiter=',')

    thp *= np.pi/180.0
    Ep *= Etot / (2.66e52 * 4*np.pi)

    E = np.interp(theta, thp, Ep, right=0.0)
    g = np.zeros(E.shape)

    return g, E


def morsony2007powg5(theta, Etot=3.0e51):

    # data is E_iso, total energy is 2.66e52 erg

    filename = "StructureData/morsony2007powg5E.csv"
    thp, Ep = np.loadtxt(filename, skiprows=2, unpack=True, delimiter=',')

    thp *= np.pi/180.0
    Ep *= Etot / (2.66e52 * 4*np.pi)

    E = np.interp(theta, thp, Ep, right=0.0)
    g = np.zeros(E.shape)

    return g, E


def morsony2007powg2(theta, Etot=3.0e53):

    # data is E_iso, total energy is 2.66e52 erg

    filename = "StructureData/morsony2007powg2E.csv"
    thp, Ep = np.loadtxt(filename, skiprows=2, unpack=True, delimiter=',')

    thp *= np.pi/180.0
    Ep *= Etot / (2.66e52 * 4*np.pi)

    E = np.interp(theta, thp, Ep, right=0.0)
    g = np.zeros(E.shape)

    return g, E


def kathirgamaraju2017(theta):

    filename = "StructureData/kathirgamaraju2017g.csv"
    thp, gp = np.loadtxt(filename, skiprows=2, unpack=True, delimiter=',')

    thp *= np.pi/180.0

    g = np.interp(theta, thp, gp, right=1.0)
    E = np.zeros(g.shape)

    return g, E


def margutti2018(theta):

    # data is dE/dOmega

    filenameg = "StructureData/margutti2018g.csv"
    filenameE = "StructureData/margutti2018E.csv"
    thgp, gp = np.loadtxt(filenameg, skiprows=2, unpack=True, delimiter=',')
    thEp, Ep = np.loadtxt(filenameE, skiprows=2, unpack=True, delimiter=',')

    thgp *= np.pi/180.0
    thEp *= np.pi/180.0

    g = np.interp(theta, thgp, gp, right=1.0)
    E = np.interp(theta, thEp, Ep, right=0.0)

    return g, E


def mizuta2009(theta, E0=1.0e52):

    # data is dE/dOmega, scaled to 1 at theta=0

    filename = "StructureData/mizuta2009E.csv"
    thp, Ep = np.loadtxt(filename, skiprows=2, unpack=True, delimiter=',')

    Ep *= E0

    thp *= np.pi/180.0

    E = np.interp(theta, thp, Ep, right=0.0)
    g = np.zeros(E.shape)

    return g, E


def lazzati2017(theta):

    # data is E_iso

    filenameg = "StructureData/lazzati2017g.csv"
    filenameE = "StructureData/lazzati2017E.csv"
    thgp, gp = np.loadtxt(filenameg, skiprows=2, unpack=True, delimiter=',')
    thEp, Ep = np.loadtxt(filenameE, skiprows=2, unpack=True, delimiter=',')

    thgp *= np.pi/180.0
    thEp *= np.pi/180.0

    Ep /= 4.0*np.pi

    g = np.interp(theta, thgp, gp, right=1.0)
    E = np.interp(theta, thEp, Ep, right=0.0)

    return g, E


def aloy2005(theta):

    # data is E_iso

    filenameE = "StructureData/Aloy05_B01_500ms_data.csv"
    thEp, Ep = np.loadtxt(filenameE, unpack=True, delimiter=',')

    thEp *= np.pi/180.0
    Ep /= 4.0*np.pi

    g = np.zeros(theta.shape)
    E = np.interp(theta, thEp, Ep, right=0.0)

    return g, E


if __name__ == "__main__":

    print("begin")

    theta = np.linspace(0.0, 0.25*np.pi, 100)
    deg = 180.0/np.pi
    rad = np.pi/180.0

    gG, EG = gaussian(theta, thC=6*rad, thW=15*rad, E0=3e52)
    gPL, EPL = powerlaw(theta, thW=60*deg, thC=4*deg, b=4.0, E0=1e52)
    gBF0303, EBF0303 = boostedFireball(theta)
    gBF1003, EBF1003 = boostedFireball(theta, gb=10)
    gBF0310, EBF0310 = boostedFireball(theta, eta0=10)
    gBF1010, EBF1010 = boostedFireball(theta, gb=10, eta0=10)
    gD15s, ED15s = duffell2015Spherical(theta)
    gD15o, ED15o = duffell2015Oblate(theta)
    gM07r5, EM07r5 = morsony2007realg5(theta)
    gM07r2, EM07r2 = morsony2007realg2(theta)
    gM07p5, EM07p5 = morsony2007powg5(theta)
    gM07p2, EM07p2 = morsony2007powg2(theta)
    gK17, EK17 = kathirgamaraju2017(theta)
    gM18, EM18 = margutti2018(theta)
    gM09, EM09 = mizuta2009(theta)
    gL17, EL17 = lazzati2017(theta)
    gA05, EA05 = aloy2005(theta)

    print("Plotting")
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))

    clr = ["k", "grey", "tab:blue", "tab:orange", "tab:green", "tab:red",
                "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan"]
    ls = ['-', '--', ':', '-.']

    ax[0].plot(theta*deg, gG, label="Gaussian", lw=4.0,
                                    color=clr[0], ls=ls[0])
    ax[0].plot(theta*deg, gPL, label="Power law", lw=4.0,
               color=clr[1], ls=ls[0])
    # ax[0].plot(theta*deg, gBF0303,
    #            label="Boosted Fireball $\gamma_B=3$ $\eta_0=3$",
    #                                color=clr[2], ls=ls[0])
    # ax[0].plot(theta*deg, gBF0310,
    #            label="Boosted Fireball $\gamma_B=3$ $\eta_0=10$",
    #                                color=clr[2], ls=ls[1])
    ax[0].plot(theta*deg, gBF1003,
               label="Boosted Fireball $\gamma_B=10$ $\eta_0=3$",
               color=clr[2], ls=ls[0])
    # ax[0].plot(theta*deg, gBF1010,
    #            label="Boosted Fireball $\gamma_B=10$ $\eta_0=10$",
    #                                color=clr[2], ls=ls[3])
    # ax[0].plot(theta*deg, gD15s, label="Duffell 2015 Spherical")
    ax[0].plot(theta*deg, gD15o, label="Duffell 2015 Oblate",
               color=clr[3], ls=ls[0])
    ax[0].plot(theta*deg, gM07r5, label="Morsony 2007 realg5",
               color=clr[4], ls=ls[0])
    # ax[0].plot(theta*deg, gM07r2, label="Morsony 2007 realg2")
    # ax[0].plot(theta*deg, gM07p5, label="Morsony 2007 powg5")
    ax[0].plot(theta*deg, gM07p2, label="Morsony 2007 powg2",
               color=clr[4], ls=ls[1])
    ax[0].plot(theta*deg, gK17, label="Kathirgamaraju 2017",
               color=clr[5], ls=ls[0])
    ax[0].plot(theta*deg, gM18, label="Margutti 2018",
               color=clr[6], ls=ls[0])
    ax[0].plot(theta*deg, gM09, label="Mizuta 2009",
               color=clr[7], ls=ls[0])
    ax[0].plot(theta*deg, gL17, label="Lazzati 2017",
               color=clr[8], ls=ls[0])
    ax[1].plot(theta*deg, EG, label="Gaussian", lw=4.0,
               color=clr[0], ls=ls[0])
    ax[1].plot(theta*deg, EPL, label="Power law", lw=4.0,
               color=clr[1], ls=ls[0])
    # ax[1].plot(theta*deg, EBF0303,
    #           label="Boosted Fireball $\gamma_B=3$ $\eta_0=3$",
    #                                color=clr[2], ls=ls[0])
    # ax[1].plot(theta*deg, EBF0310,
    #            label="Boosted Fireball $\gamma_B=3$ $\eta_0=10$",
    #                                color=clr[2], ls=ls[1])
    ax[1].plot(theta*deg, EBF1003,
               label="Boosted Fireball $\gamma_B=10$ $\eta_0=3$",
               color=clr[2], ls=ls[0])
    # ax[1].plot(theta*deg, EBF1010,
    #            label="Boosted Fireball $\gamma_B=10$ $\eta_0=10$",
    #                                color=clr[2], ls=ls[3])
    # ax[1].plot(theta*deg, ED15s, label="Duffell 2015 Spherical")
    ax[1].plot(theta*deg, ED15o, label="Duffell 2015 Oblate",
               color=clr[3], ls=ls[0])
    ax[1].plot(theta*deg, EM07r5, label="Morsony 2007 realg5",
               color=clr[4], ls=ls[0])
    # ax[1].plot(theta*deg, EM07r2, label="Morsony 2007 realg2")
    # ax[1].plot(theta*deg, EM07p5, label="Morsony 2007 powg5")
    ax[1].plot(theta*deg, EM07p2, label="Morsony 2007 powg2",
               color=clr[4], ls=ls[1])
    ax[1].plot(theta*deg, EK17, label="Kathirgamaraju 2017",
               color=clr[5], ls=ls[0])
    ax[1].plot(theta*deg, EM18, label="Margutti 2018",
               color=clr[6], ls=ls[0])
    ax[1].plot(theta*deg, EM09, label="Mizuta 2009",
               color=clr[7], ls=ls[0])
    ax[1].plot(theta*deg, EL17, label="Lazzati 2017",
               color=clr[8], ls=ls[0])

    ax[0].legend()

    ax[0].set_xlabel(r'$\theta$ $({}^\circ)$')
    ax[0].set_ylabel(r'$\Gamma$')
    ax[1].set_xlabel(r'$\theta$ $({}^\circ)$')
    ax[1].set_ylabel(r'$dE/d\Omega$ (erg/sr)')
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    fig.tight_layout()

    print("Saving structCompE.pdf")
    fig.savefig("structCompE.pdf")

    print("Plotting")
    figE, axE = plt.subplots(1, 1, figsize=(12, 9))

    clr = ["k", "grey", "tab:blue", "tab:orange", "tab:green", "tab:red",
                "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan"]
    ls = ['-', '--', ':', '-.']

    axE.plot(theta*deg, EG, label="Gaussian", lw=4.0,
             color=clr[0], ls=ls[0])
    axE.plot(theta*deg, EPL, label="Power law", lw=4.0,
             color=clr[1], ls=ls[0])
    axE.plot(theta*deg, EBF1003,
             label="Boosted Fireball $\gamma_B=10$ $\eta_0=3$",
             color=clr[2], ls=ls[0])
    axE.plot(theta*deg, ED15o, label="Duffell 2015 Oblate",
             color=clr[3], ls=ls[0])
    axE.plot(theta*deg, EM07r5, label="Morsony 2007 realg5",
             color=clr[4], ls=ls[0])
    axE.plot(theta*deg, EM07p2, label="Morsony 2007 powg2",
             color=clr[4], ls=ls[1])
    axE.plot(theta*deg, EM18, label="Margutti 2018",
             color=clr[6], ls=ls[0])
    axE.plot(theta*deg, EM09, label="Mizuta 2009",
             color=clr[7], ls=ls[0])
    axE.plot(theta*deg, EL17, label="Lazzati 2017",
             color=clr[8], ls=ls[0])

    axE.legend(loc='upper right', fontsize=12)

    axE.set_xlabel(r'$\theta$ $({}^\circ)$', fontsize=16)
    axE.set_ylabel(r'$dE/d\Omega$ (erg/sr)', fontsize=16)
    axE.set_yscale('log')
    axE.set_ylim(3.0e48, 1.0e53)
    figE.tight_layout()

    print("Saving structCompE.pdf")
    figE.savefig("structCompE.pdf")

    print("Plotting")
    figE2, axE2 = plt.subplots(1, 1, figsize=(6, 4.5))

    gG, EG = gaussian(theta, thC=6*rad, thW=13*rad, E0=3e52)
    gPL, EPL = powerlaw(theta, thW=60*rad, thC=4*rad, b=4.5, E0=1e52)

    clr = ["k", "grey", "tab:blue", "tab:orange", "tab:green", "tab:red",
                "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan"]
    ls = ['-', '--', ':', '-.']

    gaussNorm = 3.0
    plNorm = 1.0

    axE2.plot(theta*deg, EG*gaussNorm/EG[0],
              # label=r"Gaussian $\theta_C=6^\circ$ $\theta_W=12^\circ$",
              label=r"Gaussian",
              lw=4.0, color=clr[1], ls=ls[0], alpha=0.65)
    axE2.plot(theta*deg, EPL*plNorm/EPL[0],
              # label=r"Power law $\theta_C=4^\circ$ $b=4.5$",
              label=r"Power law",
              lw=4.0, color=clr[1], ls=ls[1], alpha=0.65)
    axE2.plot(theta*deg, EA05*gaussNorm/EA05[0], label="Aloy 2005",
              color=clr[2], ls=ls[0])
    axE2.plot(theta*deg, EM09*plNorm/EM09[0], label="Mizuta 2009",
              color=clr[3], ls=ls[0])
    axE2.plot(theta*deg, EBF1003*plNorm/EBF1003[0],
              # label=r"Duffell 2013 $\gamma_B=10$ $\eta_0=3$",
              label=r"Duffell 2013",
              color=clr[4], ls=ls[0])
    axE2.plot(theta*deg, EL17*plNorm/EL17[0], label="Lazzati 2017",
              color=clr[5], ls=ls[0])
    axE2.plot(theta*deg, EM18*gaussNorm/EM18[0], label="Margutti 2018",
              color=clr[6], ls=ls[0])
    # axE2.plot(theta*deg, ED15o, label="Duffell 2015 Oblate",
    #           color=clr[3], ls=ls[0])
    # axE2.plot(theta*deg, EM07r5, label="Morsony 2007 realg5",
    #           color=clr[4], ls=ls[0])
    # axE2.plot(theta*deg, EM07p2, label="Morsony 2007 powg2",
    #           color=clr[4], ls=ls[1])

    axE2.legend(loc='upper right', fontsize=13)

    axE2.set_xlabel(r'$\theta$ (degrees)', fontsize=18)
    axE2.set_ylabel(r'$dE/d\Omega$ (arbitrary units)', fontsize=18)
    axE2.set_yscale('log')
    axE2.set_ylim(1.0e-3, 1.0e1)
    axE2.set_xlim(0.0, 30.0)
    axE2.tick_params(labelsize=16)
    figE2.tight_layout()

    print("Saving structCompE2.pdf")
    figE2.savefig("structCompE2.pdf")

    print("Plotting")
    figEc, axEc = plt.subplots(1, 1, figsize=(3.5, 3.0))

    gG, EG = gaussian(theta, thC=6*rad, thW=12*rad, E0=3e52)
    gPL, EPL = powerlaw(theta, thW=60*rad, thC=4*rad, b=4.5, E0=1e52)

    clr = ["k", "grey", "tab:blue", "tab:orange", "tab:green", "tab:red",
                "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan"]
    ls = ['-', '--', ':', '-.']

    gaussNorm = 3.0
    plNorm = 1.0

    axEc.plot(theta*deg, EG*gaussNorm/EG[0],
              # label=r"Gaussian $\theta_C=6^\circ$ $\theta_W=12^\circ$",
              label=r"Gaussian",
              lw=2.5, color=clr[1], ls=ls[0], alpha=0.65)
    axEc.plot(theta*deg, EPL*plNorm/EPL[0],
              # label=r"Power law $\theta_C=4^\circ$ $b=4.5$",
              label=r"Power Law",
              lw=2.5, color=clr[1], ls=ls[1], alpha=0.8)
    axEc.plot(theta*deg, EA05*gaussNorm/EA05[0], label="Aloy 2005",
              color=clr[2], ls=ls[0])
    axEc.plot(theta*deg, EM09*plNorm/EM09[0], label="Mizuta 2009",
              color=clr[3], ls=ls[0])
    axEc.plot(theta*deg, EBF1003*plNorm/EBF1003[0],
              # label=r"Duffell 2013 $\gamma_B=10$ $\eta_0=3$",
              label=r"Duffell 2013",
              color=clr[4], ls=ls[0])
    axEc.plot(theta*deg, EL17*plNorm/EL17[0], label="Lazzati 2017",
              color=clr[5], ls=ls[0])
    axEc.plot(theta*deg, EM18*gaussNorm/EM18[0], label="Margutti 2018",
              color=clr[6], ls=ls[0])
    # axEc.plot(theta*deg, ED15o, label="Duffell 2015 Oblate",
    #           color=clr[3], ls=ls[0])
    # axEc.plot(theta*deg, EM07r5, label="Morsony 2007 realg5",
    #           color=clr[4], ls=ls[0])
    # axEc.plot(theta*deg, EM07p2, label="Morsony 2007 powg2",
    #           color=clr[4], ls=ls[1])

    axEc.legend(loc='upper right', fontsize=8)

    axEc.set_xlabel(r'$\theta$ (degrees)')
    axEc.set_ylabel(r'$E_{{\rm{iso}}}$ (arbitrary units)')
    axEc.set_yscale('log')
    axEc.set_ylim(1.0e-3, 1.0e1)
    axEc.set_xlim(0.0, 30.0)
    # axEc.tick_params(labelsize=16)
    figEc.tight_layout()

    print("Saving structCompEc.pdf")
    figEc.savefig("structCompEc.pdf")

    # print("Showing")
    # plt.show()
