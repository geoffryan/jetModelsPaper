import sys
import numpy as np
import matplotlib.pyplot as plt
import grbpy as grb


def loadXieData(filename):

    f = open(filename, 'r')
    line = f.readline()
    f.close()
    headers = line.split(',')[::2]
    n = len(headers)

    datasets = []

    for i in range(n):
        t, Fnu = np.genfromtxt(filename, skip_header=2, usecols=[2*i, 2*i+1],
                               delimiter=',', unpack=True)
        name = headers[i]

        good = np.isfinite(t) & (t > 0) & np.isfinite(Fnu) & (Fnu > 0)
        t = t[good]
        Fnu = Fnu[good]

        if 'x5 ' in name:
            Fnu /= 5.0
        elif 'x100 ' in name:
            Fnu /= 100.0
        elif 'x1000 ' in name:
            Fnu /= 1000.0

        if '3GHz' in name:
            nu = 3.0e9
        elif '6GHz' in name:
            nu = 6.0e9
        elif '5e14Hz' in name:
            nu = 5.0e14
        else:
            nu = 1000.0 * grb.eV2Hz

        if 'n1e-4' in name:
            Y = np.array([0.34, 9.6e52, 0.15, 0.6, 1.93, 0.0, 0.0, 0.0, 1e-4,
                          2.16, 0.02, 1e-3, 1.0, 1.234e26])
        else:
            Y = np.array([0.3, 9.6e52, 0.15, 0.6, 1.93, 0.0, 0.0, 0.0, 1e-5,
                          2.16, 0.1, 5e-4, 1.0, 1.234e26])

        datasets.append((name, nu, t, Fnu, Y))

    return datasets


def loadLazzatiData(filename):

    f = open(filename, 'r')
    line = f.readline()
    f.close()
    headers = line.split(',')[::2]
    n = len(headers)

    datasets = []

    for i in range(n):
        t, Fnu = np.genfromtxt(filename, skip_header=2, usecols=[2*i, 2*i+1],
                               delimiter=',', unpack=True)
        name = headers[i]

        good = np.isfinite(t) & (t > 0) & np.isfinite(Fnu) & (Fnu > 0)
        t = t[good]
        Fnu = Fnu[good]

        if 'x5 ' in name:
            Fnu /= 5.0
        elif 'x100 ' in name:
            Fnu /= 100.0
        elif 'x1000 ' in name:
            Fnu /= 1000.0

        if '3GHz' in name:
            nu = 3.0e9
        elif '6GHz' in name:
            nu = 6.0e9
        elif '8GHz' in name:
            nu = 8.0e9
        elif '5e14Hz' in name:
            nu = 5.0e14
        else:
            nu = 1000.0 * grb.eV2Hz

        Y = np.array([0.0, 3.0e52, 4.0*np.pi/180.0, 20.0*np.pi/180.0, 4.5,
                      0.0, 0.0, 0.0, 1e-4, 2.3, 1.0e-2, 1.0e-3, 1.0, 1.234e26])
        if ' 4deg' in name:
            Y[0] = 4 * np.pi/180
        elif ' 8deg' in name:
            Y[0] = 8 * np.pi/180
        elif ' 16deg' in name:
            Y[0] = 16 * np.pi/180
        elif ' 32deg' in name:
            Y[0] = 32 * np.pi/180
        elif ' 64deg' in name:
            Y[0] = 64 * np.pi/180

        datasets.append((name, nu, t, Fnu, Y))

    return datasets


def loadLazzatiEtheta(filename):

    thetaDeg, Eiso = np.loadtxt(filename, unpack=True, skiprows=2,
                                delimiter=',')

    theta = thetaDeg * np.pi/180.0
    return theta, Eiso


def makeXiePlot(filename):
    data = loadXieData(filename)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    for i in range(len(data)):
        ax = axes.flat[i]
        name, nud, td, Fnud, Y = data[i]
        t = np.geomspace(td.min(), td.max(), 300)
        nu = np.empty(t.shape)
        nu[:] = nud
        Fnu = grb.fluxDensity(t*grb.day2sec, nu, 5, 0, *Y, spread=False)
        YG = Y.copy()
        YG[2] /= np.sqrt(2)
        FnuG = grb.fluxDensity(t*grb.day2sec, nu, 0, 0, *YG, spread=False)
        YG[2] /= np.sqrt(2)
        Fnu2 = grb.fluxDensity(t*grb.day2sec, nu, 0, 0, *YG, spread=False)
        YG[2] /= np.sqrt(2)
        Fnu3 = grb.fluxDensity(t*grb.day2sec, nu, 0, 0, *YG, spread=False)
        ax.plot(td, Fnud)
        ax.plot(t, Fnu)
        ax.plot(t, FnuG)
        ax.plot(t, Fnu2)
        ax.plot(t, Fnu3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$t$ (d)')
        ax.set_ylabel(r'$F_\nu$ (mJy)')
        ax.set_title(name)

    th = np.linspace(0.0, 0.5*np.pi)
    E1 = np.exp(-np.power(th/0.15, 1.93))
    E2 = np.exp(-0.5*th*th/(0.106**2))
    axes[2, 2].plot(th, E1, color='C1')
    axes[2, 2].plot(th, E2, color='C2')
    axes[2, 2].set_yscale('log')
    axes[2, 2].set_xlabel(r'$\theta$ (rad)')
    axes[2, 2].set_ylabel(r'$E_{\mathrm{iso}}$ (erg)')

    fig.tight_layout()

    print("Saving xie_comp.pdf")
    fig.savefig('xie_comp.pdf')


def makeLazzatiPlot(filename, structFile, figname):
    data = loadLazzatiData(filename)

    theta, E = loadLazzatiEtheta(structFile)

    fig, axes = plt.subplots(2, 3, figsize=(10, 7))

    for i in range(len(data)):
        ax = axes.flat[i]
        name, nud, td, Fnud, Y = data[i]
        t = np.geomspace(td.min(), td.max(), 300)
        nu = np.empty(t.shape)
        nu[:] = nud
        Fnu = grb.fluxDensity(t*grb.day2sec, nu, 4, 0, *Y, spread=False)
        """
        YG = Y.copy()
        YG[2] /= np.sqrt(2)
        FnuG = grb.fluxDensity(t*grb.day2sec, nu, 0, 0, *YG, spread=False)
        YG[2] /= np.sqrt(2)
        Fnu2 = grb.fluxDensity(t*grb.day2sec, nu, 0, 0, *YG, spread=False)
        YG[2] /= np.sqrt(2)
        Fnu3 = grb.fluxDensity(t*grb.day2sec, nu, 0, 0, *YG, spread=False)
        """
        ax.plot(td, Fnud)
        ax.plot(t, Fnu)
        # ax.plot(t, FnuG)
        # ax.plot(t, Fnu2)
        # ax.plot(t, Fnu3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$t$ (d)')
        ax.set_ylabel(r'$F_\nu$ (mJy)')
        ax.set_title(name)

    th = np.linspace(0.0, 0.5*np.pi)
    Epl = Y[1] * np.power(1.0 + th*th/(Y[2]*Y[2]), -0.5*Y[4])
    Epl[th > Y[3]] = 0.0
    axes[1, 2].plot(theta, E, color='C0')
    axes[1, 2].plot(th, Epl, color='C1')
    axes[1, 2].set_yscale('log')
    axes[1, 2].set_xlabel(r'$\theta$ (rad)')
    axes[1, 2].set_ylabel(r'$E_{\mathrm{iso}}$ (erg)')

    fig.tight_layout()

    print("Saving " + figname)
    fig.savefig(figname)


if __name__ == "__main__":

    makeXiePlot(sys.argv[1])
    makeLazzatiPlot(sys.argv[2], sys.argv[4], "lazzati_comp_radio.pdf")
    makeLazzatiPlot(sys.argv[3], sys.argv[4], "lazzati_comp_xray.pdf")

    plt.show()
