import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import grbpy as grb

dataDir = "data/"


def processBoxfitFile(filename):

    t, nu, Fnu = np.loadtxt(filename, unpack=True, usecols=[1, 2, 3],
                            delimiter=',')

    Eiso = 0.0
    n0 = 0.0
    th0 = 0.0
    thV = 0.0
    p = 0.0
    epse = 0.0
    epsB = 0.0
    dL = 0.0
    z = 0.0
    ksiN = 0.0

    f = open(filename, "r")
    for line in f:
        if len(line) <= 0:
            continue
        words = line.split()
        if len(words) < 3:
            continue
        if words[1] == "theta_0":
            th0 = float(words[3])
        elif words[1] == "E_iso":
            Eiso = float(words[3])
        elif words[1] == "n_0":
            n0 = float(words[3])
        elif words[1] == "theta_obs":
            thV = float(words[3])
        elif words[1] == "p":
            p = float(words[3])
        elif words[1] == "epsilon_B":
            epsB = float(words[3])
        elif words[1] == "epsilon_E":
            epse = float(words[3])
        elif words[1] == "ksi_N":
            ksiN = float(words[3])
        elif words[1] == "redshift:":
            z = float(words[2])
        elif words[1] == "luminosity" and words[2] == "distance:":
            dL = float(words[3])

    Y = np.array([thV, Eiso, th0, th0, 0.0, 0.0, 0.0, 0.0, n0, p,
                  epse, epsB, ksiN, dL])
    jetType = -1
    specType = 0

    return t, nu, Fnu, jetType, specType, Y, z


def compareAll():

    NC = 3
    NV = 6
    NN = 3

    colors = ['C0', 'C1', 'C2', 'C3', 'C4']

    for i in range(NC):
        for j in range(NV):
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            thV = 0.0
            thC = 0.0
            for k in range(NN):
                filename = dataDir + "boxfit{0:d}{1:d}{2:d}".format(i, j, k)
                out = processBoxfitFile(filename)
                t, nu, FnuBF, jT, sT, Y, z = out
                thV = Y[0]
                thC = Y[2]
                Fnu = grb.fluxDensity(t, nu, jT, sT, *Y, z=z)
                ax.plot(t, FnuBF, ls='--', color=colors[k],
                        label=r'BoxFit $\nu=${0:.1e}Hz'.format(nu[0]))
                ax.plot(t, Fnu, ls='-', color=colors[k],
                        label=r'grbpy $\nu=${0:.1e}Hz'.format(nu[0]))
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r'$t$ (s)')
            ax.set_ylabel(r'$F_\nu$ (mJy)')
            # ax.legend(labelspacing=0, fontsize=6)
            ax.legend()
            ax.set_title(r"$\theta_C=${0:.2f}  $\theta_V=${1:.2f}".format(
                         thC, thV))

            fig.tight_layout()
            figname = dataDir + "boxfit_comp_{0:d}{1:d}.png".format(i, j)
            print("Saving " + figname)
            fig.savefig(figname)
            plt.close(fig)


def makeNiceFigure():

    curves = ['100', '130', '102', '132']

    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']

    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(4, 1, hspace=0)
    axF = fig.add_subplot(gs[0:-1, 0])
    axE = fig.add_subplot(gs[-1, 0])

    bfLines = []
    apLines = []

    solid_line = mpl.lines.Line2D([], [], ls='-', color='grey')
    dashed_line = mpl.lines.Line2D([], [], ls='--', color='grey')

    handles = [solid_line, dashed_line]
    labels = [r'\tt{afterglowypy}', r'\tt{BoxFit}']

    for i, code in enumerate(curves):

        filename = dataDir + "boxfit{0:s}".format(code)
        out = processBoxfitFile(filename)
        t, nu, FnuBF, jT, sT, Y, z = out
        thV = Y[0]
        # thC = Y[2]
        Fnu = grb.fluxDensity(t, nu, jT, sT, *Y, z=z, spread=5)
        if code[2] == '0':
            log10nu = 9
        elif code[2] == '0':
            log10nu = 14
        else:
            log10nu = 18
        lineBF, = axF.plot(t, FnuBF, ls='--', color=colors[i])
        lineAP, = axF.plot(t, Fnu, ls='-', color=colors[i])

        label = (r'$\theta_{{\mathrm{{obs}}}}={0:.2f}$ rad'
                 r'  $\nu=10^{{{1:d}}}$Hz').format(thV, log10nu)

        apLines.append(lineAP)
        bfLines.append(lineBF)
        handles.append(lineAP)
        labels.append(label)

        err = (Fnu-FnuBF)[FnuBF > 0]/FnuBF[FnuBF > 0]
        # err = Fnu[FnuBF > 0]/FnuBF[FnuBF > 0]
        axE.plot(t[FnuBF > 0], err, ls='-', color=colors[i])
    axF.set_xscale('log')
    axF.set_yscale('log')
    axF.set_ylabel(r'$F_\nu$ (mJy)')

    axE.set_ylim(-1.0, 1.0)
    # axE.set_ylim(0.25, 4.0)
    axE.set_xscale('log')
    # axE.set_yscale('log')
    axE.set_ylabel(r'Fractional Difference')
    axE.set_xlabel(r'$t$ (s)')
    # ax.legend(labelspacing=0, fontsize=6)
    axF.legend(handles, labels)
    # axF.legend(handles, labels, labelspacing=0)

    fig.tight_layout()
    figname = dataDir + "boxfit_comp.pdf"
    print("Saving " + figname)
    fig.savefig(figname)
    plt.close(fig)


if __name__ == "__main__":

    # compareAll()
    makeNiceFigure()
