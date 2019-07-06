import sys
import math
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import paperPlots as pp


def calcG(alpha, p, s, regime):
    beta, al1, D, s1, s2, p = pp.get_powers(regime, p)
    al2 = 3+s2

    """
    if regime is 'D':
        return (-2-8*alpha+3*s) / (alpha-3-s)
    elif regime is 'E':
        return (-14./3.+3*s-8*alpha) / (alpha-3-s)
    elif regime is 'F':
        return (-8+3*s-8*alpha) / (alpha-3-s)
    elif regime is 'G':
        return (-6*p+3*s-8*alpha) / (alpha-3-s)
    elif regime is 'H':
        return (-6*p-2+3*s-8*alpha) / (alpha-3-s)
    """

    return (8*alpha + 3*al1 - 3*s) / (al2 - alpha)
    # return ((8-4*s/D)*alpha + 3*al1 - 3*s*al1/(2*D)) / (al2 - alpha)


def calcTh(g, jetModel, Y):

    thV = Y[0]
    thC = Y[2]
    # thW = Y[3]
    b = Y[4]

    if jetModel is 0:
        thS = 0.5*thV*(1.0-np.sqrt(1.0-4*g*thC*thC/(thV*thV)))
    elif jetModel is 4:
        gob = g/b
        # des = 4*thC*thC*gob*(1+gob)/(thV*thV)
        des = 4*thC*thC*g*(1+gob)/(thV*thV)
        thS = 0.5*thV*(1.0+np.sqrt(1.0-des)) / (1+gob)

    return thS


def geff(jetModel, thV, thC, thW, b):
    if jetModel is 0:
        return 0.25*thV*thV/(thC*thC)
    elif jetModel is 4:
        # return 2*thV*thV / (thW*thW)
        # return 2*thV*thV / (4*thC*thC + thV*thV)
        return thV*thV / (4*thC*thC + thV*thV/b)


def f_g(jetModel, thV, thC, thW, b, th):
    if th > thW:
        mdlogEdth = 0.0
    elif jetModel is 0:
        mdlogEdth = th/(thC*thC)
    elif jetModel is 4:
        Th = np.sqrt(1.0 + th*th/(b*thC*thC))
        mdlogEdth = th/(thC*thC*Th*Th)

    return 2*math.tan(0.5*math.fabs(thV-th)) * mdlogEdth


def makeSlopePlots(filename, jetModel):

    f = h5.File(filename, "r")
    thW = f['thW'][...]
    thV = f['thV'][...]
    thC = f['thC'][...]
    alOffN = f['alOffN'][...]
    alStructN = f['alStructN'][...]
    alStructN2 = f['alStructN2'][...]
    alSNlog = f['alStructNAveLog'][...]
    alSNlin = f['alStructNAveLin'][...]
    alPostN = f['alPostN'][...]
    FbN = f['FbN'][...]
    Y = f['Y'][...]
    f.close()

    regime = filename[-4]

    NW = len(thW)
    NV = len(thV)
    NC = len(thC)

    color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
             'tab:purple', 'tab:brown', 'tab:pink', 'tab:grey',
             'tab:olive', 'tab:cyan']
    ls = ['-', '--', '-.', ':']

    alOffA = np.empty(alOffN.shape)
    alStructA = np.empty(alStructN.shape)
    alPreA = np.empty(alPostN.shape)
    alPostA = np.empty(alPostN.shape)
    gNP = np.empty(alStructN.shape)
    gNP2 = np.empty(alStructN.shape)
    gNLog = np.empty(alStructN.shape)
    gNLin = np.empty(alStructN.shape)
    gA = np.empty(alStructN.shape)
    thNP = np.empty(alStructN.shape)
    thNP2 = np.empty(alStructN.shape)
    thNLog = np.empty(alStructN.shape)
    thNLin = np.empty(alStructN.shape)
    thA = np.empty(alStructN.shape)

    for k in range(NW):
        for j in range(NV):
            for i in range(NC):
                Y[0] = thV[j]
                Y[2] = thC[i]
                Y[3] = thW[k]

                b = Y[4]
                p = Y[9]
                s = 1.0

                gNP[k, j, i] = calcG(alStructN[k, j, i], p, s, regime)
                gNP2[k, j, i] = calcG(alStructN2[k, j, i], p, s, regime)
                gNLog[k, j, i] = calcG(alSNlog[k, j, i], p, s, regime)
                gNLin[k, j, i] = calcG(alSNlin[k, j, i], p, s, regime)
                # gA[k, j, i] = geff(jetModel, thV[j], thC[i], thW[k], b)

                thNP[k, j, i] = calcTh(gNP[k, j, i], jetModel, Y)
                thNP2[k, j, i] = calcTh(gNP2[k, j, i], jetModel, Y)
                thNLog[k, j, i] = calcTh(gNLog[k, j, i], jetModel, Y)
                thNLin[k, j, i] = calcTh(gNLin[k, j, i], jetModel, Y)

                if jetModel == 0:
                    thS = min(0.5*thW[k], 0.5*thV[j])
                else:
                    thS = min(0.5*thW[k], 0.5*thV[j],
                              0.2*thV[j]+2.0/3.0*thC[i])

                    thS = thV[j] / (1 + np.sqrt(
                                    1+thV[j]*thV[j]/(16*b*thC[i]*thC[i])))
                    thS = min(0.5*thV[j],
                              thV[j] * np.power(thV[j]/thC[i], -0.5))

                    thS = thV[j] / (0.66 + 0.55*np.sqrt(thV[j]/thC[i]))

                    thS = thV[j] / np.sqrt(1.9 + 2.5*np.power(b, -1.7)
                                           + (0.46-0.9*np.power(b, -1.3))
                                           * thV[j]/thC[i])
                    thS = thV[j] / np.sqrt(1.8 + 2.1*np.power(b, -1.25)
                                           + (0.49-0.86*np.power(b, -1.15))
                                           * thV[j]/thC[i])
                    # thS = thV[j] / np.sqrt(2 + 2*np.power(b, -1)
                    #                        + (0.5-1.0*np.power(b, -1))
                    #                        * thV[j]/thC[i])

                    # thS = 0.67*thV[j]
                    # thS = 0.5*thV[j]
                    # thS = thC[i]*thC[i]/(b*thV[j]) * max(1.0,
                    #            (thV[j]/thW[k])**2)
                    # thS = thC[i]

                thA[k, j, i] = thS
                gA[k, j, i] = f_g(jetModel, thV[j], thC[i], thW[k], b, thS)

                alOffA[k, j, i] = pp.calcSlopeOffaxis(jetModel, Y, regime)
                alStructA[k, j, i] = pp.calcSlopeStruct(jetModel, Y, regime,
                                                        gA[k, j, i], s)
                alPreA[k, j, i] = pp.calcSlopePre(jetModel, Y, regime)
                alPostA[k, j, i] = pp.calcSlopePost(jetModel, Y, regime)

    f = h5.File("../thetaFit.h5", "a")
    group = f.require_group(regime)
    f.require_dataset("thV", shape=thV.shape, dtype=thV.dtype, data=thV)
    f.require_dataset("thC", shape=thC.shape, dtype=thC.dtype, data=thC)
    f.require_dataset("thW", shape=thW.shape, dtype=thW.dtype, data=thW)
    dset = group.require_dataset(str(b), shape=(NW, NC, NV), dtype=thNP2.dtype)
    dset[:, :, :] = np.swapaxes(thNP2, 1, 2)
    f.close()

    f = h5.File("../gFit.h5", "a")
    group = f.require_group(regime)
    f.require_dataset("thV", shape=thV.shape, dtype=thV.dtype, data=thV)
    f.require_dataset("thC", shape=thC.shape, dtype=thC.dtype, data=thC)
    f.require_dataset("thW", shape=thW.shape, dtype=thW.dtype, data=thW)
    dset = group.require_dataset(str(b), shape=(NW, NC, NV), dtype=gNP2.dtype)
    dset[:, :, :] = np.swapaxes(gNP2, 1, 2)
    f.close()

    figOff, axOff = plt.subplots(1, 1)
    figStructP, axStructP = plt.subplots(1, 1)
    figStructP2, axStructP2 = plt.subplots(1, 1)
    figStructLin, axStructLin = plt.subplots(1, 1)
    figStructLog, axStructLog = plt.subplots(1, 1)
    figPost, axPost = plt.subplots(1, 1)
    figGP, axGP = plt.subplots(1, 1)
    figGP2, axGP2 = plt.subplots(1, 1)
    figGLog, axGLog = plt.subplots(1, 1)
    figGLin, axGLin = plt.subplots(1, 1)
    figTP, axTP = plt.subplots(1, 1)
    figTP2, axTP2 = plt.subplots(1, 1)
    figTLog, axTLog = plt.subplots(1, 1)
    figTLin, axTLin = plt.subplots(1, 1)
    figF, axF = plt.subplots(1, 1)
    for k in range(0, NW, 1):
        for i in range(0, NC, 1):
            axOff.plot(thV, alOffA[k, :, i], color=color[k], ls=ls[i],
                       lw=2, alpha=0.5)
            axStructP.plot(thV, alStructA[k, :, i], color=color[k], ls=ls[i],
                           lw=2, alpha=0.5)
            axStructP2.plot(thV, alStructA[k, :, i], color=color[k], ls=ls[i],
                            lw=2, alpha=0.2)
            axStructLog.plot(thV, alStructA[k, :, i], color=color[k], ls=ls[i],
                             lw=2, alpha=0.5)
            axStructLin.plot(thV, alStructA[k, :, i], color=color[k], ls=ls[i],
                             lw=2, alpha=0.5)
            axStructP.plot(thV, alPreA[k, :, i], color=color[k], ls=ls[i],
                           lw=3, alpha=0.5)
            axStructP2.plot(thV, alPreA[k, :, i], color=color[k], ls=ls[i],
                            lw=3, alpha=0.2)
            axStructLog.plot(thV, alPreA[k, :, i], color=color[k], ls=ls[i],
                             lw=3, alpha=0.5)
            axStructLin.plot(thV, alPreA[k, :, i], color=color[k], ls=ls[i],
                             lw=3, alpha=0.5)
            axPost.plot(thV, alPostA[k, :, i], color=color[k], ls=ls[i],
                        lw=2, alpha=0.5)

            axOff.plot(thV, alOffN[k, :, i], color=color[k], ls=ls[i])
            axStructP.plot(thV, alStructN[k, :, i], color=color[k], ls=ls[i])
            axStructP2.plot(thV, alStructN2[k, :, i], color=color[k], ls=ls[i])
            axStructLog.plot(thV, alSNlog[k, :, i], color=color[k], ls=ls[i])
            axStructLin.plot(thV, alSNlin[k, :, i], color=color[k], ls=ls[i])
            axPost.plot(thV, alPostN[k, :, i], color=color[k], ls=ls[i])

            axGP.plot(thV/thC[i], gA[k, :, i], color=color[k], ls=ls[i],
                      lw=2, alpha=0.5)
            axGP2.plot(thV/thC[i], gA[k, :, i], color=color[k], ls=ls[i],
                       lw=2, alpha=0.2)
            axGLog.plot(thV, gA[k, :, i], color=color[k], ls=ls[i],
                        lw=2, alpha=0.5)
            axGLin.plot(thV, gA[k, :, i], color=color[k], ls=ls[i],
                        lw=2, alpha=0.5)

            axGP.plot(thV/thC[i], gNP[k, :, i], color=color[k], ls=ls[i])
            axGP2.plot(thV/thC[i], gNP2[k, :, i], color=color[k], ls=ls[i])
            axGLog.plot(thV, gNLog[k, :, i], color=color[k], ls=ls[i])
            axGLin.plot(thV, gNLin[k, :, i], color=color[k], ls=ls[i])

            axTP.plot(thV, thA[k, :, i], color=color[k], ls=ls[i],
                      lw=2, alpha=0.5)
            axTP2.plot(thV/thC[i], (thV / thA[k, :, i])**2,
                       color=color[k], ls=ls[i], lw=2, alpha=0.2)
            axTP2.plot(thV/thC[i], (thV / (0.5*thV))**2,
                       color='grey', lw=1)
            axTLog.plot(thV, thA[k, :, i], color=color[k], ls=ls[i],
                        lw=2, alpha=0.5)
            axTLin.plot(thV, thA[k, :, i], color=color[k], ls=ls[i],
                        lw=2, alpha=0.5)
            axTP.plot(thV, thNP[k, :, i], color=color[k], ls=ls[i])
            axTP2.plot(thV/thC[i], (thV / thNP2[k, :, i])**2,
                       color=color[k], ls=ls[i])
            axTLog.plot(thV, thNLog[k, :, i], color=color[k], ls=ls[i])
            axTLin.plot(thV, thNLin[k, :, i], color=color[k], ls=ls[i])

            axF.plot(thV, FbN[k, :, i], color=color[k], ls=ls[i])

    for ax in [axOff, axStructP, axStructP2, axStructLin, axStructLog, axPost]:
        ax.set_xlabel(r'$\theta_V$')
        ax.set_ylabel(r'$\alpha$')
        ax.set_ylim(-2, 3)
        # ax.set_xscale('log')

    for ax in [axGP, axGP2, axGLog, axGLin]:
        ax.set_xlabel(r'$\theta_V / \theta_C$')
        ax.set_ylabel(r'$g$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        # ax.set_ylim(0, 15)
        ax.set_ylim(0.3, 100)
        ax.set_xlim(0.3, None)

    for ax in [axTP, axTP2, axTLog, axTLin]:
        ax.set_xlabel(r'$\theta_V$')
        ax.set_ylabel(r'$\theta_*$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        # ax.set_ylim(0, 0.6)
        # ax.set_xlim(0.0, 1.0)
        # ax.set_xlim(0.1, 1.0)
        ax.set_xlim(0.5, 50.0)
        # ax.set_ylim(0.04, 0.5)
        ax.set_ylim(1.0, 30.0)

    axF.set_xlabel(r'$\theta_V$')
    axF.set_ylabel(r'$F_\nu(t_b)$')
    axF.set_xscale('log')
    axF.set_yscale('log')

    pp.save(figOff, "alOff_{0:s}_thV.png".format(regime))
    pp.save(figStructP, "alStructP_{0:s}_thV.png".format(regime))
    pp.save(figStructP2, "alStructP2_{0:s}_thV.png".format(regime))
    pp.save(figStructLin, "alStructLin_{0:s}_thV.png".format(regime))
    pp.save(figStructLog, "alStructLog_{0:s}_thV.png".format(regime))
    pp.save(figPost, "alPost_{0:s}_thV.png".format(regime))
    pp.save(figGP, "alGP_{0:s}_thV.png".format(regime))
    pp.save(figGP2, "alGP2_{0:s}_thV.png".format(regime))
    pp.save(figGLog, "alGLog_{0:s}_thV.png".format(regime))
    pp.save(figGLin, "alGLin_{0:s}_thV.png".format(regime))
    pp.save(figTP, "alTP_{0:s}_thV.png".format(regime))
    pp.save(figTP2, "alTP2_{0:s}_thV.png".format(regime))
    pp.save(figTLog, "alTLog_{0:s}_thV.png".format(regime))
    pp.save(figTLin, "alTLin_{0:s}_thV.png".format(regime))
    pp.save(figF, "Fb_{0:s}_thV.png".format(regime))
    plt.close(figOff)
    plt.close(figStructP)
    plt.close(figStructP2)
    plt.close(figStructLin)
    plt.close(figStructLog)
    plt.close(figPost)
    plt.close(figGP)
    plt.close(figGP2)
    plt.close(figGLog)
    plt.close(figGLin)
    plt.close(figTP)
    plt.close(figTP2)
    plt.close(figTLog)
    plt.close(figTLin)
    plt.close(figF)

    for i in range(NC):

        thS = np.zeros(thV.shape)
        ind = np.empty(thV.shape, dtype=np.bool)

        for j in range(NV):
            count = 0
            for k in range(NW):
                th = thNP2[k, j, i]
                if th < 0.5*thW[k] and th < 0.5*thV[j]:
                    thS[j] += th
                    count += 1
            if count is 0:
                ind[j] = False
            else:
                ind[j] = True
                thS[j] /= count

        if ind.any():
            res = np.polyfit(thV[ind], thS[ind], 1)

            print("b={0:f} thC={1:g} slope={2:g} yint={3:g}".format(
                  b, thC[i], res[0], res[1]))
            res = np.polyfit(np.sqrt(thV[ind]/thC[i]),
                             thV[ind]/thS[ind], 1)

            print("b={0:f} thC={1:g} a1={2:g} a0={3:g}".format(
                  b, thC[i], res[0], res[1]))
            # print("b={0:f} thC={1:g} a2={2:g} a1={3:g} a0={4:g}".format(
            #       b, thC[i], res[0], res[1], res[2]))


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Need a slope file and model number bub!")
        sys.exit()

    filename = sys.argv[1]
    jetModel = int(sys.argv[2])
    makeSlopePlots(filename, jetModel)
