import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

f = h5.File("gFit.h5", "r")

thV = f['thV'][...]
thC = f['thC'][...]
thW = f['thW'][...]

b = np.array([float(x) for x in f['D'].keys()])

NV = len(thV)
NC = len(thC)
NW = len(thW)
Nb = len(b)

g_D = np.empty((Nb, NW, NC, NV))
g_E = np.empty((Nb, NW, NC, NV))
g_F = np.empty((Nb, NW, NC, NV))
g_G = np.empty((Nb, NW, NC, NV))
g_H = np.empty((Nb, NW, NC, NV))

for i, bb in enumerate(b):
    g_D[i, :, :, :] = f['D'][str(bb)][...]
    g_E[i, :, :, :] = f['E'][str(bb)][...]
    g_F[i, :, :, :] = f['F'][str(bb)][...]
    g_G[i, :, :, :] = f['G'][str(bb)][...]
    g_H[i, :, :, :] = f['H'][str(bb)][...]
f.close()

athV = []
athC = []
athW = []
ab = []
ag = []

for i in range(Nb):
    for j in range(NW):
        for k in range(NC):
            for l in range(NV):
                g = g_E[i, j, k, l]
                if np.isfinite(g):
                    ab.append(b[i])
                    athW.append(thW[j])
                    athC.append(thC[k])
                    athV.append(thV[l])
                    ag.append(g)
                g = g_G[i, j, k, l]
                if np.isfinite(g):
                    ab.append(b[i])
                    athW.append(thW[j])
                    athC.append(thC[k])
                    athV.append(thV[l])
                    ag.append(g)
                g = g_H[i, j, k, l]
                if np.isfinite(g):
                    ab.append(b[i])
                    athW.append(thW[j])
                    athC.append(thC[k])
                    athV.append(thV[l])
                    ag.append(g)

ab = np.array(ab)
athW = np.array(athW)
athC = np.array(athC)
athV = np.array(athV)
ag = np.array(ag)

cs = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
ms = ['.', '^', 's', 'v']

thVoC = np.unique((thV[:, None] / thC[None, :]).flat)
NVC = len(thVoC)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for i in range(Nb):
    for k in range(NC):
        good = (ab == b[i]) & (athC == thC[k])
        ax.plot((athV)[good], (ag)[good],
                ls='', color=cs[i], marker=ms[k], alpha=0.2)
    gmed = np.empty(NV)
    for l in range(NV):
        good = (ab == b[i]) & (athV == thV[l])
        if good.any():
            gmed[l] = np.median(ag[good])
        else:
            gmed[l] = np.nan
    ax.plot(thV, gmed, color=cs[i], marker='o')
ax.set_xscale('log')
ax.set_yscale('log')

coeffs = []

aVoC = athV/athC

fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
for i in range(Nb):
    for k in range(NC):
        good = (ab == b[i]) & (athC == thC[k])
        ax2.plot((aVoC)[good], (aVoC**2/ag)[good],
                 ls='', color=cs[i], marker=ms[k], alpha=0.2)
    g2med = np.empty(NVC)
    for l in range(NVC):
        good = (ab == b[i]) & (aVoC == thVoC[l])
        if good.any():
            g2med[l] = np.median((aVoC**2/ag)[good])
        else:
            g2med[l] = np.nan
    fin = np.isfinite(g2med)

    # res = np.polyfit(thVoC[fin], g2med[fin], 2)
    # ax2.plot(thVoC, res[0]*thVoC**2 + res[1]*thVoC + res[2],
    #          lw=3, color=cs[i], alpha=0.5)
    res = np.polyfit(np.sqrt(thVoC[fin]), g2med[fin], 2)
    ax2.plot(thVoC, res[0]*thVoC + res[1]*np.sqrt(thVoC) + res[2],
             lw=3, color=cs[i], alpha=0.5)

    print(res)
    coeffs.append(res)
    ax2.plot(thVoC, g2med, color=cs[i], marker='o')
ax2.set_xscale('log')
ax2.set_yscale('log')
# ax2.set_ylim(0, 30)

coeffs = np.array(coeffs)
print(coeffs.shape)


def f_apbxp(x, a, b, p):
    return a + b*np.power(x, p)


fig5, ax5 = plt.subplots(coeffs.shape[1], 1, figsize=(6, 6))
B = np.linspace(b.min()-1, b.max()+1, 100)
for i in range(coeffs.shape[1]):
    try:
        res = opt.curve_fit(f_apbxp, b, coeffs[:, i], [0.0, 0.0, 0.0])
    except RuntimeError:
        res = (np.zeros(3), )

    print(res[0])
    ax5[i].plot(B, f_apbxp(B, *res[0]))
    # ax5[i].plot(B, 0.5-1.0/B)
    ax5[i].plot(b, coeffs[:, i], marker='.', ls='')

    # ax5[i].set_xscale('log')
    # ax5[i].set_yscale('log')


"""
fig3, ax3 = plt.subplots(1, 1, figsize=(8, 6))
for i in range(Nb):
    for k in range(NC):
        good = (ab == b[i]) & (athC == thC[k]) & (ag < 0.5*athV)
        ax3.plot((athV)[good], (ag)[good],
                 ls='', color=cs[i], marker=ms[k], alpha=0.2)
    gmed = np.empty(NV)
    for l in range(NV):
        good = (ab == b[i]) & (athV == thV[l]) & (ag < 0.5*athV)
        if good.any():
            gmed[l] = np.median(ag[good])
        else:
            gmed[l] = np.nan
    ax3.plot(thV, gmed, color=cs[i], marker='o')
ax3.set_xscale('log')
ax3.set_yscale('log')

fig4, ax4 = plt.subplots(1, 1, figsize=(8, 6))
for i in range(Nb):
    for k in range(NC):
        good = (ab == b[i]) & (athC == thC[k]) & (ag < 0.5*athV)
        ax4.plot((athV/athC)[good], (athV/ag)[good],
                 ls='', color=cs[i], marker=ms[k], alpha=0.2)
    g2med = np.empty(NVC)
    for l in range(NVC):
        good = (ab == b[i]) & (athV/athC == thVoC[l]) & (ag < 0.5*athV)
        if good.any():
            g2med[l] = np.median((athV/ag)[good])
        else:
            g2med[l] = np.nan
    ax4.plot(thVoC, g2med, color=cs[i], marker='o')
ax4.set_xscale('log')
ax4.set_yscale('log')
"""

plt.show()
