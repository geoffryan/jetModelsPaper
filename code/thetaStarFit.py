import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

f = h5.File("thetaFit.h5", "r")

thV = f['thV'][...]
thC = f['thC'][...]
thW = f['thW'][...]

b = np.array([float(x) for x in f['D'].keys()])

NV = len(thV)
NC = len(thC)
NW = len(thW)
Nb = len(b)

thS_D = np.empty((Nb, NW, NC, NV))
thS_E = np.empty((Nb, NW, NC, NV))
thS_F = np.empty((Nb, NW, NC, NV))
thS_G = np.empty((Nb, NW, NC, NV))
thS_H = np.empty((Nb, NW, NC, NV))

for i, bb in enumerate(b):
    thS_D[i, :, :, :] = f['D'][str(bb)][...]
    thS_E[i, :, :, :] = f['E'][str(bb)][...]
    thS_F[i, :, :, :] = f['F'][str(bb)][...]
    thS_G[i, :, :, :] = f['G'][str(bb)][...]
    thS_H[i, :, :, :] = f['H'][str(bb)][...]
f.close()

athV = []
athC = []
athW = []
ab = []
athS = []

for i in range(Nb):
    for j in range(NW):
        for k in range(NC):
            for l in range(NV):
                thS = thS_E[i, j, k, l]
                if np.isfinite(thS) and thS > 0:
                    ab.append(b[i])
                    athW.append(thW[j])
                    athC.append(thC[k])
                    athV.append(thV[l])
                    athS.append(thS)
                thS = thS_G[i, j, k, l]
                if np.isfinite(thS) and thS > 0:
                    ab.append(b[i])
                    athW.append(thW[j])
                    athC.append(thC[k])
                    athV.append(thV[l])
                    athS.append(thS)
                thS = thS_H[i, j, k, l]
                if np.isfinite(thS) and thS > 0:
                    ab.append(b[i])
                    athW.append(thW[j])
                    athC.append(thC[k])
                    athV.append(thV[l])
                    athS.append(thS)

ab = np.array(ab)
athW = np.array(athW)
athC = np.array(athC)
athV = np.array(athV)
athS = np.array(athS)

cs = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
ms = ['.', '^', 's', 'v']

thVoC = np.unique((thV[:, None] / thC[None, :]).flat)
NVC = len(thVoC)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for i in range(Nb):
    for k in range(NC):
        good = (ab == b[i]) & (athC == thC[k])
        ax.plot((athV)[good], (athS)[good],
                ls='', color=cs[i], marker=ms[k], alpha=0.2)
    thSmed = np.empty(NV)
    for l in range(NV):
        good = (ab == b[i]) & (athV == thV[l])
        if good.any():
            thSmed[l] = np.median(athS[good])
        else:
            thSmed[l] = np.nan
    ax.plot(thV, thSmed, color=cs[i], marker='o')
ax.set_xscale('log')
ax.set_yscale('log')

coeffs = []

fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
for i in range(Nb):
    for k in range(NC):
        good = (ab == b[i]) & (athC == thC[k])
        ax2.plot((athV/athC)[good], (athV/athS)[good],
                 ls='', color=cs[i], marker=ms[k], alpha=0.2)
    thS2med = np.empty(NVC)
    for l in range(NVC):
        good = (ab == b[i]) & (athV/athC == thVoC[l])
        if good.any():
            thS2med[l] = np.median((athV/athS)[good])
        else:
            thS2med[l] = np.nan
    fin = np.isfinite(thS2med)
    # res = np.polyfit(np.sqrt(thVoC)[fin], thS2med[fin], 1)
    # print(res)
    # ax2.plot(thVoC, res[0]*np.sqrt(thVoC) + res[1],
    #          lw=3, color=cs[i], alpha=0.5)
    # res = np.polyfit(np.sqrt(thVoC)[fin], thS2med[fin], 2)
    # print(res)
    # ax2.plot(thVoC, res[0]*thVoC + res[1]*np.sqrt(thVoC) + res[2],
    #          lw=3, color=cs[i], alpha=0.5)
    res = np.polyfit(thVoC[fin], thS2med[fin]**2, 1)
    print(res)
    ax2.plot(thVoC, np.sqrt(res[0]*thVoC + res[1]),
             lw=3, color=cs[i], alpha=0.5)
    # res = np.polyfit(np.sqrt(thVoC)[fin], thS2med[fin]**2, 2)
    # print(res)
    # ax2.plot(thVoC, np.sqrt(res[0]*thVoC + res[1]*np.sqrt(thVoC) + res[2]),
    #         lw=3, color=cs[i], alpha=0.5)

    coeffs.append(res)
    ax2.plot(thVoC, thS2med, color=cs[i], marker='o')
ax2.set_xscale('log')
ax2.set_yscale('log')

coeffs = np.array(coeffs)


def f_apbxp(x, a, b, p):
    return a + b*np.power(x, p)


res0 = opt.curve_fit(f_apbxp, b, coeffs[:, 0], [0.5, -1, -1])
res1 = opt.curve_fit(f_apbxp, b, coeffs[:, 1], [2.0, 1, -1])

fig5, ax5 = plt.subplots(coeffs.shape[1], 1, figsize=(6, 6))
B = np.linspace(b.min()-1, b.max()+1, 100)
ax5[0].plot(B, f_apbxp(B, *res0[0]))
ax5[1].plot(B, f_apbxp(B, *res1[0]))

ax5[0].plot(B, 0.5-1.0/B)
ax5[1].plot(B, 2+2/B)

print(res0[0])
print(res1[0])

ax5[0].plot(b, coeffs[:, 0], marker='.', ls='')
ax5[1].plot(b, coeffs[:, 1], marker='.', ls='')
ax5[0].set_xscale('log')
ax5[0].set_yscale('log')
ax5[1].set_xscale('log')
ax5[1].set_yscale('log')

# for i in range(coeffs.shape[1]):
#     ax5[i].plot(b, coeffs[:, i], marker='.', ls='')

"""
fig3, ax3 = plt.subplots(1, 1, figsize=(8, 6))
for i in range(Nb):
    for k in range(NC):
        good = (ab == b[i]) & (athC == thC[k]) & (athS < 0.5*athV)
        ax3.plot((athV)[good], (athS)[good],
                 ls='', color=cs[i], marker=ms[k], alpha=0.2)
    thSmed = np.empty(NV)
    for l in range(NV):
        good = (ab == b[i]) & (athV == thV[l]) & (athS < 0.5*athV)
        if good.any():
            thSmed[l] = np.median(athS[good])
        else:
            thSmed[l] = np.nan
    ax3.plot(thV, thSmed, color=cs[i], marker='o')
ax3.set_xscale('log')
ax3.set_yscale('log')

fig4, ax4 = plt.subplots(1, 1, figsize=(8, 6))
for i in range(Nb):
    for k in range(NC):
        good = (ab == b[i]) & (athC == thC[k]) & (athS < 0.5*athV)
        ax4.plot((athV/athC)[good], (athV/athS)[good],
                 ls='', color=cs[i], marker=ms[k], alpha=0.2)
    thS2med = np.empty(NVC)
    for l in range(NVC):
        good = (ab == b[i]) & (athV/athC == thVoC[l]) & (athS < 0.5*athV)
        if good.any():
            thS2med[l] = np.median((athV/athS)[good])
        else:
            thS2med[l] = np.nan
    ax4.plot(thVoC, thS2med, color=cs[i], marker='o')
ax4.set_xscale('log')
ax4.set_yscale('log')
"""

plt.show()
