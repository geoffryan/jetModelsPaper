import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
# import grbpy as grb
import tbScalings as tbpy

pltDir = "scalingsGaussian/"

regime = 'G'
nu = 1.0e16
jetModel = 0
b = 0
Y = np.array([0.6, 1.0e53, 0.1, 0.4, b, 0.0, 0.0, 0.0, 1.0, 2.2,
              0.1, 1.0e-6, 1.0, 1.0e26])
spread = False
n = 3

NW = 3
NV = 5
NC = 3

panelSize = 2

thVs = np.linspace(0.0, 1.0, NV)
thCs = np.linspace(0.05, 0.2, NC)
thWs = np.linspace(0.1, 0.8, NW)

tb0 = np.empty((NV, NC, NW))
tb1 = np.empty((NV, NC, NW))
al0 = np.empty((NV, NC, NW))
al1 = np.empty((NV, NC, NW))
al2 = np.empty((NV, NC, NW))

color = ['C{0:d}'.format(i) for i in range(10)]
ls = ['-', '--', '-.', ':']

for k in range(NW):
    fig = plt.figure(figsize=(NC*2, NV*2))
    gs0 = fig.add_gridspec(NV, NC)

    thW = thWs[k]

    for i in range(NV):
        thV = thVs[i]
        for j in range(NC):
            thC = thCs[j]
            print("thW = {0:.3f} thV={0:.3f} thC={1:.3f}".format(
                    thW, thV, thC))
            gs = gs0[i, j].subgridspec(4, 1, hspace=0)
            ax0 = fig.add_subplot(gs[:-1])
            ax1 = fig.add_subplot(gs[-1])
            ax = [ax0, ax1]

            Y[0] = thV
            Y[2] = thC
            Y[3] = thW
            _, t0, t1, a0, a1, a2 = tbpy.findBreaks(
                regime, nu, jetModel, Y, 3, spread,
                True,  ax, fig, printMode='summary')
            tb0[i, j,  k] = t0
            tb1[i, j,  k] = t1
            al0[i, j,  k] = a0
            al1[i, j,  k] = a1
            al2[i, j,  k] = a2
    fig.tight_layout()

    figname = pltDir + "lc_grid_thW={0:.3}.png".format(thW)
    print("Saving " + figname)
    fig.savefig(figname, dpi=200)
    plt.close(fig)

archiveName = pltDir + "closureFits.h5"
f = h5.File(archiveName, "w")
f.create_dataset("Y", data=Y)
f.create_dataset("nu", data=np.array([nu]))
f.create_dataset("thV", data=thVs)
f.create_dataset("thC", data=thCs)
f.create_dataset("thW", data=thWs)
f.create_dataset("tb0", data=tb0)
f.create_dataset("tb1", data=tb1)
f.create_dataset("al0", data=al0)
f.create_dataset("al1", data=al1)
f.create_dataset("al2", data=al2)
f.close()

fig, ax = plt.subplots(1, 1)
for k in range(NW):
    for i in range(NC):
        ax.plot(thVs, tb0[:, i, k], color=color[i], ls=ls[k])
ax.set_yscale('log')
ax.set_xlabel(r'$\theta_{\mathrm{obs}}$')
ax.set_ylabel(r'$t_{b, 0}$')
fig.tight_layout()
figname = pltDir + "tb0_thV.pdf"
print("Saving " + figname)
fig.savefig(figname)
plt.close()

fig, ax = plt.subplots(1, 1)
for k in range(NW):
    for i in range(NC):
        ax.plot(thVs, tb1[:, i, k], color=color[i], ls=ls[k])
ax.set_yscale('log')
ax.set_xlabel(r'$\theta_{\mathrm{obs}}$')
ax.set_ylabel(r'$t_{b, 1}$')
fig.tight_layout()
figname = pltDir + "tb1_thV.pdf"
print("Saving " + figname)
fig.savefig(figname)
plt.close()

fig, ax = plt.subplots(1, 1)
for k in range(NW):
    for i in range(NC):
        ax.plot(thVs, al0[:, i, k], color=color[i], ls=ls[k])
ax.set_xlabel(r'$\theta_{\mathrm{obs}}$')
ax.set_ylabel(r'$\alpha_0$')
fig.tight_layout()
figname = pltDir + "al0_thV.pdf"
print("Saving " + figname)
fig.savefig(figname)
plt.close()

fig, ax = plt.subplots(1, 1)
for k in range(NW):
    for i in range(NC):
        ax.plot(thVs, al1[:, i, k], color=color[i], ls=ls[k])
ax.set_xlabel(r'$\theta_{\mathrm{obs}}$')
ax.set_ylabel(r'$\alpha_1$')
fig.tight_layout()
figname = pltDir + "al1_thV.pdf"
print("Saving " + figname)
fig.savefig(figname)
plt.close()

fig, ax = plt.subplots(1, 1)
for k in range(NW):
    for i in range(NC):
        ax.plot(thVs, al2[:, i, k], color=color[i], ls=ls[k])
ax.set_xlabel(r'$\theta_{\mathrm{obs}}$')
ax.set_ylabel(r'$\alpha_2$')
fig.tight_layout()
figname = pltDir + "al2_thV.pdf"
print("Saving " + figname)
fig.savefig(figname)
plt.close()

fig, ax = plt.subplots(1, 1)
for k in range(NW):
    for i in range(NC):
        ax.plot(thVs/thCs[i], al1[:, i, k], color=color[i], ls=ls[k])
ax.set_xlabel(r'$\theta_{\mathrm{obs}}$')
ax.set_ylabel(r'$\alpha_1$')
fig.tight_layout()
figname = pltDir + "al1_thVothC.pdf"
print("Saving " + figname)
fig.savefig(figname)
plt.close()
