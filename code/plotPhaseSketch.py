import math
import numpy as np
import afterglowpy as grb
import matplotlib.pyplot as plt
import paperPlots as pp


thV = 0.0
E0 = 1.0e53
thC = 0.05
thW = 6*thC
b = 6
L0 = 0
ts = 0
q = 0
n0 = 1.0e-3
p = 2.2
epse = 0.1
epsB = 0.001
xiN = 1.0
dL = 1.0e28
z = 0.5454
g0 = 1000

underPanel = False
addG0 = False

highlightPhases = False
highlightTransitions = True

phaseLabels = True

Y0 = np.array([thV, E0, thC, thW, b, L0, ts, q, n0, p, epse, epsB, xiN, dL])
Z = {'z': z, 'spread': False}
Zg = Z.copy()
Zg['g0'] = g0

width = 7.5
height = 3.5

fig = plt.figure(figsize=(width, height))
gs = fig.add_gridspec(1, 3, wspace=0.0)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

axs = [ax1, ax2, ax3]

t = np.geomspace(1.0e3, 3.0e8, 300)
nu = np.empty(t.shape)
nu[:] = 2.41798926e17  # 1 keV

jetType = 4
Y1 = Y0.copy()
Y2 = Y0.copy()
Y3 = Y0.copy()

Y1[0] = 0.0
Y2[0] = thW - 2*thC
Y3[0] = thW + 2*thC

tmin1 = t.min()
tmax1 = t.max()
tmin2 = 1.0e4
tmax2 = t.max()
tmin3 = 1.0e4
tmax3 = t.max()

Fmax = 3e-3
Fmin = 3e-13

Fnu1 = grb.fluxDensity(t, nu, jetType, 0, *Y1, **Z)
Fnu2 = grb.fluxDensity(t, nu, jetType, 0, *Y2, **Z)
Fnu3 = grb.fluxDensity(t, nu, jetType, 0, *Y3, **Z)
Fnu1g = grb.fluxDensity(t, nu, jetType, 0, *Y1, **Zg)
Fnu2g = grb.fluxDensity(t, nu, jetType, 0, *Y2, **Zg)
Fnu3g = grb.fluxDensity(t, nu, jetType, 0, *Y3, **Zg)

al_G_FOA = pp.calcSlopeOffaxis(jetType, Y1, 'G')
al_G_PRE = pp.calcSlopePre(jetType, Y1, 'G')
al_G_POS = pp.calcSlopePost(jetType, Y1, 'G')
al_G_STR1 = pp.calcSlopeStructEff(jetType, Y1, 'G')
al_G_STR2 = pp.calcSlopeStructEff(jetType, Y2, 'G')
al_G_STR3 = pp.calcSlopeStructEff(jetType, Y3, 'G')
be_G = 0.5*(1-p)

al_G_POS2 = -p

print(al_G_FOA, al_G_PRE, al_G_POS)
print(al_G_STR1, al_G_STR2, al_G_STR3)

print(Y1[0], pp.calcThetaEff(jetType, Y1), pp.calcGEff(jetType, Y1))
print(Y2[0], pp.calcThetaEff(jetType, Y2), pp.calcGEff(jetType, Y2))
print(Y3[0], pp.calcThetaEff(jetType, Y3), pp.calcGEff(jetType, Y3))

t1_PRE = np.geomspace(t.min(), 3.0e5, 100)
t1_POS = np.geomspace(1.0e6, t.max(), 100)
t2_PRE = np.geomspace(t.min(), 3.0e5, 100)
t2_STR = np.geomspace(1.0e6, 1.0e7, 100)
t2_POS = np.geomspace(2.0e7, t.max(), 100)
t3_FOA = np.geomspace(t.min(), 1.0e5, 100)
t3_STR = np.geomspace(3.0e5, 2.0e7, 100)
t3_POS = np.geomspace(4.0e7, t.max(), 100)

F1_PRE = Fnu1[np.searchsorted(t, t1_PRE[len(t1_PRE)//2])] * 3
F1_POS = Fnu1[np.searchsorted(t, t1_POS[len(t1_POS)//2])] * 3
F1_POS2 = Fnu1[np.searchsorted(t, t1_POS[len(t1_POS)//2])] / 3

F2_PRE = Fnu2[np.searchsorted(t, t2_PRE[len(t2_PRE)//2])] * 3
F2_STR = Fnu2[np.searchsorted(t, t2_STR[len(t2_STR)//2])] * 3
F2_POS = Fnu2[np.searchsorted(t, t2_POS[len(t2_POS)//2])] * 3
F2_POS2 = Fnu2[np.searchsorted(t, t2_POS[len(t2_POS)//2])] / 3

F3_FOA = Fnu3[np.searchsorted(t, t3_FOA[len(t3_FOA)//2])] * 3
F3_STR = Fnu3[np.searchsorted(t, t3_STR[len(t3_STR)//2])] * 3
F3_POS = Fnu3[np.searchsorted(t, t3_POS[len(t3_POS)//2])] * 3
F3_POS2 = Fnu3[np.searchsorted(t, t3_POS[len(t3_POS)//2])] / 3


buf = 1.1
spanProps = {'color': 'lightgrey',
             'alpha': 0.7,
             'lw': 0}
if highlightPhases:
    ax1.axvspan(t1_PRE.min()/buf, t1_PRE.max()*buf, **spanProps)
    ax1.axvspan(t1_POS.min()/buf, t1_POS.max()*buf, **spanProps)

    ax2.axvspan(t2_PRE.min()/buf, t2_PRE.max()*buf, **spanProps)
    ax2.axvspan(t2_STR.min()/buf, t2_STR.max()*buf, **spanProps)
    ax2.axvspan(t2_POS.min()/buf, t2_POS.max()*buf, **spanProps)

    ax3.axvspan(t3_FOA.min()/buf, t3_FOA.max()*buf, **spanProps)
    ax3.axvspan(t3_STR.min()/buf, t3_STR.max()*buf, **spanProps)
    ax3.axvspan(t3_POS.min()/buf, t3_POS.max()*buf, **spanProps)

if highlightTransitions:
    ax1.axvspan(t1_PRE.max()*buf, t1_POS.min()/buf, **spanProps)

    ax2.axvspan(t2_PRE.max()*buf, t2_STR.min()/buf, **spanProps)
    ax2.axvspan(t2_STR.max()*buf, t2_POS.min()/buf, **spanProps)

    ax3.axvspan(t3_FOA.max()*buf, t3_STR.min()/buf, **spanProps)
    ax3.axvspan(t3_STR.max()*buf, t3_POS.min()/buf, **spanProps)

ax1.plot(t, Fnu1, lw=3)
ax2.plot(t, Fnu2, lw=3)
ax3.plot(t, Fnu3, lw=3)
if addG0:
    ax1.plot(t, Fnu1g)
    ax2.plot(t, Fnu2g)
    ax3.plot(t, Fnu3g)

lineProps = {'lw': 2.5, 'ls': '--', 'color': 'k'}

ax1.plot(t1_PRE, F1_PRE * np.power(t1_PRE/t1_PRE[len(t1_PRE)//2], al_G_PRE),
         **lineProps)
ax1.plot(t1_POS, F1_POS * np.power(t1_POS/t1_POS[len(t1_POS)//2], al_G_POS),
         **lineProps)
ax1.plot(t1_POS, F1_POS2 * np.power(t1_POS/t1_POS[len(t1_POS)//2], al_G_POS2),
         **lineProps)

ax2.plot(t2_PRE, F2_PRE * np.power(t2_PRE/t2_PRE[len(t2_PRE)//2], al_G_PRE),
         **lineProps)
ax2.plot(t2_STR, F2_STR * np.power(t2_STR/t2_STR[len(t2_STR)//2], al_G_STR2),
         **lineProps)
ax2.plot(t2_POS, F2_POS * np.power(t2_POS/t2_POS[len(t2_POS)//2], al_G_POS),
         **lineProps)
ax2.plot(t2_POS, F2_POS2 * np.power(t2_POS/t2_POS[len(t2_POS)//2], al_G_POS2),
         **lineProps)

ax3.plot(t3_FOA, F3_FOA * np.power(t3_FOA/t3_FOA[len(t3_FOA)//2], al_G_FOA),
         **lineProps)
ax3.plot(t3_STR, F3_STR * np.power(t3_STR/t3_STR[len(t3_STR)//2], al_G_STR3),
         **lineProps)
ax3.plot(t3_POS, F3_POS * np.power(t3_POS/t3_POS[len(t3_POS)//2], al_G_POS),
         **lineProps)
ax3.plot(t3_POS, F3_POS2 * np.power(t3_POS/t3_POS[len(t3_POS)//2], al_G_POS2),
         **lineProps)

textProps = {'horizontalalignment': 'right',
             'verticalalignment': 'top',
             'backgroundcolor': 'white'}
ax1.text(0.95, 0.95,
         r"$\theta_{\mathrm{obs}} <  \theta_{\mathrm{C}}$",
         transform=ax1.transAxes, **textProps)
ax2.text(0.95, 0.95,
         r"$\theta_{\mathrm{C}} < \theta_{\mathrm{obs}}$"
         r"$<\theta_{\mathrm{W}}$",
         transform=ax2.transAxes, **textProps)
ax3.text(0.95, 0.95,
         r"$\theta_{\mathrm{W}} < \theta_{\mathrm{obs}}$",
         transform=ax3.transAxes, **textProps)

if phaseLabels:
    textProps = {'horizontalalignment': 'center',
                 'verticalalignment': 'center',
                 'fontsize': 8}

    ax1.text(math.sqrt(max(tmin1, t1_PRE.min()) * min(tmax1, t1_PRE.max())),
             3e-5, "pre-\njet break", **textProps)
    ax1.text(math.sqrt(max(tmin1, t1_POS.min()) * min(tmax1, t1_POS.max())),
             1e-6, "post-\njet break", **textProps)

    ax2.text(math.sqrt(max(tmin2, t2_PRE.min()) * min(tmax2, t2_PRE.max())),
             1e-7, "pre-\njet break", **textProps)
    ax2.text(math.sqrt(max(tmin2, t2_STR.min()) * min(tmax2, t2_STR.max())),
             1e-8, "structured", **textProps)
    ax2.text(math.sqrt(max(tmin2, t2_POS.min()) * min(tmax2, t2_POS.max())),
             1e-7, "post-\njet break", **textProps)

    ax3.text(math.sqrt(max(tmin3, t3_FOA.min()) * min(tmax3, t3_FOA.max())),
             1e-9, "far-\noff-axis", **textProps)
    ax3.text(math.sqrt(max(tmin3, t3_STR.min()) * min(tmax3, t3_STR.max())),
             3e-8, "structured", **textProps)
    ax3.text(math.sqrt(max(tmin3, t3_POS.min()) * min(tmax3, t3_POS.max())),
             3e-8, "post-\njet break", **textProps)

for ax in axs:
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$t$ (s)')
    ax.set_ylim(Fmin, Fmax)

ax1.set_xlim(tmin1, tmax1)
ax2.set_xlim(tmin2, tmax2)
ax3.set_xlim(tmin3, tmax3)

for ax in axs[1:]:
    ax.set_yticklabels([])

ax1.set_ylabel(r'$F_\nu$[1 keV] (mJy)')

fig.tight_layout()

figname = "lc_phase_sketch.pdf"
print("Saving " + figname)
fig.savefig(figname)
plt.close(fig)

if underPanel:
    figu = plt.figure(figsize=(width, 2*height))
    gs = figu.add_gridspec(2, 3, wspace=0.0, hspace=0.0)
    axa1 = figu.add_subplot(gs[0, 0])
    axa2 = figu.add_subplot(gs[0, 1])
    axa3 = figu.add_subplot(gs[0, 2])
    axb1 = figu.add_subplot(gs[1, 0])
    axb2 = figu.add_subplot(gs[1, 1])
    axb3 = figu.add_subplot(gs[1, 2])

    axas = [axa1, axa2, axa3]
    axbs = [axb1, axb2, axb3]
    axs = axas + axbs

    tfac = 1.3
    nfac = 1.3
    Fnu1ta = grb.fluxDensity(t/tfac, nu, jetType, 0, *Y1, **Z)
    Fnu1tb = grb.fluxDensity(t*tfac, nu, jetType, 0, *Y1, **Z)
    Fnu2ta = grb.fluxDensity(t/tfac, nu, jetType, 0, *Y2, **Z)
    Fnu2tb = grb.fluxDensity(t*tfac, nu, jetType, 0, *Y2, **Z)
    Fnu3ta = grb.fluxDensity(t/tfac, nu, jetType, 0, *Y3, **Z)
    Fnu3tb = grb.fluxDensity(t*tfac, nu, jetType, 0, *Y3, **Z)

    Fnu1na = grb.fluxDensity(t, nu/nfac, jetType, 0, *Y1, **Z)
    Fnu1nb = grb.fluxDensity(t, nu*nfac, jetType, 0, *Y1, **Z)
    Fnu2na = grb.fluxDensity(t, nu/nfac, jetType, 0, *Y2, **Z)
    Fnu2nb = grb.fluxDensity(t, nu*nfac, jetType, 0, *Y2, **Z)
    Fnu3na = grb.fluxDensity(t, nu/nfac, jetType, 0, *Y3, **Z)
    Fnu3nb = grb.fluxDensity(t, nu*nfac, jetType, 0, *Y3, **Z)

    al1 = np.log(Fnu1tb/Fnu1ta) / (np.log(tfac**2))
    al2 = np.log(Fnu2tb/Fnu2ta) / (np.log(tfac**2))
    al3 = np.log(Fnu3tb/Fnu3ta) / (np.log(tfac**2))
    be1 = np.log(Fnu1nb/Fnu1na) / (np.log(nfac**2))
    be2 = np.log(Fnu2nb/Fnu2na) / (np.log(nfac**2))
    be3 = np.log(Fnu3nb/Fnu3na) / (np.log(nfac**2))

    axa1.plot(t, al1)
    axa2.plot(t, al2)
    axa3.plot(t, al3)
    axb1.plot(t, be1)
    axb2.plot(t, be2)
    axb3.plot(t, be3)

    amin = -4
    amax = 6
    bmin = -2.0
    bmax = 1.0

    for ax in axs:
        ax.set_xscale('log')
        ax.set_xlim(t.min(), t.max())
    for ax in axas[1:] + axbs[1:]:
        ax.set_yticklabels([])
    for ax in axas:
        ax.axhline(al_G_PRE, ls='-', lw=0.5, color='k')
        ax.axhline(al_G_POS, ls='--', lw=0.5, color='k')
        ax.axhline(al_G_FOA, ls='-.', lw=0.5, color='k')
        ax.set_ylim(amin, amax)
    axa1.axhline(al_G_STR1, ls=':', lw=0.5, color='k')
    axa2.axhline(al_G_STR2, ls=':', lw=0.5, color='k')
    axa3.axhline(al_G_STR3, ls=':', lw=0.5, color='k')
    for ax in axbs:
        ax.axhline(be_G, ls='-', lw=0.5, color='k')
        ax.set_ylim(bmin, bmax)

    axa1.set_ylabel(r'$\alpha$')
    axb1.set_ylabel(r'$\beta$')

    figu.tight_layout()

    figname = "lc_phase_sketch_slopes.pdf"
    print("Saving " + figname)
    figu.savefig(figname)
    plt.close(figu)
