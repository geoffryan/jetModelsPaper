import numpy as np
import matplotlib.pyplot as plt
import grbpy as grb


jetType = 0

thV = 0.5
thC = 0.1
thW = 0.4
E0 = 1.0e52
n0 = 1.0e-2
p = 2.3
epse = 0.1
epsB = 0.001
dL = 1.0e27


t = np.logspace(3, 8, 200)
nu = np.empty(t.shape)
nu[:] = 1.0e14

Y = np.array([thV, E0, thC, thW, 0, 0, 0, n0, p, epse, epsB, 1.0, dL])

Fnu = grb.fluxDensity(t, nu, jetType, 0, *Y)

Nth = 10

FnuR = np.empty((Nth, t.shape[0]))
dth = thW/Nth

for i in range(Nth):
    thout = thW - i*dth
    thin = thW - (i+1)*dth
    if thout < 0.0:
        thin = 0.0
    x = float(i)/(Nth-1)
    th = x*thin+(1-x)*thout
    E = E0*np.exp(-0.5*th*th/(thC*thC))
    Y[1] = E
    Y[2] = thin
    Y[3] = thout
    FnuR[i,:] = grb.fluxDensity(t, nu, -2, 0, *Y)

fig, ax =  plt.subplots(1,1)
for i in range(Nth):
    ax.plot(t, FnuR[i])
ax.plot(t, Fnu, color='k')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$t$ (s)')
ax.set_ylabel(r'$F_{\nu}$ (erg/cm$^2$s Hz)')

ax.set_ylim(1.0e-12, None)

name = "jetDecomp.pdf"

print("Saving "+name)
fig.savefig(name)
