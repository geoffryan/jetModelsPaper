import math
import numpy as np
import matplotlib.pyplot as plt
# import afterglowpy as grb


def g2al(g, p):
    al = (3 - 6*p + 3*g) / (8+g)
    return al


def al2g(al, p):
    g = (3 - 6*p - 8*al) / (al-3)
    return g


def al2eg(al, p, eal, ep):
    dgdp = -6/(al-3)
    dgdal = (21 + 6*p) / ((al-3)*(al-3))

    eg = math.sqrt(dgdp**2 * ep**2 + dgdal**2 * eal**2)

    return eg


def th2g_pl(thVoC, b):
    thEffoC = thVoC / np.sqrt(1.83+2.10*np.power(b, -1.25)
                              + (0.486-0.862*np.power(b, -1.15)) * thVoC)
    g = -(thEffoC-thVoC) * thEffoC / (1 + thEffoC*thEffoC/b)
    return g


def tb2thp(tb, Eon):
    thVpC = 0.3 * np.power(tb/6.54, 0.375) * np.power(Eon, -0.125)
    return thVpC


def tb2ethp(tb, Eon, etb, elEon):
    thVpC = tb2thp(tb, Eon)
    dthpdtb = 0.375*thVpC/tb
    dthpdEon = -0.125*thVpC/Eon
    dthpdlEon = math.log(10.0) * Eon * dthpdEon

    ethVpC = math.sqrt((dthpdtb*etb)**2 + (dthpdlEon*elEon)**2)
    return ethVpC


def po2C(thVpC, thVoC):
    thC = thVpC / (1+thVoC)
    return thC


def po2eC(thVpC, thVoC, ethVpC, ethVoC):
    dCdp = 1.0 / (1+thVoC)
    dCdo = -thVpC / (1+thVoC)**2

    ethC = np.sqrt((dCdp*ethVpC)**2 + (dCdo*ethVoC)**2)

    return ethC


def po2V(thVpC, thVoC):
    thC = thVpC * thVoC / (1+thVoC)
    return thC


def po2eV(thVpC, thVoC, ethVpC, ethVoC):
    dVdp = thVoC / (1+thVoC)
    dVdo = thVpC / (1+thVoC)**2

    ethV = np.sqrt((dVdp*ethVpC)**2 + (dVdo*ethVoC)**2)

    return ethV


be = -0.585
ebe = 0.005

al = 0.90
eal = 0.06

tb = 164
etb = 12

Eon = 20.0/10.0
# Eon = 30
elEon = math.sqrt(2) * 1.0

p = 1-2*be
ep = 2*ebe

g = al2g(al, p)
eg = al2eg(al, p, eal, ep)

thVpC = tb2thp(tb, Eon)
ethVpC = tb2ethp(tb, Eon, etb, elEon)

thVoC_g = math.sqrt(4*g)
ethVoC_g = 0.5*thVoC_g*eg/g

athVoC = np.linspace(1.0, 40, 10000)
bs = [4, 6, 8, 9, 10]
thVoC_pl = []
ethVoC_pl = []
fig, ax = plt.subplots(1, 1)
for b in bs:
    g_pl = th2g_pl(athVoC, b)
    ax.plot(athVoC, g_pl)

    i = np.searchsorted(g_pl, g)
    dgdth = (g_pl[i]-g_pl[i-1]) / (athVoC[i] - athVoC[i-1])

    thVoC_pl.append(0.5*(athVoC[i-1] + athVoC[i]))
    ethVoC_pl.append(eg / dgdth)

thVoC_pl = np.array(thVoC_pl)
ethVoC_pl = np.array(ethVoC_pl)


thC_g = po2C(thVpC, thVoC_g)
ethC_g = po2eC(thVpC, thVoC_g, ethVpC, ethVoC_g)
thC_pl = po2C(thVpC, thVoC_pl)
ethC_pl = po2eC(thVpC, thVoC_pl, ethVpC, ethVoC_pl)

thV_g = po2V(thVpC, thVoC_g)
ethV_g = po2eV(thVpC, thVoC_g, ethVpC, ethVoC_g)
thV_pl = po2V(thVpC, thVoC_pl)
ethV_pl = po2eV(thVpC, thVoC_pl, ethVpC, ethVoC_pl)

# plt.show()

print("beta  = {0:.3f} +/- {1:.3f}".format(be, ebe))
print("alpha = {0:.3f} +/- {1:.3f}".format(al, eal))
print("tb    = {0:.1f} +/- {1:.1f} days".format(tb, etb))
print("p     = {0:.3f} +/- {1:.3f}".format(p, ep))
print("g     = {0:.3f} +/- {1:.3f}".format(g, eg))
print("thV+thC = {0:.3f} +/- {1:.3f} rad".format(thVpC, ethVpC))

print("thV/thC (Gaussian) = {0:.3f} +/- {1:.3f}".format(thVoC_g, ethVoC_g))
for i, b in enumerate(bs):
    print("thV/thC (PL {0:.1f}) = {1:.3f} +/- {2:.3f}".format(b, thVoC_pl[i],
          ethVoC_pl[i]))

print("thV (Gaussian) = {0:.3f} +/- {1:.3f} rad".format(thV_g, ethV_g))
for i, b in enumerate(bs):
    print("thV (PL {0:.1f}) = {1:.3f} +/- {2:.3f}".format(b, thV_pl[i],
          ethV_pl[i]))
print("thC (Gaussian) = {0:.3f} +/- {1:.3f} rad".format(thC_g, ethC_g))
for i, b in enumerate(bs):
    print("thC (PL {0:.1f}) = {1:.3f} +/- {2:.3f}".format(b, thC_pl[i],
          ethC_pl[i]))
