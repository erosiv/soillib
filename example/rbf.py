#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from scipy.stats.qmc import Halton


rng = np.random.default_rng()
xobs = 2*Halton(2, seed=rng).random(100) - 1

# print(xobs)

yobs = np.sum(xobs, axis=1)*np.exp(-6*np.sum(xobs**2, axis=1))
rbf = RBFInterpolator(xobs, yobs)

xgrid = np.mgrid[-1:1:50j, -1:1:50j]
xflat = xgrid.reshape(2, -1).T
yflat = rbf(xflat)
ygrid = yflat.reshape(50, 50)

fig, ax = plt.subplots()
ax.pcolormesh(*xgrid, ygrid, vmin=-0.25, vmax=0.25, shading='gouraud')
p = ax.scatter(*xobs.T, c=yobs, s=50, ec='k', vmin=-0.25, vmax=0.25)
fig.colorbar(p)
plt.show()